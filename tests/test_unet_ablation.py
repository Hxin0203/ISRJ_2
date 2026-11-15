"""Scientific-control and crash-window tests; synthetic tensors are engineering only."""
import copy
import numpy as np
import pytest
import torch

from paper2.network import CorrectionNet, training_loss
from paper2.train_1103 import batches
from paper2.train_256_8seed import seed_rng, rng_snapshot, restore_rng, optimizer_for
from tools.unet_ablation.models import make_model, objective
from tools.unet_ablation.train import FORMAT, choose_resume, publish_checkpoint, valid_checkpoint, minibatch_update

torch.set_num_threads(4)


def test_local_training_is_refused_and_budget_survives_restart(tmp_path, monkeypatch):
    from tools.unet_ablation import runtime
    from tools.durable_io import atomic_json
    monkeypatch.setattr(runtime.platform, "node", lambda: "LOCAL-NOT-SERVER")
    with pytest.raises(RuntimeError, match="Local training is prohibited"):
        runtime.require_server()
    monkeypatch.setattr(runtime.time, "time", lambda: 200.0)
    atomic_json(tmp_path / "BUDGET.json", {"spent_seconds": 10.0, "active_since_unix": 100.0,
                                           "limit_seconds": runtime.BUDGET_SECONDS})
    state = runtime.begin_budget(tmp_path)
    assert state["spent_seconds"] == 110.0
    assert runtime.budget_remaining(state) == runtime.BUDGET_SECONDS - 110.0
    (tmp_path / "STOP").touch()
    with pytest.raises(runtime.UserStop):
        runtime.user_control(tmp_path)


def test_parameter_budget_has_no_dummy_padding_and_handles_odd_length():
    model = make_model("unet")
    assert sum(p.numel() for p in model.parameters()) == 59714
    for n in (1800, 1799):
        model.zero_grad(set_to_none=True)
        y = model(torch.randn(2, 2, n))
        assert y.shape == (2, 2, n) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    original = CorrectionNet()
    original(torch.randn(1, 2, 1800)).sum().backward()
    assert sum(p.numel() for p in original.parameters() if p.grad is not None) == 59714


def test_same_seed_structural_ablation_preserves_initial_state_and_only_changes_preprocessing():
    seed_rng(17)
    full = make_model("full")
    x = torch.randn(2, 2, 1800)
    for variant in ("no_dechirp", "no_rms"):
        seed_rng(17)
        model = make_model(variant)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in full.state_dict().items())
        # Turn the sole intervention back on: exact original output must return.
        model.variant = "full"
        torch.testing.assert_close(model(x), full(x), rtol=0, atol=0)
        model.variant = variant
        assert not torch.allclose(model(x), full(x))
    # Batch shuffling uses an independent epoch seed regardless of model RNG use.
    a = list(batches(512, 12, 17))
    torch.randn(1000)
    b = list(batches(512, 12, 17))
    assert len(a) == 48 and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert sorted(np.concatenate(a)) == list(range(1536))


def test_loss_removal_preserves_exact_remaining_objective_and_gradients():
    s, j, e = (torch.randn(3, 2, 80) for _ in range(3))
    c = torch.randn(3, 4, 2, 80, requires_grad=True)
    full = training_loss(c, s, j, e, "B4")
    for variant, expression in (("unet", full["total"]),
                                ("no_pc", full["endpoint"] + full["damage"]),
                                ("no_damage", full["endpoint"] + .1 * full["pc"])):
        actual = objective(c, s, j, e, variant)
        torch.testing.assert_close(actual["total"], expression)
        a = torch.autograd.grad(actual["total"], c, retain_graph=True)[0]
        b = torch.autograd.grad(expression, c, retain_graph=True)[0]
        torch.testing.assert_close(a, b)


def test_microbatch_accumulation_matches_full_effective_batch():
    rng = np.random.default_rng(77)
    arrays = {k: (rng.normal(size=(2, 3, 8, 40)) + 1j * rng.normal(size=(2, 3, 8, 40))).astype(np.complex64)
              for k in ("s", "j", "epsilon")}
    a = make_model("unet")
    b = copy.deepcopy(a)
    opt_a, opt_b = optimizer_for(a), optimizer_for(b)
    ids = np.arange(6)
    ra = minibatch_update(a, opt_a, arrays, ids, "unet", "cpu", 6)
    rb = minibatch_update(b, opt_b, arrays, ids, "unet", "cpu", 2)
    for key in ra:
        assert ra[key] == pytest.approx(rb[key], rel=2e-5, abs=2e-6)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-4, atol=2e-6)


def small_payload(net, optimizer, epoch, identity):
    return {"format": FORMAT, "identity": identity, "epoch": epoch, "seed": 17, "global_step": epoch * 48,
            "model": copy.deepcopy(net.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": rng_snapshot(), "epoch_record": {"epoch": epoch, "loss": float(epoch)}}


def test_newest_archive_recovers_from_torn_latest_and_rejects_wrong_identity(tmp_path):
    net = torch.nn.Linear(2, 2)
    opt = optimizer_for(net)
    identity = {"seed": 17, "variant": "unet"}
    first = small_payload(net, opt, 31, identity)
    second = small_payload(net, opt, 32, identity)
    publish_checkpoint(tmp_path, first)
    publish_checkpoint(tmp_path, second)
    (tmp_path / "latest.pt").write_bytes(b"torn checkpoint")
    assert choose_resume(tmp_path, identity)["epoch"] == 32
    assert valid_checkpoint(tmp_path / "epoch_032.pt", {"seed": 29}) is None
    before = (tmp_path / "epoch_032.pt").read_bytes()
    publish_checkpoint(tmp_path, second)
    assert before == (tmp_path / "epoch_032.pt").read_bytes()
    bad = copy.deepcopy(second)
    bad["model"]["weight"] += 1
    with pytest.raises(RuntimeError, match="immutable"):
        publish_checkpoint(tmp_path, bad)


def test_optimizer_rng_resume_matches_uninterrupted_next_update(tmp_path):
    seed_rng(41)
    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Dropout(.2), torch.nn.Linear(7, 2))
    opt = optimizer_for(model)
    def update(net, optimizer):
        x = torch.randn(8, 3)
        optimizer.zero_grad(set_to_none=True)
        net(x).square().mean().backward()
        optimizer.step()
    update(model, opt)
    saved = small_payload(model, opt, 1, {"seed": 17})
    publish_checkpoint(tmp_path, saved)
    update(model, opt)
    resumed = copy.deepcopy(model)
    recovered = choose_resume(tmp_path, {"seed": 17})
    resumed.load_state_dict(recovered["model"])
    resumed_opt = optimizer_for(resumed)
    resumed_opt.load_state_dict(recovered["optimizer"])
    restore_rng(recovered["rng"])
    update(resumed, resumed_opt)
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[k], rtol=0, atol=0)
