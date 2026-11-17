"""Server-only numerical and persistence checks; no DEV performance tuning."""
from __future__ import annotations

import hashlib
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read, sha
from paper2.train_1103 import load_arrays, make_batch
from paper2.train_256_8seed import optimizer_for, seed_rng, rng_snapshot, restore_rng
from tools.durable_io import atomic_json
from tools.unet_ablation.models import PlainUNet, objective
from .model import NoiseReferenceUNet, observed_scale
from .runtime import require_server, configure_precision


def _summary(v):
    v = v.detach().cpu().numpy().ravel()
    return {"mean": float(v.mean()), "q05": float(np.quantile(v, .05)),
            "median": float(np.median(v)), "q95": float(np.quantile(v, .95)),
            "max": float(v.max())}


def _parameters_sha(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                   for p in model.parameters())).hexdigest()


def preflight(base):
    require_server()
    from .train import local_sources, publish, choose_resume, FORMAT, DATA, ABLATION_BASE
    sources = local_sources()
    path = base / "NUMERICAL_PREFLIGHT.json"
    if path.exists():
        saved = read(path)
        if saved["stage"] != "NUMERICAL_PREFLIGHT_PASSED" or saved["source_sha256"] != sources:
            raise RuntimeError("Preflight exists for different sources; preserve and use a new run")
        return saved
    precision = configure_precision()
    torch.set_num_threads(4)
    seed_rng(2026111704)
    noise = torch.randn(2048, 2, 1800) / np.sqrt(2)
    estimate = observed_scale(noise)
    assert torch.isfinite(estimate).all() and abs(float(estimate.square().mean()) - 1) < .05
    scale_error = max(float(((observed_scale(noise * a) / a - estimate).abs()
                             / estimate).max()) for a in (.01, 100))
    assert scale_error < 2e-5
    assert torch.isfinite(observed_scale(torch.zeros(2, 2, 1800))).all()
    initialization = {}
    for seed in (17, 29, 41):
        seed_rng(seed)
        control = PlainUNet()
        seed_rng(seed)
        model = NoiseReferenceUNet()
        assert control.state_dict().keys() == model.state_dict().keys()
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
        initial_cp = ABLATION_BASE / "unet" / f"seed_{seed}" / "recovery/epoch_000.pt"
        digest = sha(initial_cp)
        assert digest == read(initial_cp.with_suffix(".json"))["checkpoint_sha256"]
        original = torch.load(initial_cp, map_location="cpu", weights_only=False)
        assert all(torch.equal(v, original["model"][k]) for k, v in model.state_dict().items())
        initialization[str(seed)] = {"parameter_bytes_sha256": _parameters_sha(model),
                                    "baseline_epoch0_sha256": digest, "exact_match": True}
    model = model.cuda()
    x = noise[:4].cuda()
    with torch.no_grad():
        y = model(x)
        net_error = float((model(x * 10) / 10 - y).abs().max())
    assert net_error < 2e-5
    optimizer = optimizer_for(model)
    arrays = load_arrays(DATA / "train.npz")
    batch = make_batch(arrays, np.arange(2), "cuda")

    def update(net, opt):
        opt.zero_grad(set_to_none=True)
        correction = net(batch["y"]).reshape(-1, 4, 2, 1800)
        losses = objective(correction, batch["s"], batch["j"], batch["e"], "unet")
        losses["total"].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
        opt.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    first = update(model, optimizer)
    active = sum(p.numel() for p in model.parameters() if p.grad is not None)
    assert active == sum(p.numel() for p in model.parameters()) == 59714
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preflight_", dir=base) as scratch:
        folder = Path(scratch)
        identity = {"scope": "ENGINEERING_PREFLIGHT_ONLY", "source_sha256": sources}
        payload = {"format": FORMAT, "identity": identity, "seed": 41, "epoch": 1,
                   "global_step": 48, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng": rng_snapshot(), "epoch_record": first}
        digest = publish(folder, payload)
        saved = choose_resume(folder, identity)
        assert saved is not None and digest == sha(folder / "recovery/epoch_001.pt")
        resumed = NoiseReferenceUNet().cuda()
        resumed.load_state_dict(saved["model"], strict=True)
        assert all(torch.equal(p.cpu(), saved["model"][k]) for k, p in resumed.state_dict().items())
        opt_resumed = optimizer_for(resumed)
        opt_resumed.load_state_dict(saved["optimizer"])
        state_resumed = opt_resumed.state_dict()
        assert state_resumed["param_groups"] == saved["optimizer"]["param_groups"]
        for parameter, state in saved["optimizer"]["state"].items():
            for key, value in state.items():
                restored = state_resumed["state"][parameter][key]
                assert torch.equal(value.cpu(), restored.cpu()) if torch.is_tensor(value) else value == restored
        restore_rng(saved["rng"])
        expected_draw = torch.rand(4, device="cuda")
        second = update(model, optimizer)
        restore_rng(saved["rng"])
        assert torch.equal(expected_draw, torch.rand(4, device="cuda"))
        second_resumed = update(resumed, opt_resumed)
        resume_error = max(float((p - resumed.state_dict()[k]).abs().max())
                           for k, p in model.state_dict().items())
        loss_resume_error = max(abs(second[k] - second_resumed[k]) for k in second)
        assert resume_error <= 1e-6 and loss_resume_error <= 1e-6, {"resume_error": resume_error, "loss_error": loss_resume_error}
    # Labels enter this offline contamination diagnostic only, never the model.
    with torch.no_grad():
        small = [torch.from_numpy(np.stack((arrays[k][:32].real, arrays[k][:32].imag), axis=-2)
                                  .astype(np.float32)).reshape(-1, 2, 1800)
                 for k in ("s", "j", "epsilon")]
        s, j, e = small
        e_scale = observed_scale(e)
        contamination = {"noise_proxy_over_realized_noise_rms": _summary(e_scale / e.square().sum(1).mean(-1).sqrt()[:, None, None]),
                         "signal_plus_noise_proxy_over_noise_proxy": _summary(observed_scale(s + e) / e_scale),
                         "jammer_plus_noise_proxy_over_noise_proxy": _summary(observed_scale(j + e) / e_scale)}
    result = {"stage": "NUMERICAL_PREFLIGHT_PASSED", "source_sha256": sources,
              "created_utc": now(), "host": platform.node(), "physical_gpu": 0,
              "torch": torch.__version__, "python": platform.python_version(),
              "cuda_device_properties": str(torch.cuda.get_device_properties(0)),
              "precision": precision, "parameters_total_and_active": active,
              "noise_monte_carlo": {"waveforms": 2048, "complex_noise_power": 1,
                                    "estimated_power": _summary(estimate.square())},
              "scale_relative_error": scale_error, "network_scale_max_absolute_error": net_error,
              "baseline_initialization": initialization, "finite_forward_backward": True,
              "recovery_model_max_absolute_error": resume_error, "recovery_rng_exact": True,
              "recovery_loaded_model_and_optimizer_exact": True,
              "recovery_next_loss_max_absolute_error": loss_resume_error,
              "recovery_numeric_tolerance_absolute": 1e-6,
              "first_preflight_batch_losses": first, "contamination_train_first32parents": contamination,
              "scope": "engineering checks and fixed-estimator leakage diagnostics; no DEV performance consumed; no tuning"}
    atomic_json(path, result)
    atomic_json(ROOT / "evidence/outputs" / base.name / path.name, result)
    del model, resumed, optimizer, opt_resumed, batch, arrays
    torch.cuda.empty_cache()
    return result
