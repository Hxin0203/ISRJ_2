"""CPU-only shape/pairing and temporary-file identity regressions."""
import numpy as np
import pytest

from tools.vp_1103_recovp_hybrid_evaluate import _scaled_states
from tools import vp_1103_recovp_hybrid_evaluate as evaluator


def test_complex_scale_preserves_parent_variant_state_and_pulse_pairing():
    shape = (2, 3, 8, 1800)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000
    s, j, e = values + 1j, values * 2 - 3j, values * .25 + 2j
    factors = np.array([[.2j, 2, -3j], [-1, 4j, 5]], dtype=np.complex64)

    actual = _scaled_states(s, j, e, factors)

    assert actual.shape == (6, 4, 8, 1800)
    for parent in range(2):
        for variant in range(3):
            states = np.stack((e[parent, variant], s[parent, variant] + e[parent, variant],
                               j[parent, variant] + e[parent, variant],
                               s[parent, variant] + j[parent, variant] + e[parent, variant]))
            np.testing.assert_allclose(actual[parent * 3 + variant],
                                       states * factors[parent, variant])


def test_scalar_scale_and_invalid_scale_shape():
    components = np.ones((2, 3, 8, 1800), dtype=np.complex64)
    np.testing.assert_allclose(_scaled_states(components, components, components, .1),
                               _scaled_states(components, components, components, 1) * .1)
    with pytest.raises(ValueError, match="scale must be scalar"):
        _scaled_states(components, components, components, np.ones((2, 1)))


def test_metric_batch_passes_all_four_states_and_keeps_parent_variant_means(monkeypatch):
    components = np.ones((2, 3, 8, 1800), dtype=np.complex64)
    restored = np.ones((2, 3, 4, 8, 1800), dtype=np.complex64)
    calls = []

    def radar_contract(s, observed, output, doppler):
        assert s.shape == (6, 8, 1800)
        assert observed.shape == output.shape == (6, 4, 8, 1800)
        np.testing.assert_array_equal(observed[:, :, 0, 0], np.tile([1, 2, 2, 3], (6, 1)))
        np.testing.assert_array_equal(doppler, [10, 10, 10, 20, 20, 20])
        calls.append(True)
        return {"radar_probe": np.arange(6, dtype=float)}

    monkeypatch.setattr(evaluator, "group_metrics", lambda *args: {})
    monkeypatch.setattr(evaluator, "radar_metrics", radar_contract)
    values = evaluator._metric_batch(components, components, components,
                                     restored, restored, np.array([10, 20]))
    assert len(calls) == 2
    np.testing.assert_array_equal(values["hybrid__radar_probe"], [1, 4])
    np.testing.assert_array_equal(values["stage1__radar_probe"], [1, 4])


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    for name, filename in (("CONFIG", "CONFIG.json"), ("PREFLIGHT", "PREFLIGHT.json"),
                           ("DATA", "dev.npz"), ("DATA_RECEIPT", "dev.receipt.json"),
                           ("CHECKPOINT", "epoch_256.pt"), ("COMPLETE", "TRAINING_COMPLETE.json")):
        monkeypatch.setattr(evaluator, name, tmp_path / filename)
    evaluator.DATA.write_bytes(b"original synthetic DEV content")
    evaluator.write(evaluator.DATA_RECEIPT, {"sha256": evaluator.sha(evaluator.DATA)})
    data = {"train.npz": "frozen-train-sha", "train.receipt.json": "frozen-train-receipt-sha",
            "dev.npz": evaluator.sha(evaluator.DATA),
            "dev.receipt.json": evaluator.sha(evaluator.DATA_RECEIPT)}
    preflight = {"stage": "PREFLIGHT_PASSED", "status": "PASS", "seed": 17,
                 "scope": "TRAIN_AND_DEV_ONLY", "data_sha256": data}
    evaluator.write(evaluator.PREFLIGHT, preflight)
    config = {"run_id": evaluator.RUN, "seed": 17, "epochs": 256,
              "source_sha256": {}, "data_sha256": data,
              "preflight_sha256": evaluator.sha(evaluator.PREFLIGHT)}
    evaluator.write(evaluator.CONFIG, config)
    identity = {"run": evaluator.RUN, "seed": 17, "source_sha256": {},
                "config_sha256": evaluator.sha(evaluator.CONFIG),
                "preflight_sha256": evaluator.sha(evaluator.PREFLIGHT),
                "train_sha256": data["train.npz"]}
    return {"stage": "TRAINING_COMPLETE", "seed": 17, "epochs": 256, "identity": identity}


def test_matched_live_dev_and_receipt_replacement_rejected_before_numpy_load(frozen_run, monkeypatch):
    expected = evaluator.frozen_data_bindings(frozen_run)
    assert expected["dev.npz"] == evaluator.sha(evaluator.DATA)
    evaluator.DATA.write_bytes(b"replacement synthetic DEV content")
    evaluator.write(evaluator.DATA_RECEIPT, {"sha256": evaluator.sha(evaluator.DATA)})

    def forbidden_load(*args, **kwargs):
        pytest.fail("np.load must not run before frozen DEV binding is verified")

    monkeypatch.setattr(evaluator.np, "load", forbidden_load)
    with pytest.raises(RuntimeError, match="live DEV data/receipt differs"):
        evaluator.load_dev(frozen_run)


@pytest.mark.parametrize("record", ["CONFIG", "PREFLIGHT"])
def test_frozen_record_replacement_rejected_by_checkpoint_sha(frozen_run, record):
    path = getattr(evaluator, record)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="CONFIG/PREFLIGHT SHA"):
        evaluator.frozen_data_bindings(frozen_run)


def test_completion_must_record_all_256_epochs(frozen_run):
    evaluator.CHECKPOINT.write_bytes(b"synthetic checkpoint envelope")
    evaluator.write(evaluator.CHECKPOINT.with_suffix(".json"), {})
    frozen_run["epochs"] = 255
    evaluator.write(evaluator.COMPLETE, frozen_run)
    with pytest.raises(RuntimeError, match="training completion record missing or invalid"):
        evaluator.validate_checkpoint({})
