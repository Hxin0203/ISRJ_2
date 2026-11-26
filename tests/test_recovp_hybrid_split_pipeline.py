import json

import pytest

from tools import server_gpu_admission
from tools import start_recovp_hybrid_extended as launcher
from tools.recovp_hybrid_seed_schedule import (
    EPOCHS,
    GPU_LANES,
    SEED_GPU,
    assigned_gpu,
    validate_completion,
)


def test_remaining_seeds_are_partitioned_once_across_fixed_gpu_lanes():
    assert GPU_LANES == {1: (53, 67, 79), 0: (97, 109)}
    assert set(SEED_GPU) == {53, 67, 79, 97, 109}
    assert len(SEED_GPU) == len(set(SEED_GPU))
    assert {seed: assigned_gpu(seed) for seed in SEED_GPU} == SEED_GPU
    with pytest.raises(ValueError, match="not in the remaining"):
        assigned_gpu(41)


def test_launcher_observes_only_the_seed_assigned_gpu(monkeypatch):
    samples = [
        {"physical_gpu_index": 0, "gpu_uuid": server_gpu_admission.GPU_UUIDS[0],
         "gpu_name": "gpu0", "utilization_percent": 12.0},
        {"physical_gpu_index": 1, "gpu_uuid": server_gpu_admission.GPU_UUIDS[1],
         "gpu_name": "gpu1", "utilization_percent": 24.0},
    ]
    monkeypatch.setattr(server_gpu_admission, "query_gpus", lambda: samples)
    selected = launcher.choose_gpu(1, poll_seconds=0)
    assert selected["physical_gpu_index"] == 1

    busy = [dict(row) for row in samples]
    busy[1]["utilization_percent"] = 85.0
    monkeypatch.setattr(server_gpu_admission, "query_gpus", lambda: busy)
    with pytest.raises(server_gpu_admission.NoGpuAvailable):
        launcher.choose_gpu(1, poll_seconds=0)


def test_completion_receipt_requires_all_epoch_recovery_hashes(tmp_path):
    path = tmp_path / "TRAINING_COMPLETE.json"
    receipt = {
        "stage": "TRAINING_COMPLETE",
        "seed": 53,
        "epochs": EPOCHS,
        "recovery_sha256": {str(epoch): "sha" for epoch in range(EPOCHS + 1)},
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_completion(path, 53)["epochs"] == EPOCHS

    receipt["recovery_sha256"].pop("32")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid completion receipt"):
        validate_completion(path, 53)
