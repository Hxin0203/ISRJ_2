"""Fixed two-GPU lanes for the remaining ReCoVP-Hybrid seeds."""
from __future__ import annotations

import json


GPU_LANES = {
    1: (53, 67, 79),
    0: (97, 109),
}
SEED_GPU = {seed: gpu for gpu, seeds in GPU_LANES.items() for seed in seeds}
EPOCHS = 256


def assigned_gpu(seed: int) -> int:
    try:
        return SEED_GPU[seed]
    except KeyError as exc:
        raise ValueError(f"seed {seed} is not in the remaining five-seed schedule") from exc


def validate_completion(path: Path, seed: int) -> dict | None:
    """Return a compact completion record, or None while the receipt is absent."""
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    recoveries = value.get("recovery_sha256", {})
    expected = {str(epoch) for epoch in range(EPOCHS + 1)}
    if (value.get("stage") != "TRAINING_COMPLETE" or value.get("seed") != seed
            or value.get("epochs") != EPOCHS or set(recoveries) != expected):
        raise RuntimeError(f"seed {seed} has an invalid completion receipt: {path}")
    return {
        "seed": seed,
        "receipt": str(path),
        "completed_utc": value.get("completed_utc"),
        "epochs": EPOCHS,
    }
