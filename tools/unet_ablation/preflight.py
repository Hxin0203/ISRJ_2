"""Cross-host numerical check on fixed engineering waveforms, never study scores."""
from __future__ import annotations

import json
import numpy as np
import torch

from paper2.io import ROOT, now, sha
from tools.durable_io import atomic_json
from .models import make_model, VARIANTS
from .runtime import require_server, configure_precision
from .protocol import BASE, EVIDENCE, ORIGINAL, source_hashes

ENGINEERING = ROOT / "data" / "unet_ablation_engineering"


def reference():
    """CPU only; local preparation cannot start a training loop."""
    torch.set_num_threads(4)
    configure_precision()
    ENGINEERING.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(2026111508).normal(size=(4, 2, 1800)).astype(np.float32)
    values = {"x": x}
    for variant in ("full", *VARIANTS):
        torch.manual_seed(17)
        model = make_model(variant).eval()
        if variant == "full":
            payload = torch.load(ORIGINAL / "seed_17/epoch_256.pt", weights_only=False, map_location="cpu")
            model.load_state_dict(payload["model"])
        with torch.no_grad():
            values[variant] = model(torch.from_numpy(x)).numpy()
    path = ENGINEERING / "reference.npz"
    np.savez_compressed(path, **values)
    return path


def verify():
    require_server()
    torch.set_num_threads(4)
    precision = configure_precision()
    path = ENGINEERING / "reference.npz"
    results = {}
    with np.load(path, allow_pickle=False) as z:
        for variant in ("full", *VARIANTS):
            torch.manual_seed(17)
            model = make_model(variant).eval()
            if variant == "full":
                payload = torch.load(ORIGINAL / "seed_17/epoch_256.pt", weights_only=False, map_location="cpu")
                model.load_state_dict(payload["model"])
            checks = {}
            for device in ("cpu", "cuda"):
                model.to(device)
                with torch.no_grad():
                    actual = model(torch.from_numpy(z["x"]).to(device)).cpu().numpy()
                expected = z[variant]
                delta = actual - expected
                checks[device] = {"max_absolute_error": float(np.max(np.abs(delta))),
                                  "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-20)),
                                  "failed_elements": int(np.sum(np.abs(delta) > 1e-5 + 2e-4 * np.abs(expected))),
                                  "passed": bool(np.allclose(actual, expected, rtol=2e-4, atol=1e-5))}
            results[variant] = checks
    passed = all(r[device]["passed"] for r in results.values() for device in ("cpu", "cuda"))
    record = {"stage": "NUMERICAL_PREFLIGHT_PASSED" if passed else "NUMERICAL_PREFLIGHT_FAILED", "utc": now(), "reference_sha256": sha(path),
              "source_sha256": source_hashes(),
              "rtol": 2e-4, "atol": 1e-5, "variants": results, "precision": precision,
              "scope": "CPU reference and server CPU/CUDA inference on engineering inputs; not scientific results or cross-device training trajectory equivalence"}
    atomic_json(BASE / "NUMERICAL_PREFLIGHT.json", record)
    atomic_json(EVIDENCE / "NUMERICAL_PREFLIGHT.json", record)
    print(json.dumps(record), flush=True)
    if not passed:
        raise RuntimeError("CPU/CUDA preflight failed; see persisted per-variant diagnostics")


if __name__ == "__main__":
    verify()
