"""Time frozen model(x) calls on a synthetic resident tensor. Run from repo root."""
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from tools.post_ablation_analysis import load_checkpoint

ROOT = Path.cwd()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


@torch.inference_mode()
def main():
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    x = torch.randn(1, 2, 1800, device=device, dtype=torch.float32)
    models = {}
    checkpoints = {}
    for name in ("full", "unet"):
        model, checkpoints[name], _ = load_checkpoint(name, 17)
        models[name] = model.to(device).eval()
        result = models[name](x)
        assert result.shape == x.shape and torch.isfinite(result).all()

    rounds = []
    pooled = {name: [] for name in models}
    for order in (("full", "unet"), ("unet", "full")) * 2:
        row = {}
        for name in order:
            model = models[name]
            for _ in range(40):
                model(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(160):
                start.record()
                model(x)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            pooled[name].extend(times)
            row[name] = summarize(times)
        rounds.append({"order": order, "measurements": row})
    sources = ["tools/post_ablation_analysis.py", "tools/unet_ablation/models.py",
               "paper2/network.py", "paper2/vendor/legacy_model.py"]
    print(json.dumps({
        "scope": "CUDA events around model(x), includes all forward-internal transforms; excludes input transfer, restoration subtraction and downstream processing",
        "host": platform.node(), "gpu": torch.cuda.get_device_name(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "shape": list(x.shape), "dtype": str(x.dtype), "tf32": False,
        "warmup_per_block": 40, "samples_per_block": 160, "seed": 17, "epoch": 256,
        "checkpoint_sha256": checkpoints, "source_sha256": {s: digest(ROOT/s) for s in sources},
        "rounds": rounds, "pooled": {n: summarize(v) for n, v in pooled.items()}
    }))


if __name__ == "__main__":
    main()
