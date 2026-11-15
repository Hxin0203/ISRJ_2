"""Single-update engineering checks, never scientific training results."""
from __future__ import annotations

import time
import numpy as np
import torch

from paper2.io import ROOT, now
from paper2.train_1103 import load_arrays, make_batch
from paper2.train_256_8seed import optimizer_for, seed_rng
from tools.durable_io import atomic_json
from .models import VARIANTS, make_model, objective
from .protocol import DATA, RUN_ID
from .train import minibatch_update
from .runtime import require_server, configure_precision, admit


def main():
    require_server()
    configure_precision()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    arrays = load_arrays(DATA / "train.npz")
    rows = []
    for variant in VARIANTS:
        admit(ROOT / "outputs" / f"{RUN_ID}_engineering", ROOT / "evidence/outputs" / f"{RUN_ID}_engineering")
        seed_rng(2026111500)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = make_model(variant).cuda()
        opt = optimizer_for(net)
        ids = np.arange(32)
        microbatch = 8 if variant == "unet" else 32
        started = time.perf_counter()
        losses = minibatch_update(net, opt, arrays, ids, variant, "cuda", microbatch_records=microbatch)
        torch.cuda.synchronize()
        if not all(np.isfinite(v) for v in losses.values()):
            raise RuntimeError("nonfinite engineering loss")
        row = {"variant": variant, "physical_records": len(ids), "waveforms": len(ids) * 32,
               "seconds": time.perf_counter() - started,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "microbatch_records": microbatch, "loss": losses}
        rows.append(row)
        print(row, flush=True)
        del net, opt, losses
    out = ROOT / "outputs" / f"{RUN_ID}_engineering" / "SMOKE.json"
    record = {"scope": "ENGINEERING_ONLY_NOT_RESEARCH_RESULTS", "utc": now(), "rows": rows,
              "precision": configure_precision(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
    atomic_json(out, record)
    atomic_json(ROOT / "evidence/outputs" / f"{RUN_ID}_engineering" / "SMOKE.json", record)


if __name__ == "__main__":
    main()
