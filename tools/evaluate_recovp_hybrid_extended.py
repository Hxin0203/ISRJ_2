"""Bind the frozen DEV256 evaluator to an additional ReCoVP-Hybrid seed."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from paper2.io import ROOT, sha
from tools.vp_1103_recovp_hybrid_extended import ALLOWED_SEEDS, SERVER_ROOT, _gpu_identity, _run_id


def evaluate(seed: int, gpu_index: int, batch_parents: int = 2) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"extended evaluation permits {ALLOWED_SEEDS}")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"evaluation root must be {SERVER_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match selected physical GPU")
    import tools.vp_1103_recovp_hybrid_evaluate as evaluator

    run = _run_id(seed)
    base = ROOT / "outputs" / run
    seed_base = base / f"seed_{seed}"
    evidence = ROOT / "evidence" / "outputs" / run
    evaluator.RUN = run
    evaluator.SEED = seed
    evaluator.BASE = seed_base
    evaluator.CHECKPOINT = seed_base / f"epoch_{evaluator.EPOCH:03d}.pt"
    evaluator.COMPLETE = seed_base / "TRAINING_COMPLETE.json"
    evaluator.CONFIG = base / "CONFIG.json"
    evaluator.PREFLIGHT = evidence / "PREFLIGHT.json"
    evaluator.COMPLEX_SCALE_SEED = seed * 1000 + evaluator.EPOCH
    evaluator.require_server_gpu = lambda: _gpu_identity(gpu_index)

    expected_new = evidence / f"seed_{seed}" / "dev_epoch_256_v1.json"
    original_write = evaluator.write

    def routed_write(path: Path, value):
        target = Path(path)
        if target == evidence / "seed_17" / "dev_epoch_256_v1.json":
            target = expected_new
        return original_write(target, value)

    evaluator.write = routed_write
    admission = _gpu_identity(gpu_index)
    result = evaluator.evaluate(batch_parents=batch_parents)
    output = seed_base / f"dev_epoch_{evaluator.EPOCH:03d}_v1.json"
    if (result.get("stage") != "DEV_ONLY_RECOVP_HYBRID" or result.get("seed") != seed
            or result.get("epoch") != evaluator.EPOCH or result.get("parents") != 256):
        raise RuntimeError("DEV evaluation did not return a complete fixed-DEV receipt")
    return {"result": result, "output": str(output), "output_sha256": sha(output),
            "evidence_output": str(expected_new), "evidence_sha256": sha(expected_new),
            "admission": admission}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-parents", type=int, default=2)
    args = parser.parse_args()
    report = evaluate(args.seed, args.gpu_index, args.batch_parents)
    print({"seed": args.seed, "stage": report["result"]["stage"],
           "output": report["output"], "output_sha256": report["output_sha256"]}, flush=True)


if __name__ == "__main__":
    main()
