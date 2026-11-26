"""Dispatch one independent ReCoVP-Hybrid seed on server B.

The completed seed-17 run is deliberately outside this entry point.  Only the
predeclared seeds 29 and 41 are accepted, and every seed gets its own
``outputs/recovp_hybrid_seed<seed>`` and
``evidence/outputs/recovp_hybrid_seed<seed>`` trees.  A new dispatch chooses
the least-utilized physical GPU whose utilization is strictly below 85 percent
and rechecks that choice in the worker immediately before CUDA work starts.

Stages:

``preflight``
    Run the read-only TRAIN/DEV numerical preflight.
``train``
    Run the preflight and the crash-recoverable 256-epoch training.
``evaluate``
    Evaluate an already completed epoch-256 checkpoint on fixed DEV256.
``pipeline``
    Run training followed by fixed DEV256 evaluation on the same admitted GPU.

No CONFIRM or historical TEST data is opened by any stage.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from paper2.io import ROOT, now, sha
from tools import server_gpu_admission


ALLOWED_SEEDS = (29, 41)
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
RUN_PREFIX = "recovp_hybrid_seed"
STAGES = ("preflight", "train", "evaluate", "pipeline")


def _run_id(seed: int) -> str:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"only independent seeds {ALLOWED_SEEDS} are accepted; seed17 is frozen")
    return f"{RUN_PREFIX}{seed}"


def _paths(seed: int) -> dict[str, Path]:
    run = _run_id(seed)
    return {"base": ROOT / "outputs" / run,
            "evidence": ROOT / "evidence" / "outputs" / run}


def _select_gpu(requested: int | None, poll_seconds: float) -> dict[str, Any]:
    """Select an eligible GPU, waiting only when neither physical GPU qualifies."""
    server_gpu_admission.require_server_identity()
    while True:
        try:
            samples = server_gpu_admission.query_gpus()
            if requested is None:
                return server_gpu_admission.select_gpu(samples)
            selected = next(row for row in samples
                            if int(row["physical_gpu_index"]) == requested)
            if float(selected["utilization_percent"]) < server_gpu_admission.UTILIZATION_LIMIT:
                return {**selected, "admission_policy": server_gpu_admission.ADMISSION_POLICY}
            raise server_gpu_admission.NoGpuAvailable(
                f"requested GPU-{requested} is at or above 85 percent: {selected}")
        except server_gpu_admission.NoGpuAvailable:
            if poll_seconds <= 0:
                raise
            time.sleep(poll_seconds)


def _record_dispatch(seed: int, stage: str, selected: dict[str, Any]) -> dict[str, Any]:
    """Write a small auditable dispatch receipt to the isolated run trees."""
    from tools.durable_io import atomic_json

    paths = _paths(seed)
    receipt = {
        "format": "recovp-hybrid-seed-dispatch-v1",
        "run": _run_id(seed), "seed": seed, "stage": stage,
        "created_utc": now(), "hostname": selected.get("hostname"),
        "physical_gpu_index": int(selected["physical_gpu_index"]),
        "gpu_uuid": selected.get("gpu_uuid"),
        "gpu_name": selected.get("gpu_name"),
        "utilization_percent_at_dispatch": float(selected["utilization_percent"]),
        "admission_policy": selected.get("admission_policy"),
        "local_training": False,
        "scope": "TRAIN_AND_DEV_ONLY",
    }
    for path in (paths["base"] / "DISPATCH.json", paths["evidence"] / "DISPATCH.json"):
        atomic_json(path, receipt)
    return receipt


def _prepare(seed: int, gpu_index: int):
    """Patch the frozen seed-17 implementation into this seed's namespaces."""
    # Imports happen only after the caller sets CUDA_VISIBLE_DEVICES.  This is
    # important on CUDA hosts because the worker must see exactly one physical
    # device, mapped as logical cuda:0.
    from tools import vp_1103_recovp_hybrid_multiseed as worker

    preflight, train, binding = worker._patch_modules(
        seed, gpu_index, Path(worker.__file__).resolve())
    # Bind this public dispatcher too, so changing orchestration cannot leave a
    # resumable run with an unrecorded source change.
    wrapper_name = Path(__file__).resolve().relative_to(ROOT).as_posix()
    binding["source_sha256"][wrapper_name] = sha(Path(__file__).resolve())
    source_copy = dict(binding["source_sha256"])
    preflight.source_bindings = lambda: dict(source_copy)
    train.source_hashes = lambda: dict(source_copy)
    return worker, preflight, train, binding


def _run_preflight(worker: Any, preflight: Any, binding: dict[str, Any]) -> dict[str, Any]:
    result = worker._run_preflight(preflight, binding)
    return result


def _patch_evaluator(seed: int, gpu_index: int, worker: Any):
    """Bind the fixed DEV evaluator to an independent seed/output tree."""
    import tools.vp_1103_recovp_hybrid_evaluate as evaluator

    run = _run_id(seed)
    paths = _paths(seed)
    base = paths["base"] / f"seed_{seed}"
    evidence = paths["evidence"]
    evaluator.RUN = run
    evaluator.SEED = seed
    evaluator.BASE = base
    evaluator.CHECKPOINT = base / f"epoch_{evaluator.EPOCH:03d}.pt"
    evaluator.COMPLETE = base / "TRAINING_COMPLETE.json"
    evaluator.CONFIG = paths["base"] / "CONFIG.json"
    evaluator.PREFLIGHT = evidence / "PREFLIGHT.json"
    evaluator.COMPLEX_SCALE_SEED = seed * 1000 + 256

    # The frozen evaluator's final evidence path contains seed_17 literally.
    # Route only that evidence write; the model output and raw parent values
    # already use the patched BASE path.
    original_write = evaluator.write
    expected_old = evidence / "seed_17" / "dev_epoch_256_v1.json"
    expected_new = evidence / f"seed_{seed}" / "dev_epoch_256_v1.json"

    def routed_write(path: Path, value: Any):
        target = Path(path)
        if target == expected_old:
            target = expected_new
        return original_write(target, value)

    evaluator.write = routed_write
    evaluator.require_server_gpu = lambda: worker._server_gpu_identity(gpu_index)
    return evaluator


def run(seed: int, stage: str, gpu_index: int | None = None,
        *, poll_seconds: float = 60.0, batch_parents: int = 2) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    selected = _select_gpu(gpu_index, poll_seconds)
    gpu = int(selected["physical_gpu_index"])
    # Set the mapping before importing torch-backed worker modules.  Existing
    # processes are never stopped or migrated; this affects only this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dispatch = _record_dispatch(seed, stage, selected)
    worker, preflight, train, binding = _prepare(seed, gpu)
    binding["dispatch"] = dispatch

    if stage in ("preflight", "train", "pipeline"):
        # The worker performs a second admission check and writes the complete
        # source/data-bound PREFLIGHT receipt before any training starts.
        binding["admission"] = worker._server_gpu_identity(gpu)
        preflight_result = _run_preflight(worker, preflight, binding)
        if stage == "preflight":
            return {"stage": "PREFLIGHT_PASSED", **binding,
                    "preflight_sha256": sha(Path(binding["evidence"]) / "PREFLIGHT.json")}
        training = train.train(seed)
    else:
        preflight_result = None
        training = None

    if stage in ("evaluate", "pipeline"):
        evaluator = _patch_evaluator(seed, gpu, worker)
        evaluation = evaluator.evaluate(batch_parents=batch_parents)
    else:
        evaluation = None

    result = {"stage": stage, "run": _run_id(seed), "seed": seed,
              "gpu_index": gpu, "gpu_uuid": selected.get("gpu_uuid"),
              "base": str(_paths(seed)["base"]),
              "evidence": str(_paths(seed)["evidence"]),
              "dispatch": dispatch}
    if preflight_result is not None:
        result["preflight"] = preflight_result
    if training is not None:
        result["training"] = training
    if evaluation is not None:
        result["evaluation"] = evaluation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--stage", choices=STAGES, default="pipeline")
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=None,
                        help="optional physical GPU; default selects the least-utilized eligible GPU")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="wait interval when neither GPU is below 85 percent; <=0 fails immediately")
    parser.add_argument("--batch-parents", type=int, default=2,
                        help="DEV evaluation parent batch size")
    args = parser.parse_args()
    result = run(args.seed, args.stage, args.gpu_index,
                 poll_seconds=args.poll_seconds, batch_parents=args.batch_parents)
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("stage", "run", "seed", "gpu_index", "gpu_uuid",
                                 "base", "evidence", "dispatch")},
                     indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
