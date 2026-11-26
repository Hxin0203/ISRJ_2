"""Auditable ReCoVP-Hybrid runner for the five additional seeds 53..109.

This runner is isolated from the frozen seed-17 and active seed-29/41 code.
It reuses the frozen training implementation while binding all per-run
identity, output, GPU, launcher, and pipeline sources before writing artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from paper2.io import ROOT, sha
from tools import server_gpu_admission


ALLOWED_SEEDS = (53, 67, 79, 97, 109)
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
GPU_UUIDS = dict(server_gpu_admission.GPU_UUIDS)


def _gpu_identity(index: int) -> dict:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError(f"training is restricted to server B / {SERVER_HOST}")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"training root must be {SERVER_ROOT}; got {ROOT}")
    selected = server_gpu_admission.require_visible_gpu()
    if int(selected["physical_gpu_index"]) != index or selected["gpu_uuid"] != GPU_UUIDS[index]:
        raise RuntimeError(f"physical GPU identity mismatch: {selected}")
    query = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [part.strip() for part in query.split(",")]
    if len(fields) != 5 or fields[0] != GPU_UUIDS[index]:
        raise RuntimeError(f"unexpected GPU inventory: {query}")
    if float(fields[4]) >= server_gpu_admission.UTILIZATION_LIMIT:
        raise server_gpu_admission.NoGpuAvailable(f"GPU-{index} at/above 85%: {query}")
    return {
        "hostname": platform.node(), "root": str(ROOT.resolve()),
        "visible_devices": str(index), "physical_gpu_index": index,
        "gpu_uuid": fields[0], "gpu_name": fields[1],
        "free_mib_at_start": float(fields[2]), "total_mib": float(fields[3]),
        "utilization_percent_at_start": float(fields[4]),
        "admission_policy": server_gpu_admission.ADMISSION_POLICY,
        "local_training": False,
    }


def _run_id(seed: int) -> str:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"extended runner only permits {ALLOWED_SEEDS}")
    return f"recovp_hybrid_seed{seed}"


def _paths(seed: int) -> dict[str, Path]:
    name = _run_id(seed)
    return {"base": ROOT / "outputs" / name,
            "evidence": ROOT / "evidence" / "outputs" / name}


def _source_hashes(module_path: Path) -> dict[str, str]:
    import tools.vp_1103_recovp_hybrid_train as frozen_train

    values = frozen_train.source_hashes()
    for rel in (
        "tools/vp_1103_recovp_hybrid_preflight.py",
        "tools/vp_1103_recovp_hybrid_extended.py",
        "tools/start_recovp_hybrid_extended.py",
        "tools/evaluate_recovp_hybrid_extended.py",
        "tools/start_recovp_hybrid_extended_eval.py",
        "tools/run_recovp_hybrid_8seed_pipeline.py",
    ):
        values[rel] = sha(ROOT / rel)
    return values


def run(seed: int, gpu_index: int) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the selected physical GPU")
    import tools.vp_1103_recovp_hybrid_preflight as preflight
    import tools.vp_1103_recovp_hybrid_train as frozen_train

    run_name = _run_id(seed)
    paths = _paths(seed)
    gpu_uuid = GPU_UUIDS[gpu_index]
    module_path = Path(__file__).resolve()
    bindings = _source_hashes(module_path)

    preflight.RUN = run_name
    preflight.BASE = paths["base"]
    preflight.EVIDENCE = paths["evidence"]
    preflight.GPU_INDEX = str(gpu_index)
    preflight.GPU_UUID = gpu_uuid
    preflight.source_bindings = lambda: dict(bindings)

    frozen_train.RUN = run_name
    frozen_train.SEED = seed
    frozen_train.SERVER_HOST = SERVER_HOST
    frozen_train.SERVER_ROOT = SERVER_ROOT
    frozen_train.GPU_INDEX = str(gpu_index)
    frozen_train.GPU_UUID = gpu_uuid
    frozen_train.BASE = paths["base"]
    frozen_train.EVIDENCE = paths["evidence"]
    frozen_train.PREFLIGHT = paths["evidence"] / "PREFLIGHT.json"
    frozen_train.FORMAT = f"recovp-hybrid-seed{seed}-epoch-v1"
    frozen_train.source_hashes = lambda: dict(bindings)
    frozen_train.require_server_and_gpu = lambda: _gpu_identity(gpu_index)
    preflight.guard_server = lambda: _gpu_identity(gpu_index)

    admission = _gpu_identity(gpu_index)
    result = preflight.run()
    result.update({"seed": seed, "requested_seed": seed,
                   "run": run_name, "preflight_rng_seed": 17})
    for target in (paths["base"] / "PREFLIGHT.json", paths["evidence"] / "PREFLIGHT.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trained = frozen_train.train(seed)
    return {"stage": trained.get("stage"), "seed": seed, "run": run_name,
            "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
            "source_sha256": bindings, "admission": admission,
            "training": trained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    result = run(args.seed, args.gpu_index)
    print(json.dumps({k: result[k] for k in ("stage", "seed", "run", "gpu_index", "gpu_uuid")},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
