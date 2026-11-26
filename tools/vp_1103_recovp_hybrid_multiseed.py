"""Auditable ReCoVP-Hybrid runner for the predeclared seeds 29 and 41.

This module is deliberately separate from the completed seed-17 runner.  It
loads the frozen implementation only after binding every run-specific value
(seed, output paths, format, physical GPU and UUID) and writes a seed-specific
preflight receipt before training.  It never changes the seed-17 module or its
checkpoints.  The launcher that selects a device is
``tools.start_recovp_hybrid_multiseed``.

The worker must run on server B with ``CUDA_VISIBLE_DEVICES`` set to exactly
one physical device.  The launcher performs the <85% admission and the worker
rechecks the selected GPU immediately before preflight/training.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools import server_gpu_admission


ALLOWED_SEEDS = (29, 41)
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
GPU_UUIDS = dict(server_gpu_admission.GPU_UUIDS)


def _parse_gpu_index(value: str | None) -> int:
    visible = value if value is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ("0", "1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify exactly physical GPU 0 or 1")
    return int(visible)


def _run_id(seed: int) -> str:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Hybrid multi-seed runner permits only {ALLOWED_SEEDS}; seed17 is frozen")
    return f"recovp_hybrid_seed{seed}"


def _paths(seed: int) -> dict[str, Path]:
    run = _run_id(seed)
    return {
        "base": ROOT / "outputs" / run,
        "evidence": ROOT / "evidence" / "outputs" / run,
    }


def _server_gpu_identity(gpu_index: int) -> dict:
    """Fail closed on host/root/device identity and current utilization."""
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError(f"Hybrid training is restricted to server B / {SERVER_HOST}")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"Hybrid training root must be {SERVER_ROOT}; got {ROOT}")
    selected = server_gpu_admission.require_visible_gpu()
    if int(selected["physical_gpu_index"]) != gpu_index:
        raise RuntimeError(f"selected physical GPU changed: expected {gpu_index}, got {selected}")
    if selected["gpu_uuid"] != GPU_UUIDS[gpu_index]:
        raise RuntimeError(f"physical GPU-{gpu_index} UUID mismatch: {selected}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeError("worker must expose exactly the selected physical GPU")
    query = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu_index),
        "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    parts = [part.strip() for part in query.split(",")]
    if len(parts) != 5 or parts[0] != GPU_UUIDS[gpu_index]:
        raise RuntimeError(f"physical GPU identity changed: {query}")
    free_mib, total_mib, utilization = map(float, parts[2:])
    if utilization >= server_gpu_admission.UTILIZATION_LIMIT:
        raise server_gpu_admission.NoGpuAvailable(
            f"selected GPU-{gpu_index} utilization is at or above 85 percent: {query}")
    return {
        "hostname": platform.node(), "root": str(ROOT.resolve()),
        "visible_devices": str(gpu_index), "physical_gpu_index": gpu_index,
        "gpu_uuid": parts[0], "gpu_name": parts[1],
        "free_mib_at_start": free_mib, "total_mib": total_mib,
        "utilization_percent_at_start": utilization,
        "admission_policy": server_gpu_admission.ADMISSION_POLICY,
        "local_training": False,
    }


def _generic_source_hashes(module_path: Path) -> dict[str, str]:
    """Bind the frozen implementation plus this worker and launcher."""
    import tools.vp_1103_recovp_hybrid_train as legacy_train

    values = legacy_train.source_hashes()
    # The frozen preflight validator adds this source explicitly when checking
    # the checkpoint identity; include it in the recorded receipt as well.
    values["tools/vp_1103_recovp_hybrid_preflight.py"] = sha(
        ROOT / "tools" / "vp_1103_recovp_hybrid_preflight.py")
    values["tools/vp_1103_recovp_hybrid_multiseed.py"] = sha(module_path)
    launcher = ROOT / "tools" / "start_recovp_hybrid_multiseed.py"
    if launcher.exists():
        values["tools/start_recovp_hybrid_multiseed.py"] = sha(launcher)
    return values


def _patch_modules(seed: int, gpu_index: int, module_path: Path) -> tuple[object, object, dict]:
    """Bind generic values into imported frozen functions before invocation.

    The seed-17 source remains untouched.  Functions are reused only as an
    implementation library; all mutable run constants are replaced in their
    own module namespaces and all artifacts receive a new identity.
    """
    import tools.vp_1103_recovp_hybrid_preflight as preflight
    import tools.vp_1103_recovp_hybrid_train as legacy_train

    run = _run_id(seed)
    paths = _paths(seed)
    gpu_uuid = GPU_UUIDS[gpu_index]
    source_hashes = _generic_source_hashes(module_path)

    # Preflight constants and guard.
    preflight.RUN = run
    preflight.BASE = paths["base"]
    preflight.EVIDENCE = paths["evidence"]
    preflight.GPU_INDEX = str(gpu_index)
    preflight.GPU_UUID = gpu_uuid
    preflight.source_bindings = lambda: dict(source_hashes)

    # Training constants and guard.
    legacy_train.RUN = run
    legacy_train.SEED = seed
    legacy_train.SERVER_HOST = SERVER_HOST
    legacy_train.SERVER_ROOT = SERVER_ROOT
    legacy_train.GPU_INDEX = str(gpu_index)
    legacy_train.GPU_UUID = gpu_uuid
    legacy_train.BASE = paths["base"]
    legacy_train.EVIDENCE = paths["evidence"]
    legacy_train.PREFLIGHT = paths["evidence"] / "PREFLIGHT.json"
    legacy_train.FORMAT = f"recovp-hybrid-seed{seed}-epoch-v1"
    legacy_train.source_hashes = lambda: dict(source_hashes)

    def require_server_and_gpu() -> dict:
        return _server_gpu_identity(gpu_index)

    legacy_train.require_server_and_gpu = require_server_and_gpu
    preflight.guard_server = lambda: _server_gpu_identity(gpu_index)

    return preflight, legacy_train, {
        "run": run, "seed": seed, "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid, "source_sha256": source_hashes,
        "base": str(paths["base"]), "evidence": str(paths["evidence"]),
    }


def _run_preflight(preflight: object, binding: dict) -> dict:
    """Run the read-only TRAIN/DEV preflight and relabel its requested seed."""
    # ``run`` performs the same numerical and gradient checks as the frozen
    # seed-17 preflight.  Its legacy literal seed only controls check RNG; the
    # resulting receipt is relabelled to the requested training seed so the
    # checkpoint identity cannot cross seed directories.
    result = preflight.run()
    result["seed"] = binding["seed"]
    result["run"] = binding["run"]
    result["requested_seed"] = binding["seed"]
    result["preflight_rng_seed"] = 17
    for path in (
        Path(binding["base"]) / "PREFLIGHT.json",
        Path(binding["evidence"]) / "PREFLIGHT.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run(seed: int, gpu_index: int | None = None, *, preflight_only: bool = False) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}; seed17 is frozen")
    selected_gpu = _parse_gpu_index(str(gpu_index) if gpu_index is not None else None)
    module_path = Path(__file__).resolve()
    preflight, legacy_train, binding = _patch_modules(seed, selected_gpu, module_path)
    # Admission is checked before preflight and again by the training function.
    binding["admission"] = _server_gpu_identity(selected_gpu)
    preflight_result = _run_preflight(preflight, binding)
    if preflight_only:
        return {"stage": "PREFLIGHT_PASSED", **binding,
                "preflight_sha256": sha(Path(binding["evidence"]) / "PREFLIGHT.json")}
    result = legacy_train.train(seed)
    return {"stage": result.get("stage"), **binding,
            "training": result, "preflight_sha256": sha(Path(binding["evidence"]) / "PREFLIGHT.json"),
            "preflight": preflight_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=None,
                        help="physical GPU selected by the launcher; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the TRAIN/DEV-only checks and stop before training")
    args = parser.parse_args()
    result = run(args.seed, args.gpu_index, preflight_only=args.preflight_only)
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("stage", "run", "seed", "gpu_index", "gpu_uuid",
                                 "base", "evidence", "preflight_sha256")},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
