"""New GPU-0 execution identity; never change the frozen GPU-1 guards."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch

from paper2.io import ROOT, now
from tools.durable_io import atomic_json
from tools.unet_ablation.runtime import (configure_precision, user_control,
                                         begin_budget, end_budget)

from .settings import FACTOR, GPU_INDEX, GPU_UUID, DEPENDENCY


def require_server():
    if (platform.node().casefold() != "desktop-rfferm3"
            or ROOT.resolve() != Path("D:/code/Xin/ISRJ_222").resolve()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX):
        raise RuntimeError("Representation screen: server B, dedicated ISRJ_222 root, factor-assigned physical GPU only. Local training prohibited.")
    uuid = subprocess.check_output(["nvidia-smi", "-i", GPU_INDEX,
                                   "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if uuid != GPU_UUID or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Unexpected physical GPU identity or CUDA mapping; do not fall back")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-").lower() != GPU_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("CUDA logical device does not map to the factor-assigned physical GPU")


def resource_sample():
    import psutil
    row = subprocess.check_output([
        "nvidia-smi", "-i", GPU_INDEX,
        "--query-gpu=utilization.gpu,memory.free,memory.total", "--format=csv,noheader,nounits"
    ], text=True).strip().split(",")
    return {"utc": now(), "gpu_index": int(GPU_INDEX), "gpu_uuid": GPU_UUID,
            "gpu_utilization_percent": float(row[0]), "gpu_free_mib": float(row[1]),
            "gpu_total_mib": float(row[2]), "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_free_bytes": psutil.virtual_memory().available,
            "disk_free_bytes": shutil.disk_usage(ROOT).free}


def admit(base):
    require_server()
    while True:
        user_control(base)
        samples = [resource_sample() for _ in range(3)]
        allowed = all(s["gpu_utilization_percent"] < 20 and s["gpu_free_mib"] >= 12 * 1024
                      and s["cpu_percent"] < 85 and s["ram_free_bytes"] >= 8 * 2**30
                      and s["disk_free_bytes"] >= 8 * 2**30 for s in samples)
        atomic_json(base / "ADMISSION.json", {"stage": "ADMITTED" if allowed else "WAIT_RESOURCES",
                    "samples": samples, "policy": "GPU0<20%,free>=12GiB,CPU<85%,RAM>=8GiB,disk>=8GiB; never stop other jobs"})
        if allowed:
            return
        time.sleep(15)


def wait_dependency(base):
    from paper2.io import read, sha
    from .train import local_sources
    manifest = base / "QUEUE_BINDING.json"
    current = {"factor": FACTOR, "dependency": DEPENDENCY, "source_sha256": local_sources()}
    if manifest.exists():
        if read(manifest) != current:
            raise RuntimeError("Queued source changed; preserve and create a new run")
    else:
        atomic_json(manifest, current)
    evidence = ROOT / "evidence/outputs" / base.name
    atomic_json(evidence / manifest.name, current)
    prerequisite = ROOT / "outputs" / DEPENDENCY
    while True:
        user_control(base)
        if local_sources() != current["source_sha256"]:
            raise RuntimeError("Source drift while queued")
        complete = prerequisite / "PIPELINE_STATUS.json"
        summary = prerequisite / "DEV_SCREEN_SUMMARY.json"
        if complete.exists():
            receipt = read(complete)
            if receipt.get("stage") != "DEV_SCREEN_COMPLETE":
                raise RuntimeError("Unexpected prerequisite completion stage")
            if not summary.exists() or sha(summary) != receipt["dev_screen_sha256"]:
                raise RuntimeError("Prerequisite DEV summary identity mismatch")
            for name, digest in read(prerequisite / "CONFIG.json")["source_sha256"].items():
                if sha(ROOT / name) != digest:
                    raise RuntimeError("Prerequisite source drift")
            released = {"stage": "DEPENDENCY_COMPLETE", "factor": FACTOR,
                        "dependency": DEPENDENCY, "completion_sha256": sha(complete),
                        "summary_sha256": sha(summary), "updated_utc": now()}
            atomic_json(base / "DEPENDENCY_RELEASE.json", released)
            atomic_json(evidence / "DEPENDENCY_RELEASE.json", released)
            return
        state = read(prerequisite / "STATUS.json")
        record = {"stage": "WAIT_DEPENDENCY", "factor": FACTOR, "physical_gpu": int(GPU_INDEX),
                  "dependency": DEPENDENCY, "dependency_stage": state.get("stage"),
                  "dependency_seed": state.get("seed"),
                  "dependency_saved_epoch": state.get("last_committed_epoch", state.get("epoch")),
                  "numerical_preflight": "PENDING_GPU_RELEASE", "updated_utc": now()}
        atomic_json(base / "STATUS.json", record)
        atomic_json(evidence / "STATUS.json", record)
        if state.get("stage") == "FAILED":
            raise RuntimeError("Prerequisite failed; no automatic bypass or competing dispatch")
        time.sleep(30)
