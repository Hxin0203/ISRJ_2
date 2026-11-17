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

GPU_INDEX = "0"
GPU_UUID = "GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d"


def require_server():
    if (platform.node().casefold() != "desktop-rfferm3"
            or ROOT.resolve() != Path("D:/code/Xin/ISRJ_222").resolve()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX):
        raise RuntimeError("Normalization screen: server B, dedicated ISRJ_222 root, physical GPU 0 only. Local training prohibited.")
    uuid = subprocess.check_output(["nvidia-smi", "-i", GPU_INDEX,
                                   "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if uuid != GPU_UUID or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Unexpected physical GPU identity or CUDA mapping; do not fall back")
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.removeprefix("GPU-").lower() != GPU_UUID.removeprefix("GPU-").lower():
        raise RuntimeError("CUDA logical device does not map to the authorized physical GPU 0")


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
