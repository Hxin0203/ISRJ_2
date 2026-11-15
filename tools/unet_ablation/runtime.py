"""Server-only admission, user controls and cumulative study time budget."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch

from paper2.io import ROOT, now, read
from tools.durable_io import atomic_json

SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = "D:/code/Xin/ISRJ_222"
GPU_INDEX = "1"
BUDGET_SECONDS = 345600


def configure_precision():
    """Use full FP32 convolution on the server as well as CPU reference checks."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


class UserStop(RuntimeError):
    pass


def require_server():
    if (platform.node().casefold() != SERVER_HOST.casefold()
            or ROOT.resolve() != Path(SERVER_ROOT).resolve()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX):
        raise RuntimeError("This study runs only on rsdr-b at D:/code/Xin/ISRJ_222, CUDA_VISIBLE_DEVICES=1. Local training is prohibited.")
    if not torch.cuda.is_available():
        raise RuntimeError("Server CUDA is unavailable; never fall back to local/CPU training")


def user_control(base):
    if (base / "BUDGET.json").exists():
        budget = read(base / "BUDGET.json")
        if budget.get("active_since_unix") is not None and budget_remaining(budget) <= 0:
            raise UserStop("Study cumulative 96-hour budget exhausted; saved checkpoints retained")
    if (base / "STOP").exists():
        raise UserStop("User STOP: stopped at a saved epoch/job boundary")
    while (base / "PAUSE").exists():
        atomic_json(base / "CONTROL_STATUS.json", {"stage": "PAUSED", "updated_utc": now()})
        if (base / "STOP").exists():
            raise UserStop("User STOP while paused")
        time.sleep(5)


def resource_sample():
    import psutil
    row = subprocess.check_output([
        "nvidia-smi", "-i", GPU_INDEX,
        "--query-gpu=utilization.gpu,memory.free,memory.total", "--format=csv,noheader,nounits"
    ], text=True).strip().split(",")
    return {"utc": now(), "gpu_index": int(GPU_INDEX), "gpu_utilization_percent": float(row[0]),
            "gpu_free_mib": float(row[1]), "gpu_total_mib": float(row[2]),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_free_bytes": psutil.virtual_memory().available,
            "disk_free_bytes": shutil.disk_usage(ROOT).free}


def admit(base, evidence):
    require_server()
    while True:
        user_control(base)
        samples = [resource_sample() for _ in range(3)]
        allowed = all(s["gpu_utilization_percent"] < 90 and s["cpu_percent"] < 85
                      and s["ram_free_bytes"] >= 5 * 2**30 and s["disk_free_bytes"] >= 8 * 2**30
                      and s["gpu_free_mib"] >= 4096 for s in samples)
        record = {"stage": "ADMITTED" if allowed else "WAIT_RESOURCES", "samples": samples,
                  "policy": "GPU<90%,CPU<85%,RAM>=5GiB,disk>=8GiB,CUDA free>=4GiB; wait for new dispatch only"}
        atomic_json(base / "ADMISSION.json", record)
        atomic_json(evidence / "ADMISSION.json", record)
        if allowed:
            return
        time.sleep(15)


def begin_budget(base):
    """Unclosed controller intervals count through the restart, conservatively."""
    path = base / "BUDGET.json"
    state = read(path) if path.exists() else {"spent_seconds": 0.0, "limit_seconds": BUDGET_SECONDS}
    current = time.time()
    if state.get("active_since_unix") is not None:
        state["spent_seconds"] += max(0.0, current - state["active_since_unix"])
    state.update(active_since_unix=current, updated_utc=now())
    atomic_json(path, state)
    return state


def budget_remaining(state):
    return BUDGET_SECONDS - state["spent_seconds"] - max(0.0, time.time() - state["active_since_unix"])


def end_budget(base, state):
    state["spent_seconds"] += max(0.0, time.time() - state["active_since_unix"])
    state.update(active_since_unix=None, updated_utc=now())
    atomic_json(base / "BUDGET.json", state)
