"""Server-B GPU admission for new jobs; existing runs are never modified.

Both physical GPUs may host concurrent experiments. Resource admission depends
only on GPU utilization being strictly below 85 percent. Host, repository root,
and physical GPU UUID checks are execution-identity checks, not resource gates.
Importing this module does not inspect or initialize CUDA or run commands.
"""
from __future__ import annotations

import csv
import io
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
GPU_UUIDS = {
    0: "GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d",
    1: "GPU-bb60dd3a-738e-2398-908f-7f77ac05549b",
}
UTILIZATION_LIMIT = 85.0
ADMISSION_POLICY = "concurrent use allowed; selected GPU utilization must be below 85 percent"


class NoGpuAvailable(RuntimeError):
    """Both GPUs, or the requested visible GPU, are at or above the limit."""


def require_server_identity() -> None:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError("GPU dispatch is restricted to server B / DESKTOP-RFFERM3")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"GPU dispatch root must be {SERVER_ROOT}; got {ROOT}")


def _validated_samples(samples: Sequence[Mapping]) -> list[dict]:
    result = []
    seen = set()
    for sample in samples:
        index = int(sample["physical_gpu_index"])
        if index not in GPU_UUIDS or index in seen:
            raise RuntimeError(f"unexpected or duplicate physical GPU index: {index}")
        if sample["gpu_uuid"] != GPU_UUIDS[index]:
            raise RuntimeError(f"physical GPU-{index} UUID mismatch: {sample['gpu_uuid']}")
        utilization = float(sample["utilization_percent"])
        if not math.isfinite(utilization) or not 0 <= utilization <= 100:
            raise RuntimeError(f"invalid GPU-{index} utilization: {utilization}")
        result.append({**sample, "physical_gpu_index": index,
                       "utilization_percent": utilization})
        seen.add(index)
    if seen != set(GPU_UUIDS):
        raise RuntimeError(f"expected both physical GPUs 0 and 1; found {sorted(seen)}")
    return result


def parse_gpu_inventory(output: str) -> list[dict]:
    samples = []
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(field.strip() for field in row):
            continue
        if len(row) != 4:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
        index, uuid, name, utilization = (field.strip() for field in row)
        samples.append({"physical_gpu_index": int(index), "gpu_uuid": uuid,
                        "gpu_name": name, "utilization_percent": float(utilization)})
    return _validated_samples(samples)


def query_gpus() -> list[dict]:
    """Query both known server GPUs without inspecting or stopping their jobs."""
    require_server_identity()
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return parse_gpu_inventory(output)


def select_gpu(samples: Sequence[Mapping]) -> dict:
    """Choose the least utilized eligible GPU; GPU index breaks exact ties."""
    validated = _validated_samples(samples)
    eligible = [row for row in validated if row["utilization_percent"] < UTILIZATION_LIMIT]
    if not eligible:
        status = {row["physical_gpu_index"]: row["utilization_percent"] for row in validated}
        raise NoGpuAvailable(f"both GPUs are at or above 85 percent utilization: {status}")
    selected = min(eligible, key=lambda row: (row["utilization_percent"], row["physical_gpu_index"]))
    return {**selected, "admission_policy": ADMISSION_POLICY}


def choose_gpu() -> dict:
    return select_gpu(query_gpus())


def require_visible_gpu() -> dict:
    """Recheck the selected device immediately before a new CUDA job starts."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ("0", "1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify exactly physical GPU 0 or 1")
    selected = next(row for row in query_gpus() if row["physical_gpu_index"] == int(visible))
    if selected["utilization_percent"] >= UTILIZATION_LIMIT:
        raise NoGpuAvailable(f"selected GPU-{visible} utilization is at or above 85 percent: {selected}")
    return {**selected, "admission_policy": ADMISSION_POLICY}
