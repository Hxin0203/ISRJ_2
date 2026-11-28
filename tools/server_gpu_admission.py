"""Server-B resource admission for new jobs.

Both physical GPUs may host concurrent experiments. A new dispatch requires
the selected GPU utilization and memory occupancy to be below the user-set
limits, together with host, repository-root and UUID identity checks.
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
SYSTEM_MEMORY_LIMIT = 90.0
ADMISSION_POLICY = (
    "concurrent use allowed; selected GPU utilization must be below 85 percent "
    "and host physical memory must be below 90 percent"
)


class NoGpuAvailable(RuntimeError):
    """No GPU and host-memory snapshot satisfies the dispatch limits."""


def _validated_memory(sample: Mapping) -> dict:
    total = int(sample["total_bytes"])
    available = int(sample["available_bytes"])
    if total <= 0 or not 0 <= available <= total:
        raise RuntimeError(f"invalid physical-memory snapshot: {sample}")
    used = total - available
    utilization = 100.0 * used / total
    if not math.isfinite(utilization) or not 0 <= utilization <= 100:
        raise RuntimeError(f"invalid physical-memory utilization: {utilization}")
    return {**sample, "total_bytes": total, "available_bytes": available,
            "used_bytes": used, "utilization_percent": utilization}


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
        if len(row) not in (4, 6):
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
        index, uuid, name, utilization = (field.strip() for field in row[:4])
        sample = {"physical_gpu_index": int(index), "gpu_uuid": uuid,
                  "gpu_name": name, "utilization_percent": float(utilization)}
        if len(row) == 6:
            sample["memory_used_mib"] = float(row[4].strip())
            sample["memory_total_mib"] = float(row[5].strip())
        samples.append(sample)
    return _validated_samples(samples)


def query_gpus() -> list[dict]:
    """Query both known server GPUs without inspecting or stopping their jobs."""
    require_server_identity()
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True)
    return parse_gpu_inventory(output)


def query_memory() -> dict:
    """Return the host physical-memory snapshot used for admission.

    ``GlobalMemoryStatusEx`` is available on Windows Server B and avoids a
    dependency on PowerShell, WMI or a third-party package. The fallback uses
    ``psutil`` when the module is imported in a compatible test environment.
    """
    require_server_identity()
    if os.name == "nt":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("GlobalMemoryStatusEx failed")
        raw = {"total_bytes": status.ullTotalPhys,
               "available_bytes": status.ullAvailPhys,
               "source": "GlobalMemoryStatusEx"}
    else:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("physical-memory query requires psutil outside Windows") from exc
        vm = psutil.virtual_memory()
        raw = {"total_bytes": vm.total, "available_bytes": vm.available,
               "source": "psutil.virtual_memory"}
    return _validated_memory(raw)


def query_resources() -> dict:
    """Return one auditable GPU and host-memory snapshot."""
    return {"gpus": query_gpus(), "memory": query_memory(),
            "queried_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}


def _admissible_memory(memory: Mapping) -> bool:
    return float(memory["utilization_percent"]) < SYSTEM_MEMORY_LIMIT


def select_gpu(samples: Sequence[Mapping], memory: Mapping | None = None) -> dict:
    """Choose the least utilized eligible GPU; current memory breaks ties."""
    validated = _validated_samples(samples)
    if memory is not None and not _admissible_memory(memory):
        raise NoGpuAvailable(f"host physical-memory utilization is at or above 90 percent: {memory}")
    eligible = [row for row in validated if row["utilization_percent"] < UTILIZATION_LIMIT]
    if not eligible:
        status = {row["physical_gpu_index"]: row["utilization_percent"] for row in validated}
        raise NoGpuAvailable(f"both GPUs are at or above 85 percent utilization: {status}")
    def rank(row: Mapping) -> tuple[float, float, int]:
        used = float(row.get("memory_used_mib", 0.0))
        total = float(row.get("memory_total_mib", 0.0))
        memory_fraction = used / total if total > 0 else 0.0
        return float(row["utilization_percent"]), memory_fraction, int(row["physical_gpu_index"])

    selected = min(eligible, key=rank)
    return {**selected, "admission_policy": ADMISSION_POLICY,
            "host_memory": dict(memory) if memory is not None else None}


def choose_gpu() -> dict:
    resources = query_resources()
    return select_gpu(resources["gpus"], resources["memory"])


def require_visible_gpu() -> dict:
    """Recheck the selected device immediately before a new CUDA job starts."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in ("0", "1"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must identify exactly physical GPU 0 or 1")
    resources = query_resources()
    selected = next(row for row in resources["gpus"] if row["physical_gpu_index"] == int(visible))
    if selected["utilization_percent"] >= UTILIZATION_LIMIT:
        raise NoGpuAvailable(f"selected GPU-{visible} utilization is at or above 85 percent: {selected}")
    if not _admissible_memory(resources["memory"]):
        raise NoGpuAvailable(f"host physical-memory utilization is at or above 90 percent: {resources['memory']}")
    return {**selected, "admission_policy": ADMISSION_POLICY,
            "host_memory": resources["memory"]}
