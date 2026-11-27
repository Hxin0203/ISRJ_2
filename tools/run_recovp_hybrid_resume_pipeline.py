"""Finish paused Hybrid seeds 79/97, then start seed 109 unattended.

Seeds 79 and 97 resume on their frozen GPU identities. Seed 109 starts after
the first resumed seed completes, using the least-utilized GPU below 85%.
All worker output and state receipts remain on server B.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from paper2.io import ROOT, now, read, sha
from tools import server_gpu_admission
from tools.durable_io import ProcessLock


SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
PIPELINE_ID = "recovp_hybrid_resume_20261127"
EVIDENCE = ROOT / "evidence" / "outputs" / PIPELINE_ID
JOBS = {
    79: {"gpu": 1, "saved_epoch": 75},
    97: {"gpu": 0, "saved_epoch": 139},
    109: {"gpu": None, "saved_epoch": 0},
}
MAX_ATTEMPTS = 3
POLL_SECONDS = 30


def _write_status(status: dict) -> None:
    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
    target = EVIDENCE / "STATUS.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(target)


def _complete(seed: int) -> bool:
    path = ROOT / "evidence" / "outputs" / f"recovp_hybrid_seed{seed}" / "TRAINING_COMPLETE.json"
    base = ROOT / "outputs" / f"recovp_hybrid_seed{seed}" / f"seed_{seed}" / "TRAINING_COMPLETE.json"
    if not path.is_file() or not base.is_file() or sha(path) != sha(base):
        return False
    result = read(path)
    return result.get("seed") == seed and result.get("epochs") == 256 \
        and len(result.get("recovery_sha256", {})) == 257


def _command(seed: int, gpu: int) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if seed in (79, 97):
        cmd = [sys.executable, "-m", "tools.resume_recovp_hybrid_paused",
               "--seed", str(seed), "--gpu-index", str(gpu)]
    elif seed == 109:
        cmd = [sys.executable, "-m", "tools.vp_1103_recovp_hybrid_extended",
               "--seed", "109", "--gpu-index", str(gpu)]
    else:
        raise ValueError(seed)
    return cmd, env


def _start(seed: int, gpu: int, status: dict) -> subprocess.Popen:
    sample = server_gpu_admission.query_gpus()[gpu]
    if sample["utilization_percent"] >= server_gpu_admission.UTILIZATION_LIMIT:
        raise server_gpu_admission.NoGpuAvailable(f"GPU-{gpu} is not below 85%: {sample}")
    expected_gpu = JOBS[seed]["gpu"]
    if expected_gpu is not None and gpu != expected_gpu:
        raise RuntimeError(f"seed {seed} must resume on its frozen GPU-{expected_gpu}")
    command, env = _command(seed, gpu)
    attempt = len(status["seeds"][str(seed)]["attempts"]) + 1
    log_path = EVIDENCE / f"seed_{seed}_attempt_{attempt}.log"
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(command, cwd=str(SERVER_ROOT), env=env,
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    status["seeds"][str(seed)]["attempts"].append({
        "attempt": attempt, "pid": process.pid, "gpu": gpu,
        "gpu_uuid": sample["gpu_uuid"],
        "utilization_percent_at_start": sample["utilization_percent"],
        "command": command, "log": str(log_path), "started_utc": now(),
    })
    _write_status(status)
    return process


def _start_when_admitted(seed: int, gpu: int, status: dict) -> subprocess.Popen:
    while True:
        try:
            return _start(seed, gpu, status)
        except server_gpu_admission.NoGpuAvailable as exc:
            status["stage"] = "WAITING_FOR_FROZEN_GPU_BELOW_85_PERCENT"
            status["last_resource_check"] = str(exc)
            _write_status(status)
            time.sleep(POLL_SECONDS)


def _wait_for_gpu(status: dict) -> int:
    while True:
        try:
            selected = server_gpu_admission.choose_gpu()
            return int(selected["physical_gpu_index"])
        except server_gpu_admission.NoGpuAvailable as exc:
            status["stage"] = "WAITING_FOR_GPU_BELOW_85_PERCENT"
            status["last_resource_check"] = str(exc)
            _write_status(status)
            time.sleep(POLL_SECONDS)


def run() -> dict:
    if platform.node().casefold() != SERVER_HOST.casefold() or ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError("resume pipeline may run only on server B in the frozen project root")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    pipeline_lock = ProcessLock(EVIDENCE / "PIPELINE.lock")
    pipeline_lock.acquire()
    status = {
        "pipeline_id": PIPELINE_ID, "stage": "STARTING",
        "planned_epochs": 256, "automatic_resume": True,
        "placement": "79/97 retain original GPU UUID; 109 selects a GPU below 85% after first completion",
        "seeds": {str(seed): {"stage": "PENDING", "saved_before_resume": item["saved_epoch"],
                              "attempts": []} for seed, item in JOBS.items()},
        "evidence_dir": str(EVIDENCE), "python": sys.executable,
    }
    _write_status(status)

    running: dict[int, tuple[subprocess.Popen, object]] = {}
    for seed in (79, 97):
        if _complete(seed):
            status["seeds"][str(seed)]["stage"] = "ALREADY_COMPLETE"
            continue
        gpu = JOBS[seed]["gpu"]
        running[seed] = (_start_when_admitted(seed, gpu, status), None)
        status["seeds"][str(seed)]["stage"] = "RUNNING"
    _write_status(status)

    if not running:
        first_completed = True
    else:
        first_completed = False
    seed109_started = _complete(109)
    if seed109_started:
        status["seeds"]["109"]["stage"] = "ALREADY_COMPLETE"
    elif first_completed:
        gpu = _wait_for_gpu(status)
        running[109] = (_start_when_admitted(109, gpu, status), None)
        status["seeds"]["109"]["stage"] = "RUNNING"
        seed109_started = True
        _write_status(status)

    while running:
        for seed, (proc, _log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            entry = status["seeds"][str(seed)]["attempts"][-1]
            entry["exit_code"] = code
            entry["finished_utc"] = now()
            if code == 0 and _complete(seed):
                status["seeds"][str(seed)]["stage"] = "TRAINING_COMPLETE"
                del running[seed]
                first_completed = True
            elif len(status["seeds"][str(seed)]["attempts"]) < MAX_ATTEMPTS:
                status["seeds"][str(seed)]["stage"] = "RETRYING_FROM_LATEST_CHECKPOINT"
                del running[seed]
                running[seed] = (_start_when_admitted(seed, JOBS[seed]["gpu"], status), None)
            else:
                status["stage"] = "FAILED"
                status["failed_seed"] = seed
                status["error"] = f"seed {seed} failed after {MAX_ATTEMPTS} attempts; checkpoint retained"
                _write_status(status)
                return status
        _write_status(status)
        if first_completed and not seed109_started:
            if _complete(109):
                status["seeds"]["109"]["stage"] = "TRAINING_COMPLETE"
                seed109_started = True
            else:
                gpu = _wait_for_gpu(status)
                running[109] = (_start_when_admitted(109, gpu, status), None)
                status["seeds"]["109"]["stage"] = "RUNNING"
                seed109_started = True
                _write_status(status)
        if running:
            time.sleep(POLL_SECONDS)

    for seed in JOBS:
        if _complete(seed):
            status["seeds"][str(seed)]["stage"] = "TRAINING_COMPLETE"
    status["stage"] = "TRAINING_COMPLETE" if all(_complete(seed) for seed in JOBS) else "INCOMPLETE"
    status["finished_utc"] = now()
    _write_status(status)
    pipeline_lock.release()
    return status


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False), flush=True)
