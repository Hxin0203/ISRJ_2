"""Recover unstarted seeds 97/109 without changing any frozen worker source.

Shares the original per-seed OS locks. Admission waits do not consume the
training-error budget; every new attempt considers both physical GPUs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools import server_gpu_admission as admission
from tools.durable_io import ProcessLock, atomic_json
from tools.recovp_hybrid_seed_schedule import validate_completion
from tools.vp_1103_recovp_hybrid_extended import _source_hashes

BASE = ROOT / "evidence/outputs/recovp_hybrid_queue_recovery_20261127"


def admission_failure(log: str) -> bool:
    # Tracebacks have a final exception line. Earlier mentions in a command or
    # training log must not hide an unrelated terminal exception.
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    return bool(lines and "NoGpuAvailable:" in lines[-1])


def main() -> None:
    admission.require_server_identity()
    BASE.mkdir(parents=True, exist_ok=True)
    state = {"stage": "STARTING", "pid": os.getpid(), "seeds": [97, 109],
             "controller_sha256": sha(Path(__file__)), "completed": []}

    def mark(**updates):
        state.update(updates, updated_utc=now())
        atomic_json(BASE / "STATUS.json", state)

    with ProcessLock(BASE / "PIPELINE.lock"):
        for seed in state["seeds"]:
            folder = ROOT / f"evidence/outputs/recovp_hybrid_seed{seed}"
            receipt = folder / "TRAINING_COMPLETE.json"
            errors = 0
            attempt = 0
            while not validate_completion(receipt, seed):
                lock = ProcessLock(folder / "DISPATCH.lock")
                try:
                    lock.acquire()
                except RuntimeError:
                    mark(stage="WAITING_SEED_LOCK", current_seed=seed)
                    time.sleep(60)
                    continue
                try:
                    if validate_completion(receipt, seed):
                        break
                    # This recovery is intentionally restricted to unstarted
                    # runs; preserve all recovery points on unexpected errors.
                    recoveries = ROOT / f"outputs/recovp_hybrid_seed{seed}/seed_{seed}/recovery"
                    if any(recoveries.glob("*.pt")):
                        raise RuntimeError("existing recovery requires identity audit; refusing automatic restart")
                    try:
                        selected = admission.choose_gpu()
                    except admission.NoGpuAvailable:
                        mark(stage="WAITING_GPU", current_seed=seed)
                        time.sleep(60)
                        continue
                    attempt += 1
                    gpu = selected["physical_gpu_index"]
                    log = BASE / f"seed{seed}_attempt{attempt:03d}.log"
                    bindings = _source_hashes(Path(__file__))
                    atomic_json(BASE / f"seed{seed}_dispatch{attempt:03d}.json", {
                        "seed": seed, "created_utc": now(), "admission": selected,
                        "source_sha256": bindings, "controller_sha256": sha(Path(__file__)),
                        "worker_validates": "preflight source/data SHA and checkpoint identity before training"})
                    mark(stage="RUNNING_SEED", current_seed=seed, attempt=attempt,
                         selected_gpu=selected, log=str(log), training_errors=errors)
                    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                           "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUTF8": "1"}
                    with log.open("w", encoding="utf-8") as stream:
                        code = subprocess.call([sys.executable, "-m",
                            "tools.vp_1103_recovp_hybrid_extended", "--seed", str(seed),
                            "--gpu-index", str(gpu)], cwd=str(ROOT), env=env,
                            stdout=stream, stderr=subprocess.STDOUT)
                    if code == 0:
                        if not validate_completion(receipt, seed):
                            raise RuntimeError("worker exited without complete epoch receipts")
                        break
                    transient = admission_failure(log.read_text(encoding="utf-8", errors="replace"))
                    errors += int(not transient)
                    mark(stage="WAITING_GPU" if transient else "RETRY_ERROR",
                         exit_code=code, training_errors=errors)
                    if errors >= 3:
                        raise RuntimeError("three non-admission errors; preserve artifacts for audit")
                finally:
                    lock.release()
                time.sleep(60)
            state["completed"].append(seed)
            mark(stage="SEED_COMPLETE", receipt_sha256=sha(receipt))
        mark(stage="QUEUE_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(BASE / "FAILURE.json", {"stage": "FAILED", "updated_utc": now(),
                    "error_type": type(exc).__name__, "error": str(exc)})
        raise
