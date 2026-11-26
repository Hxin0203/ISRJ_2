"""Run the five remaining ReCoVP-Hybrid seeds on two fixed GPU lanes.

GPU 1 runs seeds 53/67/79 and GPU 0 runs seeds 97/109. Both lanes can start
while seed 41 is active when their assigned GPU is below 85 percent utilization.
Per-seed dispatch locks coordinate with the original eight-seed controller,
which remains responsible for final DEV256 evaluation after every training
receipt is complete.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools.durable_io import ProcessLock, atomic_json
from tools.recovp_hybrid_seed_schedule import (
    EPOCHS,
    GPU_LANES,
    validate_completion,
)
from tools.server_gpu_admission import require_server_identity


BASE = ROOT / "evidence" / "outputs" / "recovp_hybrid_split_gpu_pipeline"
STATUS_PATHS = (
    BASE / "STATUS.json",
    ROOT / "outputs" / "recovp_hybrid_split_gpu_pipeline" / "STATUS.json",
)
MAX_ATTEMPTS_PER_SEED = 8
_status_lock = threading.Lock()
_state: dict = {}


def _training_receipt(seed: int) -> Path:
    return ROOT / "evidence" / "outputs" / f"recovp_hybrid_seed{seed}" / "TRAINING_COMPLETE.json"


def _completion(seed: int) -> dict | None:
    return validate_completion(_training_receipt(seed), seed)


def _write_status() -> None:
    _state["updated_utc"] = now()
    value = json.loads(json.dumps(_state))
    for path in STATUS_PATHS:
        atomic_json(path, value)


def _set_lane(gpu: int, **updates) -> None:
    with _status_lock:
        key = str(gpu)
        lane = dict(_state["gpu_lanes"].get(key, {}))
        lane.update(updates)
        _state["gpu_lanes"][key] = lane
        _write_status()


def _run_seed(gpu: int, seed: int, poll_seconds: int, retry_seconds: int) -> dict:
    result = _completion(seed)
    if result:
        _set_lane(gpu, current_seed=seed, stage="SEED_ALREADY_COMPLETE",
                  completed_seed=result, receipt_sha256=sha(Path(result["receipt"])))
        return result

    for attempt in range(1, MAX_ATTEMPTS_PER_SEED + 1):
        _set_lane(gpu, current_seed=seed, stage="RUNNING_SEED", attempt=attempt,
                  max_attempts=MAX_ATTEMPTS_PER_SEED, admission="fixed GPU utilization < 85%")
        command = [sys.executable, "-m", "tools.start_recovp_hybrid_extended",
                   "--seed", str(seed), "--gpu-index", str(gpu),
                   "--poll-seconds", str(poll_seconds)]
        log_path = BASE / f"gpu{gpu}_seed{seed}_attempt_{attempt:02d}.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now()}] launch: {command!r}\n")
            proc = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                  stderr=subprocess.STDOUT, check=False)
            log.write(f"[{now()}] exit_code={proc.returncode}\n")
        if proc.returncode == 0:
            result = _completion(seed)
            if result is None:
                raise RuntimeError(f"seed {seed} launcher exited successfully without a valid receipt")
            _set_lane(gpu, current_seed=seed, stage="SEED_COMPLETE",
                      completed_seed=result, receipt_sha256=sha(Path(result["receipt"])))
            return result
        _set_lane(gpu, current_seed=seed, stage="RETRY_WAIT", attempt=attempt,
                  exit_code=proc.returncode, retry_after_seconds=retry_seconds)
        if attempt < MAX_ATTEMPTS_PER_SEED:
            time.sleep(max(1, retry_seconds))
    _set_lane(gpu, current_seed=seed, stage="FAILED_REQUIRES_ATTENTION",
              attempts=MAX_ATTEMPTS_PER_SEED)
    raise RuntimeError(f"seed {seed} failed {MAX_ATTEMPTS_PER_SEED} dispatch attempts")


def _run_lane(gpu: int, seeds: tuple[int, ...], poll_seconds: int,
              retry_seconds: int) -> dict[str, dict]:
    lane_results = {}
    for seed in seeds:
        lane_results[str(seed)] = _run_seed(gpu, seed, poll_seconds, retry_seconds)
    _set_lane(gpu, stage="LANE_COMPLETE", completed=lane_results)
    return lane_results


def run(poll_seconds: int = 60, retry_seconds: int = 60) -> dict:
    require_server_identity()
    BASE.mkdir(parents=True, exist_ok=True)
    global _state
    _state = {
        "stage": "STARTING",
        "controller_source_sha256": sha(Path(__file__)),
        "updated_utc": now(),
        "epochs_per_seed": EPOCHS,
        "gpu_lanes": {
            str(gpu): {"seeds": list(seeds), "stage": "QUEUED"}
            for gpu, seeds in GPU_LANES.items()
        },
        "dev_evaluation": "handled by the existing eight-seed controller after all training receipts validate",
    }
    with ProcessLock(BASE / "PIPELINE.lock"):
        with _status_lock:
            _write_status()
        _set_lane(1, stage="STARTING_NOW")
        _set_lane(0, stage="WAITING_FOR_GPU_ADMISSION")
        results = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="recovp-gpu-lane") as pool:
            futures = {
                pool.submit(_run_lane, gpu, tuple(seeds), poll_seconds, retry_seconds): gpu
                for gpu, seeds in GPU_LANES.items()
            }
            for future in as_completed(futures):
                gpu = futures[future]
                try:
                    results[str(gpu)] = future.result()
                except Exception as exc:
                    _set_lane(gpu, stage="FAILED", error_type=type(exc).__name__, error=str(exc))
            failures = [gpu for future, gpu in futures.items() if future.exception() is not None]
            if failures:
                _state["stage"] = "FAILED_REQUIRES_ATTENTION"
                _state["failed_gpu_lanes"] = failures
                with _status_lock:
                    _write_status()
                raise RuntimeError(f"GPU lanes failed: {failures}")
        _state["stage"] = "FIVE_SEEDS_TRAINING_COMPLETE"
        _state["completed"] = results
        with _status_lock:
            _write_status()
        return json.loads(json.dumps(_state))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    run(args.poll_seconds, args.retry_seconds)


if __name__ == "__main__":
    main()
