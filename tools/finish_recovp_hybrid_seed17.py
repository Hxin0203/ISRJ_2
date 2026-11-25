"""Wait for the existing seed-17 run, evaluate on an admitted GPU, then stop.

This server-only controller never starts training, changes a training source,
opens CONFIRM/TEST, or expands the registered single-seed experiment scope.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

from paper2.io import ROOT, now, read, sha
from tools.durable_io import ProcessLock, atomic_json, process_alive
from tools.server_gpu_admission import (
    NoGpuAvailable, choose_gpu, require_server_identity,
)


RUN = "recovp_hybrid_seed17"
EVIDENCE = ROOT / "evidence" / "outputs" / RUN
FOLDER = ROOT / "outputs" / RUN / "seed_17"
COMPLETE = FOLDER / "TRAINING_COMPLETE.json"
RESULT = FOLDER / "dev_epoch_256_v1.json"
STATUS = EVIDENCE / "POSTTRAINING_STATUS.json"


def source_bindings() -> dict[str, str]:
    paths = ("tools/finish_recovp_hybrid_seed17.py",
             "tools/server_gpu_admission.py",
             "tools/vp_1103_recovp_hybrid_evaluate.py")
    return {name: sha(ROOT / name) for name in paths}


def existing_evaluation() -> dict | None:
    if not RESULT.exists():
        return None
    result = read(RESULT)
    complete = read(COMPLETE)
    if (result.get("stage") != "DEV_ONLY_RECOVP_HYBRID"
            or result.get("seed") != 17 or result.get("epoch") != 256
            or result.get("parents") != 256
            or result.get("checkpoint_identity") != complete.get("identity")
            or result.get("checkpoint_sha256") != complete.get("milestones", {}).get("256")):
        raise RuntimeError("existing DEV evaluation identity differs from completed seed 17")
    raw = FOLDER / "dev_epoch_256_parent_values.npz"
    if not raw.is_file() or sha(raw) != result.get("raw_parent_values_sha256"):
        raise RuntimeError("existing DEV parent-value archive SHA is invalid")
    return result


def training_liveness() -> dict:
    """Read the trainer's lock metadata without acquiring or rewriting it."""
    path = FOLDER / "PROCESS.lock"
    for attempt in range(3):
        try:
            # Byte zero is the trainer's mandatory Windows byte-range lock.
            with path.open("rb") as stream:
                stream.seek(1)
                metadata = json.loads(stream.read().decode("utf-8"))
            pid = metadata["pid"]
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError("training lock PID must be a positive integer")
            return {**metadata, "pid_alive": process_alive(pid)}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if attempt == 2:
                raise RuntimeError("cannot read valid seed-17 training lock metadata") from exc
            time.sleep(0.1)
    raise AssertionError("unreachable")


def run(poll_seconds: float = 30) -> dict:
    require_server_identity()
    if not 5 <= poll_seconds <= 60:
        raise ValueError("poll_seconds must be between 5 and 60")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    frozen_sources = {}

    def publish(stage: str, **extra) -> dict:
        status = {"stage": stage, "run": RUN, "seed": 17,
                  "updated_utc": now(), "controller_pid": os.getpid(),
                  "source_sha256": frozen_sources,
                  "scope": "existing seed17 -> fixed DEV256 -> review; no new training",
                  **extra}
        atomic_json(STATUS, status)
        return status

    with ProcessLock(EVIDENCE / "POSTTRAINING_PROCESS.lock"):
        try:
            frozen_sources = source_bindings()
            return _run_loop(poll_seconds, frozen_sources, publish)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc(), "failed_utc": now()}
            previous = None
            try:
                if STATUS.exists():
                    previous = read(STATUS)
                    if not isinstance(previous, dict):
                        previous = None
            except (OSError, ValueError):
                pass
            if previous and previous.get("stage") == "DEV_EVALUATION_FAILED":
                atomic_json(STATUS, {**previous, "updated_utc": now(), "error": error})
            else:
                publish("POSTTRAINING_FAILED", error=error, previous_status=previous)
            raise


def _run_loop(poll_seconds: float, frozen_sources: dict, publish) -> dict:
    while True:
        if source_bindings() != frozen_sources:
            raise RuntimeError("posttraining controller/evaluator source changed during wait")
        if not COMPLETE.exists():
            current = read(EVIDENCE / "STATUS.json")
            training = training_liveness()
            if not training["pid_alive"]:
                # Normal completion writes the receipt before releasing
                # the training lock, so recheck it before declaring failure.
                if COMPLETE.exists():
                    continue
                raise RuntimeError("seed-17 training process stopped without TRAINING_COMPLETE")
            publish("WAITING_FOR_SEED17_TRAINING", training_status=current,
                    training_process=training)
            time.sleep(poll_seconds)
            continue
        complete = read(COMPLETE)
        if (complete.get("stage") != "TRAINING_COMPLETE"
                or complete.get("seed") != 17 or complete.get("epochs") != 256):
            raise RuntimeError("invalid seed-17 training completion receipt")
        result = existing_evaluation()
        if result is not None:
            return publish("CORE_DEV_COMPLETE_REVIEW_REQUIRED",
                           result_path=str(RESULT), result_sha256=sha(RESULT),
                           checkpoint_sha256=result["checkpoint_sha256"],
                           remaining_task_book_work=[
                               "frozen Base/P0/UNet/RSDR paired comparisons and equal-PFA tables",
                               "Top-10 failure-parent audit and CPU latency comparisons",
                               "complete SUMMARY/REPORT/VERIFICATION/REPRODUCE delivery",
                           ])
        try:
            selected = choose_gpu()
        except NoGpuAvailable as exc:
            publish("WAITING_FOR_GPU_BELOW_85_PERCENT", detail=str(exc))
            time.sleep(poll_seconds)
            continue
        child_env = {**os.environ,
                     "CUDA_VISIBLE_DEVICES": str(selected["physical_gpu_index"])}
        publish("RUNNING_FIXED_DEV256", gpu_admission=selected)
        # Keep full logs on the server; only compact receipts are synced.
        error_path = EVIDENCE / "dev.stderr.log"
        error_start = error_path.stat().st_size if error_path.exists() else 0
        with (EVIDENCE / "dev.stdout.log").open("ab") as stdout, \
                (EVIDENCE / "dev.stderr.log").open("ab") as stderr:
            process = subprocess.run(
                [sys.executable, "-m", "tools.vp_1103_recovp_hybrid_evaluate",
                 "--batch-parents", "2"],
                cwd=ROOT, env=child_env, stdout=stdout, stderr=stderr)
        if process.returncode:
            # Admission can change between selection and CUDA startup.
            # Retry only that explicit no-work admission failure; numerical,
            # identity, memory, and evaluation failures require inspection.
            with error_path.open("rb") as error_stream:
                error_stream.seek(max(error_start, error_path.stat().st_size - 4000))
                error_tail = error_stream.read().decode("utf-8", errors="replace")
            if "NoGpuAvailable:" in error_tail:
                publish("WAITING_FOR_GPU_BELOW_85_PERCENT", detail=error_tail[-800:])
                time.sleep(poll_seconds)
                continue
            publish("DEV_EVALUATION_FAILED", returncode=process.returncode,
                    error_tail=error_tail)
            raise RuntimeError(f"fixed DEV evaluation exited with {process.returncode}")
        if existing_evaluation() is None:
            publish("DEV_EVALUATION_FAILED", returncode=0,
                    detail="evaluator exited successfully without publishing a valid result")
            raise RuntimeError("fixed DEV evaluator published no valid result")


if __name__ == "__main__":
    run()
