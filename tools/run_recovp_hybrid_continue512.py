"""Wait for all eight 256-epoch runs, then dispatch their isolated continuations."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

from paper2.io import ROOT, now, read, sha
from tools import server_gpu_admission as admission
from tools.durable_io import ProcessLock, atomic_json, process_alive
from tools.recovp_hybrid_continue512 import (
    ARCHIVE, EXTRA_SOURCES, PIPELINE, SEEDS, STATE, parent_ready, paths,
)


POLL_SECONDS = 30
MAX_FAILURES = 5
RETRY_SECONDS = 120


def write_status(status: dict) -> None:
    status["updated_utc"] = now()
    status["controller_pid"] = os.getpid()
    atomic_json(STATE / "STATUS.json", status)


def complete(seed: int) -> bool:
    base, evidence, folder = paths(seed, True)
    targets = (folder / "TRAINING_COMPLETE.json", evidence / "TRAINING_COMPLETE.json")
    if not all(path.is_file() for path in targets):
        return False
    result = read(targets[0])
    config_path = base / "CONFIG.json"
    checkpoint = folder / "recovery" / "epoch_512.pt"
    return (sha(targets[0]) == sha(targets[1]) and result.get("seed") == seed
            and result.get("epochs") == 512 and len(result.get("recovery_sha256", {})) == 257
            and result["identity"]["config_sha256"] == sha(config_path)
            and sha(checkpoint) == result["recovery_sha256"]["512"]
            and read(checkpoint.with_suffix(".json"))["identity"] == result["identity"])


def matching_worker(pid: int, seed: int) -> bool:
    """Check command identity as well as liveness before reattaching a worker."""
    if not process_alive(pid):
        return False
    command = (f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
               "if ($p) { $p.CommandLine }")
    output = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", command], text=True)
    return ("tools.recovp_hybrid_continue512" in output
            and f"--seed {seed} " in output and "--verify-only" not in output)


def start(seed: int, gpu: int, entry: dict) -> subprocess.Popen:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, "-m", "tools.recovp_hybrid_continue512",
               "--seed", str(seed), "--gpu-index", str(gpu)]
    attempt = len(entry["attempts"]) + 1
    log_path = STATE / f"seed_{seed}_attempt_{attempt}.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    entry["attempts"].append({"pid": process.pid, "gpu": gpu, "started_utc": now(),
                              "log": str(log_path), "command": command})
    entry.update(stage="RUNNING", pid=process.pid, gpu=gpu)
    return process


def run() -> dict:
    admission.require_server_identity()
    STATE.mkdir(parents=True, exist_ok=True)
    with ProcessLock(STATE / "PIPELINE.lock"):
        status_path = STATE / "STATUS.json"
        if status_path.exists():
            status = read(status_path)
        else:
            status = {"pipeline": PIPELINE, "stage": "WAITING_FOR_ALL_256",
                      "seeds": {str(seed): {"stage": "PENDING", "attempts": [], "failures": 0}
                                for seed in SEEDS},
                      "sources": {rel: sha(ROOT / rel) for rel in EXTRA_SOURCES},
                      "archive": str(ARCHIVE), "target_epochs": 512,
                      "lr": 2e-6, "barrier": "all eight original seeds complete epoch256",
                      "admission": admission.ADMISSION_POLICY, "created_utc": now()}
        for rel, digest in status["sources"].items():
            if sha(ROOT / rel) != digest:
                raise RuntimeError(f"controller source changed: {rel}")
        running = {}
        while True:
            ready = {str(seed): parent_ready(seed) for seed in SEEDS}
            status["parent256_complete"] = ready
            # Archive completed parents while the last original seed is still running.
            if not status.get("parent_archives_verified_utc"):
                archived = status.setdefault("parent_archived_seeds", [])
                for seed in SEEDS:
                    if not ready[str(seed)] or seed in archived:
                        continue
                    status["stage"] = "ARCHIVING_COMPLETED_256"
                    status["archiving_seed"] = seed
                    write_status(status)
                    audit_path = STATE / f"PARENT_AUDIT_seed{seed}.json"
                    command = [sys.executable, "-m", "tools.recovp_hybrid_continue512",
                               "--seed", str(seed), "--audit-only"]
                    with (STATE / f"archive_seed{seed}.log").open("ab", buffering=0) as log:
                        subprocess.run(command, cwd=str(ROOT), stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    status["seeds"][str(seed)]["parent_sha256"] = read(audit_path)["parent_sha256"]
                    archived.append(seed)
                    write_status(status)
                status.pop("archiving_seed", None)
                if len(archived) == len(SEEDS):
                    status["parent_archives_verified_utc"] = now()
            if not all(ready.values()):
                status["stage"] = "WAITING_FOR_ALL_256"
                write_status(status)
                time.sleep(POLL_SECONDS)
                continue

            for seed in SEEDS:
                entry = status["seeds"][str(seed)]
                if entry["stage"] == "TRAINING_COMPLETE":
                    continue
                if complete(seed):
                    entry.update(stage="TRAINING_COMPLETE", completed_utc=now())
                    running.pop(seed, None)
                    continue
                process = running.get(seed)
                if process is not None:
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    running.pop(seed)
                elif entry.get("pid") and matching_worker(entry["pid"], seed):
                    entry["stage"] = "RUNNING"
                    continue
                elif entry["stage"] == "RUNNING":
                    exit_code = None
                else:
                    continue
                if entry["attempts"]:
                    entry["attempts"][-1].update(exit_code=exit_code, ended_utc=now())
                if exit_code != 75:
                    entry["failures"] += 1
                entry.update(stage="RETRY_WAIT", pid=None,
                             retry_after_unix=time.time() + RETRY_SECONDS)

            pending = []
            for seed in SEEDS:
                entry = status["seeds"][str(seed)]
                if entry["failures"] >= MAX_FAILURES:
                    entry["stage"] = "NEEDS_ATTENTION"
                if entry["stage"] in ("PENDING", "RETRY_WAIT"):
                    if time.time() >= entry.get("retry_after_unix", 0):
                        pending.append(seed)
            if all(entry["stage"] == "TRAINING_COMPLETE" for entry in status["seeds"].values()):
                status["stage"] = "TRAINING_COMPLETE"
                status["completed_utc"] = now()
                write_status(status)
                return status
            status["stage"] = "RUNNING"
            if pending:
                try:
                    samples = admission.query_gpus()
                    status["gpu_samples"] = samples
                    for seed in pending:
                        entry = status["seeds"][str(seed)]
                        base, _, _ = paths(seed, True)
                        config_path = base / "CONFIG.json"
                        if config_path.exists():
                            gpu = int(read(config_path)["execution"]["physical_gpu_index"])
                            if samples[gpu]["utilization_percent"] >= admission.UTILIZATION_LIMIT:
                                continue
                        else:
                            eligible = [sample for sample in samples
                                        if sample["utilization_percent"] < admission.UTILIZATION_LIMIT]
                            if not eligible:
                                raise admission.NoGpuAvailable("both GPUs at or above 85 percent")
                            dispatch_counts = {gpu: sum(job.get("gpu") == gpu
                                                       for job in status["seeds"].values())
                                               for gpu in admission.GPU_UUIDS}
                            selected = min(eligible, key=lambda sample: (
                                sample["utilization_percent"], dispatch_counts[sample["physical_gpu_index"]],
                                sample["physical_gpu_index"]))
                            gpu = selected["physical_gpu_index"]
                        running[seed] = start(seed, gpu, entry)
                        write_status(status)
                        # Re-query after this worker has had time to initialize CUDA.
                        break
                except admission.NoGpuAvailable as exc:
                    status["resource_wait"] = str(exc)
            if any(entry["stage"] == "NEEDS_ATTENTION" for entry in status["seeds"].values()):
                status["stage"] = "NEEDS_ATTENTION_WITH_OTHER_SEEDS_CONTINUING"
            write_status(status)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    for controller_attempt in range(1, MAX_FAILURES + 1):
        try:
            print(run(), flush=True)
            break
        except Exception:
            traceback.print_exc()
            atomic_json(STATE / "CONTROLLER_ERROR.json", {
                "stage": "RETRYING_CONTROLLER" if controller_attempt < MAX_FAILURES else "FAILED",
                "attempt": controller_attempt, "error": traceback.format_exc(),
                "updated_utc": now()})
            if controller_attempt == MAX_FAILURES:
                raise
            time.sleep(RETRY_SECONDS)
