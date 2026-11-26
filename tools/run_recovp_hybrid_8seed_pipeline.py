"""Unattended, sequential continuation of ReCoVP-Hybrid training to 8 seeds.

Seed 41 is run with the existing frozen runner on physical GPU 0. Five new
seeds follow sequentially and use the extended runner's <85% GPU admission.
Each next seed starts only after a validated 256-epoch completion receipt.
The active seed-29 run is observed and included in final set-completion checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from paper2.io import ROOT, now, sha
from tools.durable_io import ProcessLock, atomic_json
from tools import server_gpu_admission


SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
QUEUE = (41, 53, 67, 79, 97, 109)
EPOCHS = 256
MAX_ATTEMPTS_PER_SEED = 8
BASE = ROOT / "evidence" / "outputs" / "recovp_hybrid_8seed_pipeline"


def _receipt(seed: int) -> Path:
    name = f"recovp_hybrid_seed{seed}"
    return ROOT / "evidence" / "outputs" / name / "TRAINING_COMPLETE.json"


def _validate_completion(seed: int) -> dict | None:
    path = _receipt(seed)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != "TRAINING_COMPLETE" or data.get("seed") != seed:
        raise RuntimeError(f"completion identity mismatch in {path}")
    if data.get("epochs") != EPOCHS or len(data.get("recovery_sha256", {})) != EPOCHS + 1:
        raise RuntimeError(f"incomplete epoch/recovery receipt in {path}")
    if set(data.get("recovery_sha256", {})) != {str(i) for i in range(EPOCHS + 1)}:
        raise RuntimeError(f"recovery epoch sequence is incomplete in {path}")
    return {"seed": seed, "receipt": str(path), "receipt_sha256": sha(path),
            "completed_utc": data.get("completed_utc"), "epochs": EPOCHS}


def _status(stage: str, **extra) -> None:
    value = {"stage": stage, "updated_utc": now(), "seeds": list(SEEDS),
             "epochs_per_seed": EPOCHS, "queue": list(QUEUE), **extra}
    atomic_json(BASE / "STATUS.json", value)
    atomic_json(ROOT / "outputs" / "recovp_hybrid_8seed_pipeline" / "STATUS.json", value)


def _evaluation_receipt(seed: int) -> dict | None:
    name = f"recovp_hybrid_seed{seed}"
    path = ROOT / "outputs" / name / f"seed_{seed}" / "dev_epoch_256_v1.json"
    if not path.exists():
        path = ROOT / "evidence" / "outputs" / name / f"seed_{seed}" / path.name
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    complete = _validate_completion(seed)
    if (value.get("stage") != "DEV_ONLY_RECOVP_HYBRID" or value.get("seed") != seed
            or value.get("epoch") != EPOCHS or value.get("parents") != 256
            or not complete or value.get("checkpoint_identity") !=
            json.loads(Path(complete["receipt"]).read_text(encoding="utf-8")).get("identity")):
        raise RuntimeError(f"invalid fixed DEV256 evaluation receipt for seed {seed}: {path}")
    return {"seed": seed, "receipt": str(path), "receipt_sha256": sha(path),
            "checkpoint_sha256": value.get("checkpoint_sha256"),
            "created_utc": value.get("created_utc")}


def _command(seed: int, poll_seconds: int) -> list[str]:
    py = sys.executable
    if seed == 41:
        return [py, "-m", "tools.start_recovp_hybrid_multiseed", "--seed", "41",
                "--gpu-index", "0", "--poll-seconds", str(poll_seconds)]
    return [py, "-m", "tools.start_recovp_hybrid_extended", "--seed", str(seed),
            "--poll-seconds", str(poll_seconds)]


def _evaluation_command(seed: int, poll_seconds: int) -> list[str]:
    py = sys.executable
    if seed in (29, 41):
        return [py, "-m", "tools.run_recovp_hybrid_seed", "--seed", str(seed),
                "--stage", "evaluate", "--poll-seconds", str(poll_seconds)]
    return [py, "-m", "tools.start_recovp_hybrid_extended_eval", "--seed", str(seed),
            "--poll-seconds", str(poll_seconds)]


def run(poll_seconds: int = 60, retry_seconds: int = 60) -> None:
    server_gpu_admission.require_server_identity()
    BASE.mkdir(parents=True, exist_ok=True)
    with ProcessLock(BASE / "PIPELINE.lock"):
        completed = {str(seed): _validate_completion(seed) for seed in SEEDS}
        if not completed.get("17"):
            raise RuntimeError("frozen seed17 completion receipt is missing; refusing to overwrite or assume completion")
        if not completed.get("29"):
            _status("WAIT_OR_MONITOR_SEED29", completed=completed,
                    note="seed29 is a separately running frozen task; the pipeline will wait for its verified receipt at finalization")
        for seed in QUEUE:
            finished = _validate_completion(seed)
            if finished:
                completed[str(seed)] = finished
                continue
            for attempt in range(1, MAX_ATTEMPTS_PER_SEED + 1):
                _status("RUNNING", current_seed=seed, attempt=attempt,
                        max_attempts=MAX_ATTEMPTS_PER_SEED, completed=completed,
                        gpu_admission="selected device must be below 85%; no existing process is stopped")
                command = _command(seed, poll_seconds)
                log_path = BASE / f"seed_{seed}_attempt_{attempt:02d}.log"
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write(f"\n[{now()}] launch: {command!r}\n")
                    proc = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                          stderr=subprocess.STDOUT, check=False)
                    log.write(f"[{now()}] exit_code={proc.returncode}\n")
                if proc.returncode == 0:
                    finished = _validate_completion(seed)
                    if not finished:
                        raise RuntimeError(f"seed {seed} exited successfully without a valid completion receipt")
                    completed[str(seed)] = finished
                    _status("SEED_COMPLETE", current_seed=seed, completed=completed)
                    break
                _status("RETRY_WAIT", current_seed=seed, attempt=attempt,
                        exit_code=proc.returncode, completed=completed,
                        retry_after_seconds=retry_seconds)
                if attempt < MAX_ATTEMPTS_PER_SEED:
                    time.sleep(retry_seconds)
            else:
                _status("FAILED_REQUIRES_ATTENTION", current_seed=seed,
                        completed=completed, attempts=MAX_ATTEMPTS_PER_SEED)
                raise RuntimeError(f"seed {seed} failed {MAX_ATTEMPTS_PER_SEED} attempts; queue stopped fail-closed")

        _status("WAIT_EXISTING_SEEDS", completed=completed,
                waiting_for=[seed for seed in (17, 29) if not _validate_completion(seed)])
        while True:
            for seed in (17, 29):
                finished = _validate_completion(seed)
                if finished:
                    completed[str(seed)] = finished
            if all(str(seed) in completed for seed in SEEDS):
                break
            _status("WAIT_EXISTING_SEEDS", completed=completed,
                    waiting_for=[seed for seed in (17, 29) if str(seed) not in completed])
            time.sleep(max(30, poll_seconds))
        evaluations = {}
        for seed in SEEDS:
            result = _evaluation_receipt(seed)
            if result:
                evaluations[str(seed)] = result
                continue
            for attempt in range(1, MAX_ATTEMPTS_PER_SEED + 1):
                _status("RUNNING_DEV256", current_seed=seed, attempt=attempt,
                        completed=completed, evaluated=evaluations)
                command = _evaluation_command(seed, poll_seconds)
                log_path = BASE / f"seed_{seed}_dev_attempt_{attempt:02d}.log"
                with log_path.open("a", encoding="utf-8", buffering=1) as log:
                    log.write(f"\n[{now()}] launch: {command!r}\n")
                    proc = subprocess.run(command, cwd=str(ROOT), stdout=log,
                                          stderr=subprocess.STDOUT, check=False)
                    log.write(f"[{now()}] exit_code={proc.returncode}\n")
                if proc.returncode == 0:
                    result = _evaluation_receipt(seed)
                    if not result:
                        raise RuntimeError(f"seed {seed} evaluation exited successfully without a valid DEV receipt")
                    evaluations[str(seed)] = result
                    break
                _status("DEV_RETRY_WAIT", current_seed=seed, attempt=attempt,
                        exit_code=proc.returncode, completed=completed, evaluated=evaluations,
                        retry_after_seconds=retry_seconds)
                if attempt < MAX_ATTEMPTS_PER_SEED:
                    time.sleep(retry_seconds)
            else:
                _status("DEV_FAILED_REQUIRES_ATTENTION", current_seed=seed,
                        completed=completed, evaluated=evaluations,
                        attempts=MAX_ATTEMPTS_PER_SEED)
                raise RuntimeError(f"seed {seed} DEV256 evaluation failed {MAX_ATTEMPTS_PER_SEED} attempts")
        _status("EIGHT_SEEDS_TRAINING_AND_DEV256_COMPLETE", completed=completed,
                evaluated=evaluations, verified_seed_count=len(completed),
                verified_evaluation_count=len(evaluations))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    run(args.poll_seconds, args.retry_seconds)


if __name__ == "__main__":
    main()
