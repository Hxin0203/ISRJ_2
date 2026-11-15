"""Serial GPU queue, bounded interruption recovery and held-back confirmation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
import numpy as np

from paper2.io import ROOT, now, read, sha
from paper2.data_1103 import generate
from tools.durable_io import ProcessLock, atomic_json
from .protocol import BASE, CONFIG, EVIDENCE, SEEDS, freeze, validate_config, folder_for
from .evaluate import evaluate
from .models import VARIANTS
from .runtime import require_server, user_control, admit, begin_budget, end_budget, budget_remaining


def run(budget):
    require_server()
    BASE.mkdir(parents=True, exist_ok=True)
    cfg = validate_config() if CONFIG.exists() else freeze()
    # Reuse weights, but measure epoch-256 DEV under this server's shared FP32 policy.
    for seed in SEEDS:
        admit(BASE, EVIDENCE)
        evaluate("full", seed, 256)
    for index, job in enumerate(cfg["queue"]):
        user_control(BASE)
        if budget_remaining(budget) <= 0:
            raise RuntimeError("Study cumulative 96-hour budget exhausted")
        folder = folder_for(job["variant"], job["seed"])
        folder.mkdir(parents=True, exist_ok=True)
        # Re-invoke completed jobs to validate receipts; trainer does no extra updates.
        for attempt in range(1, 4):
            admit(BASE, EVIDENCE)
            stamp = time.time_ns()
            stdout = folder / f"process_{stamp}.stdout.log"
            stderr = folder / f"process_{stamp}.stderr.log"
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen([sys.executable, "-m", "tools.unet_ablation.train",
                                            "--variant", job["variant"], "--seed", str(job["seed"])],
                                           cwd=ROOT, stdout=out, stderr=err)
                status = {"stage": "TRAINING_QUEUE", "job_index": index + 1, "jobs": len(cfg["queue"]),
                          **job, "attempt": attempt, "controller_pid": os.getpid(), "worker_pid": process.pid,
                          "config_sha256": sha(CONFIG), "updated_utc": now()}
                atomic_json(BASE / "QUEUE_STATUS.json", status)
                atomic_json(EVIDENCE / "QUEUE_STATUS.json", status)
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        from .runtime import resource_sample
                        try:
                            atomic_json(BASE / "RESOURCE_STATUS.json", resource_sample())
                        except OSError as exc:
                            # A monitoring failure must not abandon a healthy child.
                            atomic_json(BASE / "RESOURCE_MONITOR_ERROR.json", {"error": repr(exc), "utc": now()})
            if code == 0 and (folder / "TRAINING_COMPLETE.json").exists():
                break
            failure = read(folder / "FAILURE.json") if (folder / "FAILURE.json").exists() else {}
            current_failure = failure.get("pid") == process.pid
            if current_failure and not failure.get("retryable", False):
                raise RuntimeError(f"non-retryable training failure: {job}, {failure.get('error')}; see {stderr}")
            if attempt == 3:
                raise RuntimeError(f"training stopped after three interrupted/transient attempts: {job}; see {stderr}")
            time.sleep(5)
    complete = {"completed_utc": now(), "config_sha256": sha(CONFIG),
                "runs": [{**job, "receipt_sha256": sha(folder_for(job["variant"], job["seed"]) / "TRAINING_COMPLETE.json")}
                         for job in cfg["queue"]]}
    atomic_json(BASE / "ALL_TRAINING_COMPLETE.json", complete)
    atomic_json(EVIDENCE / "ALL_TRAINING_COMPLETE.json", complete)
    conf = cfg["confirmation"]
    path, receipt = generate("test", count=conf["parents"], seed=conf["seed"], out_dir=ROOT / conf["directory"])
    metadata = read(receipt)
    if (metadata["seed"] != conf["seed"] or metadata["count"] != conf["parents"]
            or metadata["role"] != "test" or metadata["snr_db"] is not None
            or metadata["jsr_db"] is not None or metadata["sha256"] != sha(path)):
        raise RuntimeError("confirmation generation identity mismatch")
    with np.load(path, allow_pickle=False) as z:
        confirmation_ids = z["parent_id"].copy()
    for role in ("train", "dev"):
        with np.load(ROOT / "data" / "paper2_1103" / f"{role}.npz", allow_pickle=False) as z:
            if np.intersect1d(confirmation_ids, z["parent_id"]).size:
                raise RuntimeError("confirmation parent overlap")
    identity = {"scope": "CONFIRMATION", "path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                "receipt_sha256": sha(receipt), "config_sha256": sha(CONFIG)}
    atomic_json(BASE / "CONFIRMATION_DATA.json", identity)
    atomic_json(EVIDENCE / "CONFIRMATION_DATA.json", identity)
    for variant in ("full", *VARIANTS):
        for seed in SEEDS:
            atomic_json(BASE / "QUEUE_STATUS.json", {"stage": "CONFIRMATION", "variant": variant,
                        "seed": seed, "controller_pid": os.getpid(), "updated_utc": now()})
            evaluate(variant, seed, 256, data_path=path, scope="CONFIRMATION")
    from .summarize import refresh
    result = refresh()
    if (len(result["training_epochs"]) != len(cfg["queue"]) * 256
            or len(result["dev_curves"]) != len(cfg["queue"]) * 8
            or {r["variant"] for r in result["confirmation_epoch256_comparison"]} != set(VARIANTS)
            or {r["variant"] for r in result["dev_epoch256_comparison"]} != set(VARIANTS)):
        raise RuntimeError("Final completeness audit failed; preserve all partial results")
    status = {"stage": "PIPELINE_COMPLETE", "completed_utc": now(), "config_sha256": sha(CONFIG),
              "summary_sha256": sha(BASE / "SUMMARY.json"), "trained_runs": 15,
              "confirmation_comparisons": len(result["confirmation_epoch256_comparison"])}
    atomic_json(BASE / "PIPELINE_COMPLETE.json", status)
    atomic_json(BASE / "QUEUE_STATUS.json", status)
    atomic_json(EVIDENCE / "PIPELINE_COMPLETE.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        cfg = freeze()
        print(f"frozen {len(cfg['queue'])} jobs config_sha256={sha(CONFIG)}", flush=True)
        return
    require_server()
    # A rejected second controller must never overwrite the active controller's status.
    with ProcessLock(BASE / "CONTROLLER.lock"):
        budget = begin_budget(BASE)
        try:
            run(budget)
        except BaseException as exc:
            failure = {"stage": "STOPPED", "utc": now(), "error": repr(exc), "traceback": traceback.format_exc(), "pid": os.getpid()}
            atomic_json(BASE / "CONTROLLER_FAILURE.json", failure)
            atomic_json(BASE / "QUEUE_STATUS.json", failure)
            atomic_json(EVIDENCE / "CONTROLLER_FAILURE.json", failure)
            raise
        finally:
            end_budget(BASE, budget)


if __name__ == "__main__":
    main()
