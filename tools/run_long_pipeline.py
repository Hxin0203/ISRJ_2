"""Run the authorized frozen TEST -> audit/report -> eight-seed training queue.

Reinvoke this module after a reboot. Completed stages and epoch archives are
retained. Only bounded retries of mechanical failures are automatic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

from paper2.io import ROOT, now, read, sha
from tools.durable_io import ProcessLock, atomic_copy, atomic_json
from tools.resume_scan_1103 import verify_frozen_sources

PIPELINE_ID = "paper2_long_pipeline_20261114"
BASE = ROOT / "outputs" / PIPELINE_ID
LIGHT = ROOT / "evidence/outputs" / PIPELINE_ID
CONFIG = ROOT / "configs" / f"{PIPELINE_ID}.json"
STAGES = (
    ("frozen_test_resume", ["-m", "tools.resume_scan_1103"]),
    ("classical_receipts_verify", ["-m", "tools.resume_scan_1103", "--classical"]),
    ("complete_test_hash_audit", ["-m", "tools.aggregate_1103"]),
    ("test_pulse_compression_doppler_plots", ["-m", "tools.plot_1103"]),
    ("test_report", ["-m", "tools.report_1103"]),
    ("eight_seed_training_and_dev", ["-m", "paper2.train_256_8seed"]),
    ("final_status_report", ["-m", "tools.report_1103"]),
)


def frozen_config():
    verify_frozen_sources(read(ROOT / "configs/paper2_1103_scan_freeze.json"))
    paths = list((ROOT / "paper2").rglob("*.py"))
    paths += [ROOT / name for name in (
        "tools/run_long_pipeline.py", "tools/durable_io.py", "tools/resume_scan_1103.py",
        "tools/aggregate_1103.py", "tools/plot_1103.py", "tools/report_1103.py",
        "tools/aggregate_convergence_256_8seed.py", "tools/training_progress_256.py",
        "configs/paper2_1103_scan_freeze.json")]
    value = dict(pipeline_id=PIPELINE_ID, stages=[dict(name=name, args=args) for name,args in STAGES],
                 seeds=[17,29,41,53,67,79,97,109], epochs=256, milestone_epochs=list(range(32,257,32)),
                 retry_policy="at most 3 attempts/stage for mechanical I/O or OS process termination only; identity/numerical failures stop",
                 source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths})
    if CONFIG.exists():
        if read(CONFIG) != value:
            raise RuntimeError("pipeline source/config binding changed; inspect before restarting")
    else:
        atomic_json(CONFIG, value)
    return value


def publish(value):
    value = {**value, "updated_utc": now(), "controller_pid": os.getpid(),
             "pipeline_id": PIPELINE_ID, "config_sha256": sha(CONFIG)}
    atomic_json(BASE / "STATUS.json", value)
    atomic_json(LIGHT / "STATUS.json", value)


def retryable_failure(stderr, code):
    text = stderr.read_text(encoding="utf-8", errors="replace")
    # Never automatically repeat scientific/identity/assertion failures.
    if any(s in text for s in ("AssertionError", "nonfinite", "identity changed", "hash mismatch", "source mismatch")):
        return False
    return (any(s in text for s in ("PermissionError", "WinError 5", "WinError 32", "WinError 33"))
            or code < 0 or code in (0xC000013A, 0xC0000005))


def verify_stage(name):
    if name == "complete_test_hash_audit":
        result = read(ROOT / "evidence/outputs/TEST_PAIRED_AGGREGATES_1103.json")
        if not result["all_expected_complete"]:
            raise RuntimeError("TEST coverage incomplete; extension training has not been started")
    if name == "test_pulse_compression_doppler_plots":
        folder = ROOT / "outputs/plots_1103_20261114"
        images = list(folder.glob("*.png"))
        if not images or not (folder / "typical_parent0_snr0_jsr20.png").is_file():
            raise RuntimeError("TEST plot artifacts missing")
        # Share figures and small scalar tables, never parent arrays.
        index = {}
        for path in folder.iterdir():
            if path.suffix.lower() in (".png", ".csv", ".json"):
                atomic_copy(path, LIGHT / "test_figures" / path.name)
                index[path.name] = sha(path)
        atomic_json(LIGHT / "TEST_FIGURE_MANIFEST.json", dict(files_sha256=index, created_utc=now()))
    if name == "eight_seed_training_and_dev":
        run = ROOT / "outputs/paper2_256_epoch_8seed_20261114"
        if read(run / "STATUS.json")["stage"] != "PIPELINE_COMPLETE":
            raise RuntimeError("training/DEV convergence artifacts have not completed")
        atomic_copy(run / "TRAINING_COMPLETE.json", LIGHT / "EIGHT_SEED_TRAINING_COMPLETE.json")


def run():
    BASE.mkdir(parents=True, exist_ok=True)
    with ProcessLock(BASE / "CONTROLLER.lock"):
        frozen_config()
        ledger_path = BASE / "STAGES.json"
        ledger = read(ledger_path) if ledger_path.exists() else {}
        try:
            for index, (name, args) in enumerate(STAGES):
                if ledger.get(name, {}).get("complete"):
                    continue
                for attempt in range(1,4):
                    tag = f"{index:02d}_{name}_{time.time_ns()}"
                    stdout, stderr = BASE / f"{tag}.stdout.log", BASE / f"{tag}.stderr.log"
                    command = [sys.executable, *args]
                    with stdout.open("wb") as out, stderr.open("wb") as err:
                        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
                        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, env=env)
                        while True:
                            publish(dict(stage=name, stage_index=index+1, stages=len(STAGES), attempt=attempt,
                                         child_pid=child.pid, child_running=child.poll() is None,
                                         command=command, completed_stages=list(ledger)))
                            try:
                                code = child.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                continue
                    if code == 0:
                        verify_stage(name)
                        ledger[name] = dict(complete=True, completed_utc=now(), command=command,
                                            stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
                        atomic_json(ledger_path, ledger)
                        atomic_json(LIGHT / "STAGES.json", ledger)
                        break
                    failure = dict(stage=name, attempt=attempt, exit_code=code,
                                   stderr_tail=stderr.read_text(encoding="utf-8", errors="replace")[-6000:],
                                   created_utc=now(), stderr_sha256=sha(stderr))
                    atomic_json(BASE / f"FAILURE_{time.time_ns()}.json", failure)
                    atomic_json(LIGHT / f"FAILURE_{time.time_ns()}.json", failure)
                    if attempt == 3 or not retryable_failure(stderr, code):
                        raise RuntimeError(f"stage {name} exited {code}; retained failure evidence")
                    time.sleep(5)
            publish(dict(stage="COMPLETE", completed_stages=list(ledger), child_running=False))
            atomic_json(BASE / "COMPLETE.json", dict(completed_utc=now(), config_sha256=sha(CONFIG), stages=ledger))
        except BaseException as exc:
            failure = dict(stage="STOPPED_WITH_FAILURE", error=repr(exc), traceback=traceback.format_exc(),
                           child_running=False, completed_stages=list(ledger))
            publish(failure)
            raise


if __name__ == "__main__":
    run()
