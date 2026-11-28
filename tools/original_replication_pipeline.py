"""Persistent server-B pipeline for the ISRJ_2 replication priority queue."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "original_paper_replication_256_v1" / "pipeline"
STATUS = OUT / "PIPELINE_STATUS.json"
LOGS = OUT / "logs"

TASKS = [
    {"id": "P2_AUDIT", "deps": [], "gpu": False,
     "command": ["tools.original_replication_p2"]},
    {"id": "P2_CALIBRATION", "deps": ["P2_AUDIT"], "gpu": False,
     "command": ["tools.paper2_baseline_suite.equal_pfa_suite", "queue",
                  "--family", "ReCoVP-Hybrid", "--seed", "67", "--seed", "97",
                  "--seed", "53", "--seed", "41", "--seed", "109"]},
    {"id": "R3_MAIN", "deps": ["P2_CALIBRATION"], "gpu": False,
     "command": ["tools.original_replication_suite", "queue"]},
    {"id": "R3_SUMMARY", "deps": ["R3_MAIN"], "gpu": False,
     "command": ["tools.original_replication_suite", "summarize"]},
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _receipt(task_id: str) -> bool:
    if task_id == "P2_AUDIT":
        return (ROOT / "outputs/original_paper_replication_256_v1/P2/P2_COMPLETE.json").is_file()
    if task_id == "P2_CALIBRATION":
        rows = [ROOT / f"outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_{seed}/ROW.json"
                for seed in (67, 97, 53, 41, 109)]
        return all(path.is_file() for path in rows)
    if task_id == "R3_MAIN":
        return (ROOT / "outputs/original_paper_replication_256_v1/R3_main/ALL_COMPLETE.json").is_file()
    return (ROOT / "outputs/original_paper_replication_256_v1/R3_main/SWEEP_SUMMARY.json").is_file()


def _pid_alive(pid: int) -> bool:
    result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"] , capture_output=True, text=True)
    return str(pid) in result.stdout


def _state() -> dict:
    if STATUS.is_file():
        value = json.loads(STATUS.read_text(encoding="utf-8"))
    else:
        value = {"created_utc": now(), "tasks": {}}
    value.setdefault("tasks", {})
    for task in TASKS:
        value["tasks"].setdefault(task["id"], {"state": "PENDING"})
    return value


def _env() -> dict:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ""}


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("pipeline runs only on server B")
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = _state()
    state["controller_pid"] = os.getpid()
    state["started_utc"] = state.get("started_utc", now())
    running = {}
    while True:
        for task in TASKS:
            tid = task["id"]
            record = state["tasks"][tid]
            if record.get("state") == "DONE" or _receipt(tid):
                record.update({"state": "DONE", "receipt_verified_utc": record.get("receipt_verified_utc", now())})
                continue
            pid = record.get("pid")
            if record.get("state") == "RUNNING" and pid:
                proc = running.get(pid)
                if proc is None:
                    if _pid_alive(int(pid)):
                        continue
                    record.update({"state": "PENDING", "reconciled_utc": now()})
                else:
                    continue
            if any(state["tasks"][dep].get("state") != "DONE" for dep in task["deps"]):
                continue
            log = LOGS / f"{tid}.log"
            handle = log.open("a", encoding="utf-8")
            command = [sys.executable, "-m", *task["command"]]
            proc = subprocess.Popen(command, cwd=str(ROOT), env=_env(), stdout=handle,
                                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            running[proc.pid] = proc
            record.update({"state": "RUNNING", "pid": proc.pid, "started_utc": now(), "log": str(log),
                           "command": command})
            write_json(STATUS, state)
            break
        for pid, proc in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            running.pop(pid)
            tid = next((t["id"] for t in TASKS if state["tasks"][t["id"]].get("pid") == pid), None)
            if tid is None:
                continue
            state["tasks"][tid].update({"state": "DONE" if code == 0 else "FAILED",
                                         "exit_code": code, "finished_utc": now()})
            write_json(STATUS, state)
        state["heartbeat_utc"] = now()
        write_json(STATUS, state)
        if all(state["tasks"][task["id"]].get("state") == "DONE" for task in TASKS):
            state["state"] = "COMPLETE"
            state["finished_utc"] = now()
            write_json(STATUS, state)
            return
        if any(state["tasks"][task["id"]].get("state") == "FAILED" for task in TASKS):
            state["state"] = "FAILED"
            write_json(STATUS, state)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
