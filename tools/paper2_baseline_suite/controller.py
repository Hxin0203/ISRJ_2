"""Server-side controller for the remaining baseline-suite work (server B only).

Runs unattended and records every step in CONTROLLER_STATUS.json so progress can
be read back later without a live session:
  1. wait for the E3 equal-Pfa queue to finish;
  2. RSDR-Net DEV256 + equal-Pfa (E7b);
  3. aggregate E3/E6/E7 into SUITE_SUMMARY.json / SUITE_TABLES.md;
  4. register + generate the E5 sweep, evaluate all methods, summarize;
  5. every 30 min: add newly completed ReCoVP-Hybrid seeds to E3 and E5 and
     re-aggregate, until all eight seeds are in or 96 h have passed.
GPU jobs pick a GPU below 85% utilization before each start and wait otherwise;
nothing running is stopped or moved. A failed step is recorded and skipped.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
SUITE = ROOT / "outputs" / "paper2_baseline_suite_20261126"
STATUS = SUITE / "CONTROLLER_STATUS.json"
LOGS = SUITE / "controller_logs"
PKG = "tools.paper2_baseline_suite"
DEADLINE_S = 96 * 3600
HYBRID_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
state = {"started_utc": None, "pid": os.getpid(), "steps": {}, "stage": "STARTING"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save() -> None:
    state["heartbeat_utc"] = now()
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, STATUS)


def mark(name: str, **kw) -> None:
    state["steps"].setdefault(name, {}).update(kw, updated_utc=now())
    save()


def _env(gpu=None) -> dict:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    return env


def run(name: str, args: list[str], gpu: bool) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    mark(name, state="RUNNING", started_utc=now())
    while True:
        device = None
        if gpu:
            from tools.server_gpu_admission import NoGpuAvailable, choose_gpu
            try:
                device = choose_gpu()["physical_gpu_index"]
            except NoGpuAvailable:
                mark(name, state="WAITING_GPU")
                time.sleep(60)
                continue
        log = LOGS / f"{name.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.call([sys.executable, "-m", *args], cwd=str(ROOT), env=_env(device),
                                   stdout=f, stderr=subprocess.STDOUT)
        if gpu and code == 3:
            mark(name, state="WAITING_GPU")
            time.sleep(60)
            continue
        mark(name, state="DONE" if code == 0 else "FAILED", exit_code=code, gpu=device, log=str(log))
        return code


def e3_done() -> bool:
    p = SUITE / "E3" / "QUEUE_STATUS.json"
    return p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("stage") == "QUEUE_DONE"


def completed_hybrid() -> list[int]:
    return [s for s in HYBRID_SEEDS
            if (ROOT / f"outputs/recovp_hybrid_seed{s}/seed_{s}/TRAINING_COMPLETE.json").is_file()]


def sweep_jobs() -> list:
    out = subprocess.check_output([sys.executable, "-m", f"{PKG}.sweep_suite", "jobs"], cwd=str(ROOT),
                                  env=_env(None), text=True)
    return json.loads(out.strip().splitlines()[-1])


def sweep_eval_all() -> None:
    jobs = sweep_jobs()
    cpu = []
    for method, seed in jobs:
        if method in ("Max-TF", "GMM-OMP"):
            args = [f"{PKG}.sweep_suite", "eval", "--method", method]
            log = LOGS / f"E5_{method}.log"
            LOGS.mkdir(parents=True, exist_ok=True)
            cpu.append((method, subprocess.Popen([sys.executable, "-m", *args], cwd=str(ROOT), env=_env(None),
                                                 stdout=open(log, "a", encoding="utf-8"), stderr=subprocess.STDOUT)))
            mark(f"E5/{method}", state="RUNNING_CPU", log=str(log))
    for method, seed in jobs:
        if method in ("Max-TF", "GMM-OMP"):
            continue
        args = [f"{PKG}.sweep_suite", "eval", "--method", method] + ([] if seed is None else ["--seed", str(seed)])
        run(f"E5/{method}/{seed}", args, gpu=True)
    for method, proc in cpu:
        code = proc.wait()
        mark(f"E5/{method}", state="DONE" if code == 0 else "FAILED", exit_code=code)


def guarded(name: str, fn) -> None:
    try:
        fn()
    except Exception:
        mark(name, state="FAILED", error=traceback.format_exc()[-3000:])


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    if STATUS.is_file():
        old = json.loads(STATUS.read_text(encoding="utf-8"))
        hb = old.get("heartbeat_utc")
        if old.get("stage") not in ("FINISHED",) and hb and \
                time.time() - time.mktime(time.strptime(hb, "%Y-%m-%dT%H:%M:%SZ")) + time.timezone < 900:
            raise RuntimeError("another controller looks alive (heartbeat < 15 min)")
        state["steps"] = old.get("steps", {})
    t_start = time.time()
    state["started_utc"] = now()
    state["stage"] = "WAIT_E3"
    save()
    while not e3_done():
        save()
        time.sleep(120)
    state["stage"] = "RSDR"
    run("E7b/RSDR-Net", [f"{PKG}.rsdr_suite", "work"], gpu=True)
    state["stage"] = "AGGREGATE"
    run("aggregate", [f"{PKG}.aggregate_suite"], gpu=False)
    state["stage"] = "E5_GENERATE"
    code = run("E5/generate", [f"{PKG}.sweep_suite", "generate"], gpu=False)
    if code == 0:
        state["stage"] = "E5_EVAL"
        guarded("E5/eval", sweep_eval_all)
        run("E5/summarize", [f"{PKG}.sweep_suite", "summarize"], gpu=False)
    state["stage"] = "FOLLOW_HYBRID"
    # Hybrid seeds that finished after the E3 queue started (e.g. seed 41) are added here once;
    # the queue skips every (family, seed) that already has a ROW.json.
    run("E3/hybrid_update", [f"{PKG}.equal_pfa_suite", "queue", "--family", "ReCoVP-Hybrid"], gpu=False)
    run("aggregate", [f"{PKG}.aggregate_suite"], gpu=False)
    seen = set(completed_hybrid())
    state["hybrid_seeds_included"] = sorted(seen)
    save()
    while time.time() - t_start < DEADLINE_S and len(seen) < len(HYBRID_SEEDS):
        time.sleep(1800)
        save()
        new = set(completed_hybrid()) - seen
        if not new:
            continue
        run("E3/hybrid_update", [f"{PKG}.equal_pfa_suite", "queue", "--family", "ReCoVP-Hybrid"], gpu=False)
        if code == 0:
            guarded("E5/eval_update", sweep_eval_all)
            run("E5/summarize", [f"{PKG}.sweep_suite", "summarize"], gpu=False)
        run("aggregate", [f"{PKG}.aggregate_suite"], gpu=False)
        seen |= new
        state["hybrid_seeds_included"] = sorted(seen)
        save()
    state["stage"] = "FINISHED"
    save()


if __name__ == "__main__":
    main()
