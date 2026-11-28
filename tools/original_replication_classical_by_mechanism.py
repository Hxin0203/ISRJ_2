"""Recover the R3 classical baselines with separate jammer-mechanism arrays."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tools.durable_io import atomic_json, atomic_write
from tools.stable_hash import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/original_paper_replication_256_v1/R3_by_mechanism"
DATA = ROOT / "data/original_paper_replication_256_v1/R3_main"
MODULE = "tools.original_replication_classical_by_mechanism"
METHODS = {"Max-TF": "max_tf_isrj1", "GMM-OMP": "gmm_omp"}
SNRS = (-30, -25, -20, -15, -10, -5, 0)
JSRS = (0, 10, 20, 30, 35)
CHUNK = 16


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def register():
    from tools.server_gpu_admission import require_server_identity
    require_server_identity()
    sources = ["tools/original_replication_classical_by_mechanism.py",
               "tools/paper2_baseline_suite/classical_suite.py",
               "tools/paper2_baseline_suite/wp5b_suite.py",
               "tools/paper2_baseline_suite/equal_pfa_suite.py",
               "paper2/baselines_1103.py", "paper2/evaluation_1103_v4.py", "paper2/radar_eval.py"]
    record = {
        "version": 1, "work_package": "R3", "methods": METHODS,
        "snr_db": SNRS, "jsr_db": JSRS, "physical_parents": 256,
        "jammer_types": ["ISDRJ", "ISRRJ", "ISCRJ"],
        "data_manifest_sha256": sha(DATA / "DATA_MANIFEST.json"),
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "reason": "Original R3 files averaged the three mechanisms before persistence",
        "metric_shape": "physical parent, jammer mechanism",
        "thresholds": "nominal cell Pfa 1e-6 and existing independently calibrated frame thresholds",
    }
    # JSON roundtrip puts immutable tuples into their on-disk list form.
    record = json.loads(json.dumps(record))
    path = OUT / "REGISTRATION.json"
    if path.exists() and read(path) != record:
        raise RuntimeError("R3 mechanism-recovery registration differs")
    if not path.exists():
        atomic_json(path, record)
    return record


def worker(method):
    import numpy as np
    import torch
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite import classical_suite as C
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite.wp5b_suite import dev_metrics_by_type

    registration = register()
    torch.set_num_threads(4)
    manifest = read(DATA / "DATA_MANIFEST.json")
    row_path = ROOT / "outputs/paper2_baseline_suite_20261126/E6/equal_pfa" / METHODS[method] / "ROW.json"
    row = read(row_path)
    if sha(E.CAL) != row["calibration_sha256"]:
        raise RuntimeError("classical calibration data SHA mismatch")
    thresholds = {str(pfa): next(float(c["threshold_ratio"]) for c in row["curve"]
                               if c["target_frame_pfa"] == pfa) for pfa in (.01, .05, .1)}
    folder = OUT / method
    binding = {"registration_sha256": sha(OUT / "REGISTRATION.json"),
               "calibration_row_sha256": sha(row_path), "thresholds": thresholds, "method": method}
    binding_path = folder / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise RuntimeError("classical recovery identity differs")
    atomic_json(binding_path, binding)
    binding_sha = sha(binding_path)
    status = {"state": "RUNNING", "pid": os.getpid(), "method": method,
              "started_utc": now(), "completed_cells": 0, "total_cells": 35,
              "completed_chunks": 0, "registration": registration}
    atomic_json(folder / "STATUS.json", status)
    parent_reference = None
    for snr in SNRS:
        for jsr in JSRS:
            name = f"snr_{snr}_jsr_{jsr}"
            entry = manifest["cells"][name]
            source = Path(entry["path"])
            if sha(source) != entry["sha256"]:
                raise RuntimeError(f"classical cell SHA mismatch {source}")
            with np.load(source, allow_pickle=False) as z:
                s, j, e = (z[key] for key in ("s", "j", "epsilon"))
                parents, doppler = z["parent_id"].copy(), z["doppler_hz"].copy()
            if s.shape != (256, 3, 8, 1800):
                raise RuntimeError("classical cell shape mismatch")
            if parent_reference is not None and not np.array_equal(parents, parent_reference):
                raise RuntimeError("classical cells are not parent-paired")
            parent_reference = parents
            target = target_peak_range(s[:, 0])
            cell = folder / name
            for lo in range(0, len(parents), CHUNK):
                hi = min(lo + CHUNK, len(parents))
                path = cell / f"parents_{lo:04d}_{hi:04d}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        if str(z["binding_sha256"]) != binding_sha or not np.array_equal(z["parent_id"], parents[lo:hi]):
                            raise RuntimeError(f"classical chunk binding mismatch {path}")
                else:
                    states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
                    restored = C.restore(METHODS[method], states, j[lo:hi])
                    values = dev_metrics_by_type(s[lo:hi], j[lo:hi], e[lo:hi], restored, doppler[lo:hi])
                    scores = E.frame_scores(restored, s[lo:hi], e[lo:hi], doppler[lo:hi], target[lo:hi], True)
                    values.update({f"frame_{key}": value for key, value in scores.items()})
                    for pfa, threshold in thresholds.items():
                        values[f"pd_eq_{pfa}"] = (scores["target"] > threshold).astype(np.float64)
                        values[f"fa_eq_{pfa}"] = (scores["h0"] > threshold).astype(np.float64)
                    values.update({"E_U": values["u_nmse"], "D_S": values["damage"],
                                   "e_S": values["damage"], "E_0": values["no_jammer_pooled_nmse"],
                                   "Hits": values["pd"].astype(np.int64),
                                   "FA_cells": np.rint(values["cell_pfa_jammer"] * 9032).astype(np.int64)})
                    if not all(value.shape == (hi-lo, 3) and np.isfinite(value).all() for value in values.values()):
                        raise RuntimeError("invalid classical metric dimensions or values")
                    atomic_write(path, lambda stream: np.savez_compressed(
                        stream, parent_id=parents[lo:hi], binding_sha256=binding_sha, **values))
                status.update({"snr_db": snr, "jsr_db": jsr, "last_parent_stop": hi,
                               "updated_utc": now(), "completed_chunks": status["completed_chunks"] + 1})
                atomic_json(folder / "STATUS.json", status)
            atomic_json(cell / "DONE.json", {"state": "COMPLETED_VERIFIED", "parents": len(parents),
                                             "binding_sha256": binding_sha, "finished_utc": now()})
            status["completed_cells"] += 1
            atomic_json(folder / "STATUS.json", status)
            print(method, name, status["completed_cells"], "/35", flush=True)
    status.update({"state": "COMPLETED_VERIFIED", "finished_utc": now()})
    atomic_json(folder / "DONE.json", status)


def processes(method):
    import psutil
    result = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        command = proc.info["cmdline"] or []
        if MODULE in command and "worker" in command and "--method" in command:
            index = command.index("--method")
            if index + 1 < len(command) and command[index+1] == method:
                result.append(proc.pid)
    return result


def queue():
    from tools.server_gpu_admission import query_resources
    register()
    queue_path = OUT / "QUEUE_STATUS.json"
    state = read(queue_path) if queue_path.exists() else {"tasks": {}}
    children = {}
    while True:
        for method in METHODS:
            task = state["tasks"].setdefault(method, {"state": "PENDING", "attempts": 0})
            done_path = OUT / method / "DONE.json"
            if done_path.exists():
                done = read(done_path)
                if done.get("completed_cells") != 35:
                    raise RuntimeError("incomplete classical done receipt")
                task["state"] = "COMPLETED_VERIFIED"
                continue
            live = processes(method)
            if live:
                task.update({"state": "RUNNING", "pids": live})
                continue
            if task.get("attempts", 0) >= 3:
                task["state"] = "FAILED"
                continue
            if time.time() < task.get("retry_after_unix", 0):
                continue
            resources = query_resources()
            if resources["memory"]["utilization_percent"] >= 90:
                task["state"] = "WAITING_MEMORY"
                continue
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            with (OUT / f"{method}.log").open("a", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "worker", "--method", method],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children[method] = proc
            task.update({"state": "STARTING", "pid": proc.pid, "started_utc": now(),
                         "attempts": task.get("attempts", 0) + 1, "dispatch_resources": resources})
            atomic_json(queue_path, state)
        for method, proc in list(children.items()):
            code = proc.poll()
            if code is not None:
                children.pop(method)
                state["tasks"][method].update({"exit_code": code, "retry_after_unix": time.time() + 60})
        state.update({"controller_pid": os.getpid(), "heartbeat_utc": now()})
        atomic_json(queue_path, state)
        if all(task["state"] == "COMPLETED_VERIFIED" for task in state["tasks"].values()):
            atomic_json(OUT / "CLASSICAL_COMPLETE.json", state)
            return
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "queue", "worker"))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "queue":
        queue()
    else:
        try:
            worker(args.method)
        except Exception:
            atomic_json(OUT / args.method / "LAST_ERROR.json", {"at_utc": now(), "error": traceback.format_exc()})
            raise
