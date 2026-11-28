"""Server-B R5 detection grid for the five frozen ReCoVP-Hybrid endpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from tools.durable_io import atomic_json, atomic_write
from tools.stable_hash import sha as stable_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/original_paper_replication_256_v1/R5_detection"
DATA = ROOT / "data/original_paper_replication_256_v1/R5_detection"
MASTER = ROOT / "data/hybrid_condition_master_20261127"
SELECTION = ROOT / "reports/Original_paper_replication_256/SELECTED_SEEDS.json"
SEEDS = (67, 97, 53, 41, 109)
PRIORITY_SNRS = tuple(range(-30, -19)) + (-15, -10, -5, 0)
JSRS = tuple(range(-10, 36, 5))
SNRS = PRIORITY_SNRS + tuple(snr for snr in range(-30, 1) if snr not in PRIORITY_SNRS)
PARENTS = 512
CHUNK = 16
MODULE = "tools.original_replication_detection"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cell_name(snr, jsr):
    return f"snr_{snr}_jsr_{jsr}"


def register():
    from tools.server_gpu_admission import require_server_identity
    require_server_identity()
    selected = read(SELECTION)
    if tuple(selected["selected_seeds"]) != SEEDS or selected["epoch"] != 256:
        raise RuntimeError("frozen five-seed selection changed")
    source_names = ["tools/original_replication_detection.py", "tools/server_gpu_admission.py",
                    "tools/paper2_baseline_suite/equal_pfa_suite.py",
                    "tools/paper2_baseline_suite/wp5b_suite.py",
                    "paper2/evaluation_1103_v4.py", "paper2/radar_eval.py"]
    master = read(MASTER / "DATA_MANIFEST.json")
    record = {
        "version": 1, "primary_model": "ReCoVP-Hybrid", "epoch": 256,
        "selected_seeds": list(SEEDS), "priority_snr_db": list(PRIORITY_SNRS),
        "snr_execution_order_db": list(SNRS), "jsr_db": list(JSRS),
        "parents": PARENTS, "jammer_types": ["ISDRJ", "ISRRJ", "ISCRJ"],
        "source_master_sha256": master["component_sha256"],
        "source_master_manifest_sha256": stable_sha256(MASTER / "DATA_MANIFEST.json"),
        "selection_sha256": stable_sha256(SELECTION),
        "transform": {"s": "S0", "j": "J0 * float32(10**(JSR/20))",
                      "epsilon": "E0 * float32(10**(-SNR/20))"},
        "threshold_policy": "nominal cell Pfa 1e-6 and frozen independent 1%, 5%, 10% frame calibration",
        "source_sha256": {name: stable_sha256(ROOT / name) for name in source_names},
        "physical_parent_resampling": "shared parent IDs across all conditions and mechanisms",
        "selection_role": "development characterization under the registered master plan",
        "seed109_preflight_limitation": "original bytes missing; documented recovery bridge retained",
    }
    path = DATA / "REGISTRATION.json"
    if path.exists():
        if read(path) != record:
            raise RuntimeError("R5 registration identity changed")
    else:
        atomic_json(path, record)
    return record


def validate_chunk(path, parent_ids, binding_sha):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        if str(z["binding_sha256"]) != binding_sha or not np.array_equal(z["parent_id"], parent_ids):
            raise RuntimeError(f"R5 chunk identity mismatch {path}")
        for name in z.files:
            if name not in ("parent_id", "binding_sha256") and not np.isfinite(z[name]).all():
                raise RuntimeError(f"R5 nonfinite metric {path} {name}")


def worker(seed):
    from tools.server_gpu_admission import require_visible_gpu
    admission = require_visible_gpu()
    import numpy as np
    import torch
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite.wp5b_suite import dev_metrics_by_type

    registration = register()
    E.configure_precision()
    torch.set_num_threads(4)
    folder = OUT / f"seed_{seed}"
    row_path = E.OUT / "ReCoVP-Hybrid" / f"seed_{seed}" / "ROW.json"
    row = read(row_path)
    if stable_sha256(E.CAL) != row["calibration_sha256"]:
        raise RuntimeError("R5 calibration data SHA mismatch")
    model, checkpoint = E.load("ReCoVP-Hybrid", seed)
    if row["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("R5 calibration and checkpoint differ")
    ranking = next(item for item in read(SELECTION)["ranking_rows"] if item["binding"]["seed"] == seed)
    if checkpoint["checkpoint_sha256"] != ranking["binding"]["checkpoint_sha256"]:
        raise RuntimeError("R5 checkpoint differs from the selected endpoint")
    component = MASTER / "components/test_0_0.npz"
    if stable_sha256(component) != registration["source_master_sha256"]:
        raise RuntimeError("R5 master component SHA mismatch")
    with np.load(component, allow_pickle=False) as z:
        s, j_base, e_base = (z[name] for name in ("s", "j", "epsilon"))
        parent_ids, doppler = z["parent_id"].copy(), z["doppler_hz"].copy()
    if s.shape != (PARENTS, 3, 8, 1800) or len(np.unique(parent_ids)) != PARENTS:
        raise RuntimeError("R5 master shape or parent identity mismatch")
    with np.load(E.CAL, allow_pickle=False) as calibration:
        if np.intersect1d(parent_ids, calibration["parent_id"]).size:
            raise RuntimeError("R5 parents overlap calibration")
    thresholds = {str(pfa): next(float(c["threshold_ratio"]) for c in row["curve"]
                               if c["target_frame_pfa"] == pfa) for pfa in (.01, .05, .1)}
    binding = {"seed": seed, "epoch": 256, "checkpoint": checkpoint,
               "registration_sha256": stable_sha256(DATA / "REGISTRATION.json"),
               "calibration_row_sha256": stable_sha256(row_path), "thresholds": thresholds}
    binding_path = folder / "BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise RuntimeError("R5 previous run binding differs")
    atomic_json(binding_path, binding)
    binding_sha = stable_sha256(binding_path)
    progress = {"state": "RUNNING", "pid": os.getpid(), "seed": seed,
                "gpu_admission": admission, "started_utc": now(),
                "total_cells": len(SNRS) * len(JSRS), "completed_cells": 0,
                "completed_parent_chunks": 0, "last_update_utc": now()}
    atomic_json(folder / "WORKER_STATUS.json", progress)
    targets = target_peak_range(s[:, 0])
    first_inference = True
    for snr in SNRS:
        e = e_base * np.float32(10 ** (-snr / 20))
        for jsr in JSRS:
            cell = folder / cell_name(snr, jsr)
            j = j_base * np.float32(10 ** (jsr / 20))
            for lo in range(0, PARENTS, CHUNK):
                hi = min(lo + CHUNK, PARENTS)
                path = cell / f"parents_{lo:04d}_{hi:04d}.npz"
                if path.exists():
                    validate_chunk(path, parent_ids[lo:hi], binding_sha)
                else:
                    st = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
                    restored = E.restore(model, "hybrid", st)["main"]
                    if first_inference:
                        torch.cuda.synchronize()
                        atomic_json(folder / "CUDA_READY.json", {
                            "pid": os.getpid(), "at_utc": now(), "seed": seed,
                            "physical_gpu_index": admission["physical_gpu_index"],
                            "gpu_uuid": admission["gpu_uuid"], "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
                        first_inference = False
                    values = dev_metrics_by_type(s[lo:hi], j[lo:hi], e[lo:hi], restored, doppler[lo:hi])
                    scores = E.frame_scores(restored, s[lo:hi], e[lo:hi], doppler[lo:hi], targets[lo:hi], True)
                    values.update({f"frame_{key}": value for key, value in scores.items()})
                    for pfa, threshold in thresholds.items():
                        values[f"pd_eq_{pfa}"] = (scores["target"] > threshold).astype(np.float64)
                        values[f"fa_eq_{pfa}"] = (scores["h0"] > threshold).astype(np.float64)
                    values.update({"E_U": values["u_nmse"], "D_S": values["damage"],
                                   "e_S": values["damage"], "E_0": values["no_jammer_pooled_nmse"],
                                   "Hits": values["pd"].astype(np.int64),
                                   "FA_cells": np.rint(values["cell_pfa_jammer"] * 9032).astype(np.int64)})
                    if not all(value.shape == (hi - lo, 3) and np.isfinite(value).all()
                               for value in values.values()):
                        raise RuntimeError("R5 nonfinite values")
                    atomic_write(path, lambda stream: np.savez_compressed(
                        stream, parent_id=parent_ids[lo:hi], binding_sha256=binding_sha, **values))
                progress.update({"last_update_utc": now(), "snr_db": snr, "jsr_db": jsr,
                                 "last_parent_stop": hi,
                                 "completed_parent_chunks": progress["completed_parent_chunks"] + 1})
                atomic_json(folder / "WORKER_STATUS.json", progress)
            chunks = []
            for lo in range(0, PARENTS, CHUNK):
                hi = min(lo + CHUNK, PARENTS)
                with np.load(cell / f"parents_{lo:04d}_{hi:04d}.npz", allow_pickle=False) as z:
                    chunks.append({key: z[key].copy() for key in z.files
                                   if key not in ("parent_id", "binding_sha256")})
            merged = {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}
            atomic_json(cell / "DONE.json", {"status": "COMPLETED_VERIFIED", "finished_utc": now(),
                         "binding_sha256": binding_sha, "parents": PARENTS, "snr_db": snr, "jsr_db": jsr,
                         "jammer_types": registration["jammer_types"],
                         "mean_by_mechanism": {key: value.mean(axis=0).tolist() for key, value in merged.items()},
                         "hits_by_mechanism": merged["Hits"].sum(axis=0).tolist(),
                         "false_cells_by_mechanism": merged["FA_cells"].sum(axis=0).tolist()})
            progress["completed_cells"] += 1
            atomic_json(folder / "WORKER_STATUS.json", progress)
            print(seed, cell.name, progress["completed_cells"], "of", progress["total_cells"], flush=True)
    progress.update({"state": "COMPLETED_VERIFIED", "finished_utc": now()})
    atomic_json(folder / "WORKER_STATUS.json", progress)
    atomic_json(folder / "DONE.json", {**progress, "binding_sha256": binding_sha})


def active_processes(seed):
    import psutil
    matches = []
    for proc in psutil.process_iter(["pid", "create_time", "cmdline"]):
        command = proc.info["cmdline"] or []
        if MODULE not in command or "worker" not in command or "--seed" not in command:
            continue
        index = command.index("--seed")
        if index + 1 < len(command) and command[index + 1] == str(seed):
            matches.append({"pid": proc.pid, "create_time": proc.info["create_time"]})
    return matches


def queue():
    import psutil
    import subprocess
    from tools.server_gpu_admission import NoGpuAvailable, query_resources, select_gpu
    register()
    queue_path = OUT / "QUEUE_STATUS.json"
    state = read(queue_path) if queue_path.exists() else {"created_utc": now(), "tasks": {}}
    state["controller_pid"] = os.getpid()
    for seed in SEEDS:
        state["tasks"].setdefault(str(seed), {"state": "PENDING", "attempts": 0})
    children = {}
    while True:
        for seed in SEEDS:
            task = state["tasks"][str(seed)]
            folder = OUT / f"seed_{seed}"
            if (folder / "DONE.json").exists():
                done = read(folder / "DONE.json")
                if done.get("completed_cells") != len(SNRS) * len(JSRS):
                    raise RuntimeError(f"R5 incomplete seed receipt {seed}")
                task.update({"state": "COMPLETED_VERIFIED", "completed_cells": done["completed_cells"]})
                continue
            active = active_processes(seed)
            if active:
                task.update({"state": "RUNNING", "processes": active})
                continue
            if task.get("state") == "FAILED" and task.get("attempts", 0) >= 3:
                continue
            if time.time() < task.get("retry_after_unix", 0):
                continue
            try:
                resources = query_resources()
                gpu = select_gpu(resources["gpus"], resources["memory"])
            except NoGpuAvailable:
                task["state"] = "WAITING_RESOURCES"
                continue
            previous = task.get("gpu")
            if previous:
                candidate = next(row for row in resources["gpus"] if row["gpu_uuid"] == previous["gpu_uuid"])
                if candidate["utilization_percent"] >= 85:
                    task["state"] = "WAITING_FROZEN_GPU"
                    continue
                gpu = {**candidate, "host_memory": resources["memory"]}
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["physical_gpu_index"]),
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            log_path = OUT / f"seed_{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "worker", "--seed", str(seed)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children[seed] = proc
            task.update({"state": "STARTING", "pid": proc.pid,
                         "create_time": psutil.Process(proc.pid).create_time(), "gpu": gpu,
                         "dispatch_resources": resources, "started_utc": now(),
                         "attempts": task.get("attempts", 0) + 1})
            atomic_json(queue_path, state)
            # Wait for CUDA initialization before observing resources for the
            # following dispatch. The wait does not limit concurrent workers.
            started = time.monotonic()
            while proc.poll() is None and time.monotonic() - started < 60:
                ready = folder / "CUDA_READY.json"
                if ready.exists() and read(ready).get("pid") in [item["pid"] for item in active_processes(seed)]:
                    break
                time.sleep(2)
            break
        for seed, proc in list(children.items()):
            code = proc.poll()
            if code is None:
                continue
            children.pop(seed)
            task = state["tasks"][str(seed)]
            task["exit_code"] = code
            if code == 75:
                task.update({"state": "WAITING_RESOURCES", "attempts": task["attempts"] - 1,
                             "retry_after_unix": time.time() + 30})
            elif code:
                task.update({"state": "FAILED", "retry_after_unix": time.time() + 60})
        state["heartbeat_utc"] = now()
        state["state"] = "RUNNING"
        atomic_json(queue_path, state)
        if all(task["state"] == "COMPLETED_VERIFIED" for task in state["tasks"].values()):
            atomic_json(OUT / "HYBRID_COMPLETE.json", {**state, "finished_utc": now()})
            return
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "queue", "worker"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "queue":
        queue()
    else:
        from tools.server_gpu_admission import NoGpuAvailable
        try:
            worker(args.seed)
        except NoGpuAvailable as exc:
            print(str(exc), flush=True)
            sys.exit(75)
        except Exception:
            atomic_json(OUT / f"seed_{args.seed}" / "LAST_ERROR.json",
                        {"at_utc": now(), "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
