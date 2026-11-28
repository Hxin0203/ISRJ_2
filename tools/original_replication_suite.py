"""ISRJ_2 ReCoVP-Hybrid epoch-256 main-condition evaluation.

This runner owns the new paper-2 result version. It reuses the audited
restorers and metric implementation from the baseline suite while materializing
paired SNR/JSR cells from ``hybrid_condition_master_20261127``. The only
learned model family evaluated here is ReCoVP-Hybrid, with the frozen five
seeds selected by the uniform DEV ranking. Max-TF, GMM-OMP and CVEDNet are
included as the registered comparison methods. RSDR-Net is intentionally not
part of this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
MASTER = ROOT / "data" / "hybrid_condition_master_20261127"
OUT = ROOT / "outputs" / "original_paper_replication_256_v1" / "R3_main"
DATA = ROOT / "data" / "original_paper_replication_256_v1" / "R3_main"
REGISTRATION = OUT / "REGISTRATION.json"
MANIFEST = DATA / "DATA_MANIFEST.json"
SEEDS = (67, 97, 53, 41, 109)
SNRS = (-30, -25, -20, -15, -10, -5, 0)
JSRS = (0, 10, 20, 30, 35)
PARENTS = 256
CHUNK = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _master_components() -> tuple[dict[str, np.ndarray], dict]:
    registration = MASTER / "REGISTRATION.json"
    manifest = MASTER / "DATA_MANIFEST.json"
    component = MASTER / "components" / "test_0_0.npz"
    for path in (registration, manifest, component):
        if not path.is_file():
            raise RuntimeError(f"missing registered master asset: {path}")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    expected = meta.get("component_sha256")
    if expected and sha(component) != expected:
        raise RuntimeError("hybrid-condition master component SHA mismatch")
    with np.load(component, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    for key in ("s", "j", "epsilon", "parent_id", "doppler_hz"):
        if key not in arrays:
            raise RuntimeError(f"master component missing {key}")
    if arrays["s"].shape[0] < PARENTS:
        raise RuntimeError(f"master has only {arrays['s'].shape[0]} parents")
    return {key: value[:PARENTS] for key, value in arrays.items()}, meta


def _cell_name(snr: float, jsr: float) -> str:
    return f"snr_{snr:g}_jsr_{jsr:g}"


def prepare() -> None:
    base, master_meta = _master_components()
    if REGISTRATION.is_file() and MANIFEST.is_file():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("registration_sha256") != sha(REGISTRATION):
            raise RuntimeError("R3 registration identity changed")
        print("R3 data already prepared", flush=True)
        return
    write_json(REGISTRATION, {
        "version": "original_paper_replication_256_v1",
        "created_utc": now(),
        "purpose": "ReCoVP-Hybrid epoch-256 main comparison grid",
        "methods": ["ReCoVP-Hybrid", "Max-TF", "GMM-OMP", "CVEDNet"],
        "hybrid_seeds": list(SEEDS), "snr_db": list(SNRS), "jsr_db": list(JSRS),
        "parents_per_cell": PARENTS, "jammer_variants": ["ISDRJ", "ISRRJ", "ISCRJ"],
        "source_master": str(MASTER), "source_master_manifest_sha256": sha(MASTER / "DATA_MANIFEST.json"),
        "source_master_component_sha256": sha(MASTER / "components" / "test_0_0.npz"),
        "transform": {"epsilon": "epsilon_master * 10**(-snr_db/20)",
                       "j": "j_master * 10**(jsr_db/20)"},
        "regime_definition": {"extreme": "-30 <= SNR < -20", "ordinary": "-20 <= SNR <= 0"},
        "threshold_policy": "nominal CFAR for cell metrics; equal-Pfa rows remain separate",
    })
    entries = {}
    for snr in SNRS:
        for jsr in JSRS:
            name = _cell_name(snr, jsr)
            path = DATA / f"{name}.npz"
            scaled = {
                "s": base["s"],
                "j": base["j"] * np.float32(10 ** (jsr / 20)),
                "epsilon": base["epsilon"] * np.float32(10 ** (-snr / 20)),
                "parent_id": base["parent_id"],
                "doppler_hz": base["doppler_hz"],
            }
            tmp = path.with_suffix(".tmp.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(tmp, **scaled)
            os.replace(tmp, path)
            entries[name] = {"snr_db": snr, "jsr_db": jsr, "path": str(path),
                             "sha256": sha(path), "parents": PARENTS}
    write_json(MANIFEST, {
        "created_utc": now(), "registration_sha256": sha(REGISTRATION),
        "master_manifest_sha256": sha(MASTER / "DATA_MANIFEST.json"),
        "master_component_sha256": sha(MASTER / "components" / "test_0_0.npz"),
        "parents_per_cell": PARENTS, "physical_parent_count": PARENTS,
        "cells": entries, "master_meta": master_meta,
        "pairing": "same physical parent IDs across all SNR, JSR and jammer variants",
    })
    print("R3_DATA_READY", len(entries), flush=True)


def _suite():
    from tools.paper2_baseline_suite import sweep_suite as suite

    suite.ROOT = ROOT
    suite.OUT = OUT
    suite.DATA = DATA
    suite.SNRS = SNRS
    suite.JSRS = JSRS
    suite.PARENTS = PARENTS
    suite.CHUNK = CHUNK
    suite.METRICS = ("u_nmse", "damage", "no_jammer_pooled_nmse", "sjrif_db", "pd",
                     "frame_pfa_jammer", "frame_pfa_noise", "amplitude")
    return suite


def jobs() -> list[tuple[str, int | None]]:
    return ([('ReCoVP-Hybrid', seed) for seed in SEEDS]
            + [('CVEDNet', None), ('Max-TF', None), ('GMM-OMP', None)])


def evaluate(method: str, seed: int | None) -> None:
    _suite().evaluate(method, seed)


def queue() -> None:
    """Run registered jobs with independent output directories.

    The persistent outer controller performs resource admission for every
    dispatch. This command is also safe to resume because each worker writes a
    DONE receipt and skips an existing completed directory.
    """
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("R3 evaluation runs only on server B")
    prepare()
    from tools.server_gpu_admission import NoGpuAvailable, choose_gpu, query_resources, select_gpu

    state_path = OUT / "QUEUE_STATUS.json"
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"started_utc": now(), "jobs": [list(job) for job in jobs()], "records": {}}
    if state_path.is_file():
        state.update(json.loads(state_path.read_text(encoding="utf-8")))
    pending = []
    for method, seed in jobs():
        folder = OUT / "results" / method / (f"seed_{seed}" if seed is not None else "single")
        if (folder / "DONE.json").is_file():
            state["records"][f"{method}:{seed}"] = {"state": "DONE"}
            continue
        pending.append((method, seed))
    running: dict[int, tuple[str, int | None, object, object, Path]] = {}
    while pending or running:
        for method, seed in list(pending):
            try:
                resources = query_resources()
                if method in ("Max-TF", "GMM-OMP"):
                    # CPU jobs still respect the host physical-memory gate.
                    select_gpu(resources["gpus"], resources["memory"])
                    gpu = None
                else:
                    gpu = select_gpu(resources["gpus"], resources["memory"])
            except NoGpuAvailable:
                break
            log = OUT / "logs" / f"{method.replace('-', '_')}_{seed or 'single'}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu["physical_gpu_index"])}
            args = [sys.executable, "-m", "tools.original_replication_suite", "eval",
                    "--method", method]
            if seed is not None:
                args += ["--seed", str(seed)]
            handle = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=handle,
                                    stderr=subprocess.STDOUT)
            key = f"{method}:{seed}"
            state["records"][key] = {"state": "RUNNING", "pid": proc.pid,
                                     "gpu": gpu, "log": str(log), "started_utc": now()}
            running[proc.pid] = (method, seed, proc, handle, log)
            pending.remove((method, seed))
            write_json(state_path, state)
        for pid, (method, seed, proc, handle, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            key = f"{method}:{seed}"
            state["records"][key].update({"state": "DONE" if rc == 0 else "FAILED",
                                           "exit_code": rc, "finished_utc": now()})
            running.pop(pid)
            write_json(state_path, state)
        if pending or running:
            time.sleep(20)
    state.update({"state": "COMPLETE", "finished_utc": now()})
    write_json(OUT / "ALL_COMPLETE.json", state)


def summarize() -> None:
    suite = _suite()
    suite.summarize()
    summary = OUT / "SWEEP_SUMMARY.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["runner_sha256"] = sha(Path(__file__))
        payload["selected_hybrid_seeds"] = list(SEEDS)
        payload["mechanism_aggregation"] = "three-mechanism average from the legacy R3 evaluator"
        payload["cvednet_role"] = "historical first-paper-data epoch320 reference"
        payload["per_mechanism_results"] = "R3_by_mechanism/postprocessing/R3_PER_SEED_BY_MECHANISM.csv"
        write_json(summary, payload)
    _summarize_by_mechanism()


def _summarize_by_mechanism() -> None:
    """Reconcile the independently persisted mechanism arrays."""
    from tools.original_replication_mechanism_summary import reconcile_once
    reconcile_once()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "eval", "queue", "summarize", "jobs"))
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "jobs":
        print(json.dumps(jobs()))
    elif args.action == "eval":
        evaluate(args.method, args.seed)
    elif args.action == "queue":
        queue()
    else:
        summarize()


if __name__ == "__main__":
    main()
