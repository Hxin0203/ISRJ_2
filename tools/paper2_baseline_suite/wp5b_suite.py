"""WP5b: jammer-structure sweep (slice width and unaffected fraction); evaluation only.

Registered before data generation (see REGISTRATION.json written by ``register``).

Grid (repeat type is primary; direct/cyclic use the same timing parameters):
  width  w in {6,18,36,54,72} samples, M=2 repeats, period=(M+1)*w+1  (paper-2 sliceB rule,
         identical to paper-1 RSDR_WIDTH_DENSE_M2_20261114 stage B)
  eta    w=36, M=2, period in {145,217}; period 109 is the width cell w=36 (paper-1
         RSDR_ETA_M2_20261114 periods 109/145/217)
Fixed SNR=-10 dB, JSR=20 dB (active-support JSR, same definition as E5/TRAIN), forward
delay 1, replay gap 0, sampling phase 0; channel phase and frequency offset as drawn.

Data: the frozen ``paper2.data_1103.generate`` in fixed-condition mode with one shared seed
for every cell, so the same physical parents (target, noise, jammed-pulse pattern, nuisance
phases) appear in every cell and only the jammer timing differs. The timing override wraps
the frozen ``_draw_jammer``: it is called first (identical RNG consumption), then the
jammer is re-rendered with the fixed timing and rescaled to the drawn active-support JSR.
No file under paper2/ is modified. Parent ids are checked against TRAIN, DEV, the
confirmation set, both equal-Pfa sets and the E5 sweep.

Metrics per (method, seed, cell): DEV-style per-parent metrics (nominal CFAR) and, with the
method's own equal-Pfa thresholds already fixed on the E3/E6/E7 calibration set (target
frame Pfa 1% and 5%; no recalibration), Pd and jammer-frame Pfa.
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
SUITE = ROOT / "outputs" / "paper2_baseline_suite_20261126"
OUT = SUITE / "WP5b"
DATA = ROOT / "data" / "paper2_baseline_suite_20261126" / "wp5b"
SEED = 2026112800
PARENTS = 64
SNR_DB, JSR_DB = -10.0, 20.0
M = 2
CELLS = ([dict(name=f"w{w:02d}", width=w, period=(M + 1) * w + 1, scan="width") for w in (6, 18, 36, 54, 72)]
         + [dict(name=f"eta_p{p}", width=36, period=p, scan="eta") for p in (145, 217)])
TRAIN_RANGE = dict(window=(6, 72), period=(36, 260), repeats=(1, 10))
PROTECTED = [ROOT / "data/paper2_1103/train.npz", ROOT / "data/paper2_1103/dev.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/confirmation/test.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/equal_pfa/calibration/test.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/equal_pfa/evaluation/test.npz"]
PFAS = (0.01, 0.05)
CHUNK = 16
HYBRID_SEEDS = (17, 29, 41, 53, 67)
CONV_SEEDS = (17, 29, 41)
CPU_METHODS = ("Max-TF", "GMM-OMP")
EXTERNAL_ROWS = {"RSDR-Net": "E7/equal_pfa/rsdr_net", "CVEDNet": "E7/equal_pfa/cvednet",
                 "Max-TF": "E6/equal_pfa/max_tf_isrj1", "GMM-OMP": "E6/equal_pfa/gmm_omp"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def register() -> None:
    reg = OUT / "REGISTRATION.json"
    if reg.is_file():
        print("already registered", sha(reg), flush=True)
        return
    def flag(c):
        out = []
        if not TRAIN_RANGE["window"][0] <= c["width"] <= TRAIN_RANGE["window"][1]:
            out.append("window outside training range")
        if not TRAIN_RANGE["period"][0] <= c["period"] <= TRAIN_RANGE["period"][1]:
            out.append("period outside training range")
        return out
    write_json(reg, {"registered_utc": now(), "purpose": "WP5b jammer-structure sweep; evaluation only",
                     "never_used_for": ["training", "threshold calibration", "model/epoch/seed selection"],
                     "script_sha256": sha(Path(__file__)), "seed": SEED, "parents_per_cell": PARENTS,
                     "snr_db": SNR_DB, "jsr_db": JSR_DB, "jsr_definition": "active-support power per pulse (E5/TRAIN)",
                     "repeat_count": M, "cells": [dict(c, extrapolation=flag(c)) for c in CELLS],
                     "eta_note": "period 109 at w=36 is shared with width cell w36",
                     "primary_jammer_type": "repeat", "training_range": TRAIN_RANGE,
                     "thresholds": "method's own E3/E6/E7 calibration curve at target frame Pfa 1% and 5%; not refit",
                     "paper1_sources": ["D:/Project Code/ISRJ_publication_width_20261113_v4/evidence/outputs/width_dense_m2_20261114/code/PROTOCOL_LOCK.json",
                                         "D:/Project Code/ISRJ_publication_width_20261113_v4/evidence/outputs/eta_m2_20261114/code/PROTOCOL_LOCK.json"],
                     "methods": [m for m, _ in jobs()]})
    print("registered", sha(reg), flush=True)


def _install_override(width: int, period: int):
    """Wrap the frozen jammer draw: same RNG use, fixed timing, same drawn JSR."""
    import paper2.data_1103 as D
    from paper2.vendor import generate_v2 as G
    original = G._draw_jammer

    def fixed(sp, active, signal_power, jam_type, cfg, rng):
        _, meta = original(sp, active, signal_power, jam_type, cfg, rng)
        params = dict(window_samples=width, period_samples=period, sampling_phase_samples=0,
                      forward_delay_samples=1, repeat_count=M, replay_gap_samples=0)
        raw = G._raw_jammer(sp, active, jam_type, params)
        t = np.arange(cfg.signal_length, dtype=np.float64) / cfg.sample_rate_hz
        raw = raw * np.exp(1j * (meta["channel_phase_rad"] + 2 * np.pi * meta["frequency_offset_hz"] * t))
        support = np.abs(raw) > np.finfo(np.float64).eps
        if not support.any():
            raise RuntimeError("empty jammer support")
        scale = np.sqrt(signal_power * 10 ** (meta["jsr_db"] / 10) / np.mean(np.abs(raw[support]) ** 2))
        j = raw * scale
        actual = 10 * np.log10(np.mean(np.abs(j[support]) ** 2) / signal_power)
        return j, dict(meta, **params, actual_jsr_db=float(actual))
    D._draw_jammer = fixed
    return lambda: setattr(D, "_draw_jammer", original)


def generate() -> None:
    manifest = DATA / "DATA_MANIFEST.json"
    if manifest.is_file():
        print("data already generated", flush=True)
        return
    reg = OUT / "REGISTRATION.json"
    if not reg.is_file():
        raise RuntimeError("register before generating")
    from paper2.data_1103 import generate as gen
    entries, ref_ids = {}, None
    for c in CELLS:
        restore = _install_override(c["width"], c["period"])
        try:
            path, _ = gen("test", count=PARENTS, seed=SEED, snr_db=SNR_DB, jsr_db=JSR_DB, out_dir=DATA / c["name"])
        finally:
            restore()
        with np.load(path, allow_pickle=False) as z:
            ids, jsr = z["parent_id"].copy(), z["jsr_db"].copy()
        if ref_ids is None:
            ref_ids = ids
        elif not np.array_equal(ids, ref_ids):
            raise RuntimeError("cells are not parent-paired")
        entries[c["name"]] = dict(c, path=str(path), sha256=sha(path), parents=int(len(ids)),
                                  actual_jsr_db_mean=float(np.nanmean(jsr)))
        print("generated", c["name"], flush=True)
    overlaps = {}
    e5 = ROOT / "data/paper2_baseline_suite_20261126/sweep"
    for p in PROTECTED + sorted(e5.glob("cell_*/*.npz")):
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                overlaps[str(p)] = int(np.intersect1d(ref_ids, z["parent_id"]).size)
    if any(overlaps.values()):
        raise RuntimeError(f"WP5b overlaps protected sets: {overlaps}")
    write_json(manifest, {"created_utc": now(), "registration_sha256": sha(reg), "cells": entries,
                          "overlap_counts": overlaps, "paired_parents": int(len(ref_ids))})
def jobs():
    out = [("ReCoVP-Hybrid", s) for s in HYBRID_SEEDS]
    from tools.paper2_baseline_suite.equal_pfa_suite import FAMILIES
    out += [(f, s) for f in FAMILIES if f != "ReCoVP-Hybrid" for s in CONV_SEEDS]
    return out + [("RSDR-Net", None), ("CVEDNet", None), ("Max-TF", None), ("GMM-OMP", None)]


def row_folder(label: str, seed) -> Path:
    if label in EXTERNAL_ROWS:
        return SUITE / EXTERNAL_ROWS[label]
    return SUITE / "E3" / label / f"seed_{seed}"


def thresholds(label: str, seed) -> dict:
    row = json.loads((row_folder(label, seed) / "ROW.json").read_text(encoding="utf-8"))
    out = {}
    for pfa in PFAS:
        hit = [r for r in row["curve"] if abs(r["target_frame_pfa"] - pfa) < 1e-12]
        if len(hit) != 1:
            raise RuntimeError(f"no unique {pfa} threshold for {label} {seed}")
        out[str(pfa)] = float(hit[0]["threshold_ratio"])
    return {"ratios": out, "row_sha256": sha(row_folder(label, seed) / "ROW.json")}


def make_restorer(method: str, seed):
    """-> (fn(states, j) -> {label: u}, binding)."""
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    if method in E.FAMILIES:
        model, binding = E.load(method, seed)
        key = E.FAMILIES[method][1]
        if key == "hybrid":
            def both(st, j):
                r = E.restore(model, key, st)
                return {"ReCoVP-Hybrid": r["main"], "ReCoVP-Stage1": r["stage1"]}
            return both, binding
        return (lambda st, j: {method: E.restore(model, key, st)["main"]}), binding
    if method == "CVEDNet":
        from tools.paper2_baseline_suite import external_suite as X
        model, ca, meta = X.load_model()
        return (lambda st, j: {"CVEDNet": X.restore(model, ca, st)}), meta
    if method == "RSDR-Net":
        from tools.paper2_baseline_suite import rsdr_suite as R
        net, meta = R.load()
        return (lambda st, j: {"RSDR-Net": R.restore(net, st)}), meta
    from tools.paper2_baseline_suite import classical_suite as C
    name = {"Max-TF": "max_tf_isrj1", "GMM-OMP": "gmm_omp"}[method]
    return (lambda st, j: {method: C.restore(name, st, j)}), {"config": name}


def dev_metrics_by_type(s, j, e, u, doppler):
    """classical_suite.dev_metrics without the jammer-type mean: arrays are [n, 3]."""
    from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, states_from_components
    n = len(s)
    s4, j4, e4 = (x.reshape(n * 3, 8, 1800).astype(np.complex128) for x in (s, j, e))
    uu = np.moveaxis(u, 2, 1).reshape(n * 3, 4, 8, 1800).astype(np.complex128)
    m = group_metrics(s4, j4, e4, uu)
    m.update(radar_metrics(s4, states_from_components(s4, j4, e4), uu, np.repeat(doppler, 3)))
    return {k: np.asarray(v).reshape(n, 3) for k, v in m.items()}


def evaluate(method: str, seed) -> None:
    tag = f"seed_{seed}" if seed is not None else "single"
    done = OUT / "results" / method / tag / "DONE.json"
    if done.is_file():
        return
    admission = None
    import torch
    if method not in CPU_METHODS:
        from tools.paper2_baseline_suite.equal_pfa_suite import configure_precision, require_server
        admission = require_server()
        configure_precision()
    torch.set_num_threads(4)
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite.equal_pfa_suite import frame_scores
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    restore, binding = make_restorer(method, seed)
    t0 = time.perf_counter()
    thr = None
    for name, cell in manifest["cells"].items():
        path = Path(cell["path"])
        if sha(path) != cell["sha256"]:
            raise RuntimeError(f"cell identity changed: {path}")
        with np.load(path, allow_pickle=False) as z:
            s, j, e = (z[x] for x in ("s", "j", "epsilon"))
            pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
        tidx = target_peak_range(s[:, 0])
        parts = {}
        for lo in range(0, len(pid), CHUNK):
            hi = min(lo + CHUNK, len(pid))
            st = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
            for label, u in restore(st, j[lo:hi]).items():
                m = dev_metrics_by_type(s[lo:hi], j[lo:hi], e[lo:hi], u, fd[lo:hi])
                f = frame_scores(u, s[lo:hi], e[lo:hi], fd[lo:hi], tidx[lo:hi], True)
                parts.setdefault(label, []).append({**m, **{f"frame_{k}": v for k, v in f.items()}})
        if thr is None:
            thr = {label: thresholds(label, seed) for label in parts}
        for label, plist in parts.items():
            folder = OUT / "results" / label / tag
            folder.mkdir(parents=True, exist_ok=True)
            merged = {k: np.concatenate([p[k] for p in plist]) for k in plist[0]}
            tmp = folder / f"cell_{name}.tmp.npz"
            np.savez(tmp, parent_id=pid, **merged)
            os.replace(tmp, folder / f"cell_{name}.npz")
        print(method, seed, name, round(time.perf_counter() - t0, 1), flush=True)
    for label in thr:
        write_json(OUT / "results" / label / tag / "DONE.json",
                   {"method": method, "label": label, "seed": seed, "finished_utc": now(), "model": binding,
                    "thresholds": thr[label], "gpu_admission": admission, "seconds": time.perf_counter() - t0,
                    "script_sha256": sha(Path(__file__))})
def _pick_gpu(preferred: int) -> dict:
    from tools.server_gpu_admission import UTILIZATION_LIMIT, query_gpus, select_gpu
    samples = query_gpus()
    pref = [r for r in samples if r["physical_gpu_index"] == preferred and r["utilization_percent"] < UTILIZATION_LIMIT]
    return pref[0] if pref else select_gpu(samples)


def queue(lane: str) -> None:
    """Sequential lane. gpuA/gpuB split GPU jobs by parity (prefer GPU 0/1); cpu runs classical methods."""
    from tools.server_gpu_admission import NoGpuAvailable
    if not (DATA / "DATA_MANIFEST.json").is_file():
        raise RuntimeError("generate data first")
    gpu_jobs = [j for j in jobs() if j[0] not in CPU_METHODS]
    todo = {"gpuA": gpu_jobs[0::2], "gpuB": gpu_jobs[1::2], "cpu": [j for j in jobs() if j[0] in CPU_METHODS]}[lane]
    status = OUT / f"QUEUE_{lane}.json"
    log = {"lane": lane, "started_utc": now(), "done": [], "failed": []}
    for method, seed in todo:
        tag = f"seed_{seed}" if seed is not None else "single"
        if (OUT / "results" / method / tag / "DONE.json").is_file():
            log["done"].append([method, seed]); continue
        attempts = 0
        while True:
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            gpu = None
            if lane != "cpu":
                try:
                    gpu = _pick_gpu(0 if lane == "gpuA" else 1)
                except NoGpuAvailable:
                    time.sleep(60); continue
                env["CUDA_VISIBLE_DEVICES"] = str(gpu["physical_gpu_index"])
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            args = [sys.executable, "-m", "tools.paper2_baseline_suite.wp5b_suite", "eval", "--method", method]
            if seed is not None:
                args += ["--seed", str(seed)]
            code = subprocess.call(args, cwd=str(ROOT), env=env)
            if code == 3:
                time.sleep(60); continue
            attempts += 1
            if code == 0 or attempts >= 2:
                (log["done"] if code == 0 else log["failed"]).append([method, seed, code, gpu and gpu["physical_gpu_index"]])
                break
        log["updated_utc"] = now()
        write_json(status, log)
    log["finished_utc"] = now()
    write_json(status, log)


CORE = ("u_nmse", "damage", "no_jammer_pooled_nmse", "amplitude", "sjrif_db", "pd", "frame_pfa_jammer")


def _cell_values(npz: Path, thr: dict) -> dict:
    with np.load(npz) as z:
        d = {k: z[k] for k in z.files}
    out = {}
    for scope, sl in (("all", slice(None)), ("repeat", slice(1, 2))):
        for k in CORE:
            if k in d:
                out[f"{scope}:{k}"] = d[k][:, sl].mean(axis=1)
        for pfa, t in thr["ratios"].items():
            out[f"{scope}:pd_eq{pfa}"] = (d["frame_target"][:, sl] > t).mean(axis=1)
            out[f"{scope}:pfa_eq{pfa}"] = (d["frame_h0"][:, sl] > t).mean(axis=1)
        out[f"{scope}:resid_peak"] = d["frame_resid_peak_noise_units"][:, sl].mean(axis=1)
    return out


def summarize() -> None:
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    cells = list(manifest["cells"])
    per = {}
    for ldir in sorted((OUT / "results").glob("*")):
        seeds = [d for d in sorted(ldir.glob("*")) if (d / "DONE.json").is_file()]
        if not seeds:
            continue
        per[ldir.name] = {c: [] for c in cells}
        for d in seeds:
            thr = json.loads((d / "DONE.json").read_text(encoding="utf-8"))["thresholds"]
            for c in cells:
                per[ldir.name][c].append(_cell_values(d / f"cell_{c}.npz", thr))
    table = {}
    for label, by in per.items():
        table[label] = {"seeds": len(next(iter(by.values()))), "cells": {}}
        for c, runs in by.items():
            keys = runs[0].keys()
            table[label]["cells"][c] = {k: {"mean": float(np.mean([r[k].mean() for r in runs])),
                                            "seed_sd": float(np.std([r[k].mean() for r in runs], ddof=1)) if len(runs) > 1 else None,
                                            "p99": float(np.quantile(np.concatenate([r[k] for r in runs]), 0.99))}
                                        for k in keys}
    paired = {}
    rng = np.random.default_rng(20261127)
    ref = per.get("ReCoVP-Hybrid", {})
    for label, by in per.items():
        if label == "ReCoVP-Hybrid" or not ref:
            continue
        paired[label] = {}
        for c in cells:
            res = {}
            for k in ("repeat:damage", "repeat:u_nmse", "repeat:pd_eq0.01", "repeat:pd_eq0.05", "repeat:frame_pfa_jammer",
                      "all:damage", "all:pd_eq0.01", "all:pd_eq0.05"):
                a = np.mean([r[k] for r in ref[c]], axis=0)
                b = np.mean([r[k] for r in by[c]], axis=0)
                diff = a - b
                boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
                res[k] = {"diff": float(diff.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
            paired[label][c] = res
    eta = {}
    for c, cell in manifest["cells"].items():
        with np.load(cell["path"], allow_pickle=False) as z:
            s, j = z["s"][:, 1], z["j"][:, 1]
        jammed = np.abs(j).max(axis=-1) > 0
        act = np.abs(s) > 0
        sel = jammed[..., None] & act
        eta[c] = float(1 - ((np.abs(j) > 0) & sel).sum() / sel.sum())
    write_json(OUT / "SUMMARY.json", {"created_utc": now(), "cells": manifest["cells"], "actual_eta_repeat": eta,
                                      "table": table, "paired_vs_hybrid": paired,
                                      "notes": "seed-averaged per parent before paired bootstrap (2000 reps, parent resampling); "
                                               "thresholds are each method's own fixed E3/E6/E7 calibration ratios"})
    print("summary", len(table), flush=True)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("register", "generate", "eval", "queue", "summarize", "jobs"))
    p.add_argument("--method")
    p.add_argument("--seed", type=int)
    p.add_argument("--lane", choices=("gpuA", "gpuB", "cpu"))
    a = p.parse_args()
    if a.action == "register":
        register()
    elif a.action == "generate":
        generate()
    elif a.action == "queue":
        queue(a.lane)
    elif a.action == "summarize":
        summarize()
    elif a.action == "jobs":
        print(json.dumps(jobs()))
    else:
        from tools.server_gpu_admission import NoGpuAvailable
        try:
            evaluate(a.method, a.seed)
        except NoGpuAvailable as exc:
            print("gpu busy", exc, flush=True)
            sys.exit(3)


if __name__ == "__main__":
    main()
