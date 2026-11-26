"""E5: controlled SNR x JSR sweep on newly registered data (evaluation only).

Grid: SNR {-25,-20,-15,-10,-5,0,5,10} dB x JSR {0,10,20,30} dB, 64 physical
parents per cell (three jammer variants each), cell k generated with seed
2026112700+k by the frozen ``paper2.data_1103.generate`` in fixed-condition mode.
A registration file is written before generation; parent identities are checked
against TRAIN, DEV, the confirmation set and both equal-Pfa sets. The sweep is
never used for training, threshold or model selection.

Per (method, seed, cell) the per-parent DEV-style metrics (nominal CFAR) are
saved; ``summarize`` averages seeds and writes light JSON/Markdown.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "paper2_baseline_suite_20261126" / "E5"
DATA = ROOT / "data" / "paper2_baseline_suite_20261126" / "sweep"
SNRS = (-25, -20, -15, -10, -5, 0, 5, 10)
JSRS = (0, 10, 20, 30)
PARENTS = 64
SEED_BASE = 2026112700
CHUNK = 16
PROTECTED = [ROOT / "data/paper2_1103/train.npz", ROOT / "data/paper2_1103/dev.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/confirmation/test.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/equal_pfa/calibration/test.npz",
             ROOT / "data/unet_ablation_3seed_256_20261115/equal_pfa/evaluation/test.npz"]
METRICS = ("u_nmse", "damage", "no_jammer_pooled_nmse", "sjrif_db", "pd", "frame_pfa_jammer",
           "frame_pfa_noise", "amplitude")
CPU_METHODS = ("Max-TF", "GMM-OMP")
HYBRID_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)


def cells():
    return [(k, snr, jsr) for k, (snr, jsr) in enumerate((a, b) for a in SNRS for b in JSRS)]


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


def generate() -> None:
    manifest_path = DATA / "DATA_MANIFEST.json"
    if manifest_path.is_file():
        print("sweep data already generated", flush=True)
        return
    from paper2.data_1103 import generate as gen
    if "out_dir" not in inspect.signature(gen).parameters:
        raise RuntimeError("generator lacks out_dir; refusing to write into the shared data folder")
    reg = OUT / "REGISTRATION.json"
    if not reg.is_file():
        write_json(reg, {"registered_utc": now(), "purpose": "E5 controlled SNR x JSR sweep; evaluation only",
                         "never_used_for": ["training", "threshold calibration", "model/epoch/seed selection"],
                         "generator": "paper2.data_1103.generate fixed-condition mode, role 'test'",
                         "snr_db": SNRS, "jsr_db": JSRS, "parents_per_cell": PARENTS,
                         "seed_rule": f"cell k (row-major over SNR then JSR) uses seed {SEED_BASE}+k",
                         "data_dir": str(DATA)})
    entries, all_ids = {}, []
    for k, snr, jsr in cells():
        result = gen("test", count=PARENTS, seed=SEED_BASE + k, snr_db=float(snr), jsr_db=float(jsr),
                     out_dir=DATA / f"cell_{k:02d}")
        path = Path(result[0] if isinstance(result, (tuple, list)) else result)
        with np.load(path, allow_pickle=False) as z:
            ids = z["parent_id"].copy()
        all_ids.append(ids)
        entries[f"{k:02d}"] = {"snr_db": snr, "jsr_db": jsr, "seed": SEED_BASE + k, "path": str(path),
                               "sha256": sha(path), "parents": int(len(ids))}
        print("generated cell", k, snr, jsr, flush=True)
    ids = np.concatenate(all_ids)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("duplicate parent identities inside the sweep")
    overlaps = {}
    for p in PROTECTED:
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                overlaps[str(p)] = int(np.intersect1d(ids, z["parent_id"]).size)
    if any(overlaps.values()):
        raise RuntimeError(f"sweep overlaps protected sets: {overlaps}")
    write_json(manifest_path, {"created_utc": now(), "registration_sha256": sha(reg), "cells": entries,
                               "overlap_counts": overlaps, "total_parents": int(len(ids))})


def jobs():
    hyb = [s for s in HYBRID_SEEDS
           if (ROOT / f"outputs/recovp_hybrid_seed{s}/seed_{s}/TRAINING_COMPLETE.json").is_file()]
    return ([("ReCoVP-Hybrid", s) for s in hyb] + [("CorrectionNet", s) for s in (17, 29, 41)]
            + [("UNet", s) for s in (17, 29, 41)]
            + [("CVEDNet", None), ("RSDR-Net", None), ("Max-TF", None), ("GMM-OMP", None)])


def make_restorer(method, seed):
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    if method in E.FAMILIES:
        model, binding = E.load(method, seed)
        key = E.FAMILIES[method][1]
        return (lambda st, j: E.restore(model, key, st)["main"]), binding
    if method == "CVEDNet":
        from tools.paper2_baseline_suite import external_suite as X
        model, ca, meta = X.load_model()
        return (lambda st, j: X.restore(model, ca, st)), meta
    if method == "RSDR-Net":
        from tools.paper2_baseline_suite import rsdr_suite as R
        net, meta = R.load()
        return (lambda st, j: R.restore(net, st)), meta
    from tools.paper2_baseline_suite import classical_suite as C
    name = {"Max-TF": "max_tf_isrj1", "GMM-OMP": "gmm_omp"}[method]
    return (lambda st, j: C.restore(name, st, j)), {"config": name}


def folder_for(method, seed) -> Path:
    return OUT / "results" / method / (f"seed_{seed}" if seed is not None else "single")


def evaluate(method: str, seed) -> None:
    folder = folder_for(method, seed)
    if (folder / "DONE.json").is_file():
        return
    admission = None
    if method not in CPU_METHODS:
        from tools.paper2_baseline_suite.equal_pfa_suite import configure_precision, require_server
        admission = require_server()
        configure_precision()
    import torch
    torch.set_num_threads(4)
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    restore, meta = make_restorer(method, seed)
    from paper2.evaluation_1103_v4 import states_from_components
    from tools.paper2_baseline_suite.classical_suite import dev_metrics
    t0 = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    for k, cell in manifest["cells"].items():
        out = folder / f"cell_{k}.npz"
        if out.is_file():
            continue
        path = Path(cell["path"])
        if sha(path) != cell["sha256"]:
            raise RuntimeError(f"sweep cell identity changed: {path}")
        with np.load(path, allow_pickle=False) as z:
            s, j, e = (z[x] for x in ("s", "j", "epsilon"))
            pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
        parts = []
        for lo in range(0, len(pid), CHUNK):
            hi = min(lo + CHUNK, len(pid))
            states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
            parts.append(dev_metrics(s[lo:hi], j[lo:hi], e[lo:hi], restore(states, j[lo:hi]), fd[lo:hi]))
        merged = {m: np.concatenate([p[m] for p in parts]) for m in parts[0]}
        tmp = folder / f"cell_{k}.tmp.npz"
        np.savez(tmp, parent_id=pid, **merged)
        os.replace(tmp, out)
        print(method, seed, "cell", k, round(time.perf_counter() - t0, 1), flush=True)
    write_json(folder / "DONE.json", {"method": method, "seed": seed, "finished_utc": now(), "model": meta,
                                      "gpu_admission": admission, "seconds": time.perf_counter() - t0})


def summarize() -> None:
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    table = {}
    for mdir in sorted((OUT / "results").glob("*")):
        seeds = [d for d in sorted(mdir.glob("*")) if (d / "DONE.json").is_file()]
        if not seeds:
            continue
        entry = {"seeds": [d.name for d in seeds], "cells": {}}
        for k, cell in manifest["cells"].items():
            vals = {m: [] for m in METRICS}
            for d in seeds:
                with np.load(d / f"cell_{k}.npz") as z:
                    for m in METRICS:
                        vals[m].append(float(np.mean(z[m])))
            entry["cells"][k] = {"snr_db": cell["snr_db"], "jsr_db": cell["jsr_db"],
                                 **{m: float(np.mean(v)) for m, v in vals.items()},
                                 **{f"{m}_seed_sd": (float(np.std(v, ddof=1)) if len(v) > 1 else None)
                                    for m, v in vals.items()}}
        table[mdir.name] = entry
    write_json(OUT / "SWEEP_SUMMARY.json", {"created_utc": now(), "parents_per_cell": PARENTS,
                                            "snr_db": SNRS, "jsr_db": JSRS, "methods": table})
    lines = ["# E5 受控扫描（每格64个场景，名义CFAR）", ""]
    for metric, pct in (("pd", True), ("frame_pfa_jammer", True), ("damage", False), ("sjrif_db", False)):
        lines += [f"## {metric}", ""]
        for name, entry in table.items():
            lines += [f"### {name}（{', '.join(entry['seeds'])}）", "",
                      "| SNR \\ JSR | " + " | ".join(f"{j} dB" for j in JSRS) + " |",
                      "|---|" + "---:|" * len(JSRS)]
            by = {(c["snr_db"], c["jsr_db"]): c[metric] for c in entry["cells"].values()}
            for snr in SNRS:
                row = [by.get((snr, jsr)) for jsr in JSRS]
                cellfmt = [("-" if v is None else (f"{100 * v:.1f}" if pct else f"{v:.3g}")) for v in row]
                lines.append(f"| {snr} dB | " + " | ".join(cellfmt) + " |")
            lines.append("")
    (OUT / "SWEEP_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary written", len(table), flush=True)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "eval", "summarize", "jobs"))
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    a = parser.parse_args()
    if a.action == "generate":
        generate()
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
