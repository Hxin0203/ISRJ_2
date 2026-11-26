"""Aggregate E3/E6/E7 into one comparison (server CPU, read-only on all inputs).

Outputs (light JSON + Markdown, safe for GitHub):
  SUITE_SUMMARY.json  - equal-Pfa table for every method (seed mean, seed SD),
                        paired equal-Pfa differences Hybrid minus method,
                        paired DEV256 differences Hybrid minus classical/CVEDNet.
  SUITE_TABLES.md     - the same numbers as tables.

Paired equal-Pfa: every method keeps its own calibration threshold (1% frame
quantile on its own J+E calibration outputs). Per evaluation parent the three
jammer variants are averaged; multi-seed methods are averaged per parent across
seeds before differencing. Unit of resampling is the physical parent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
SUITE = ROOT / "outputs" / "paper2_baseline_suite_20261126"
REPS = 4000
PFA = 0.01
HYBRID_SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
DEV_METRICS = ("u_nmse", "damage", "no_jammer_pooled_nmse", "background_nmse", "amplitude", "sjrif_db",
               "false_peak_suppression_db", "pd", "frame_pfa_jammer", "frame_pfa_noise")
LOWER = {"u_nmse", "damage", "no_jammer_pooled_nmse", "background_nmse", "frame_pfa_jammer", "frame_pfa_noise"}
# Folder names are authoritative (an early max_tf ROW.json carries the label "Max-TF").
FOLDER_LABELS = {"max_tf": "Max-TF (1103 cfg)", "max_tf_isrj1": "Max-TF", "gmm_omp": "GMM-OMP", "cvednet": "CVEDNet",
                 "rsdr_net": "RSDR-Net"}


def boot(d, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(d), size=(REPS, len(d)))
    lo, hi = np.quantile(d[idx].mean(axis=1), [.025, .975])
    return [float(lo), float(hi)]


def equal_pfa_sources():
    """family -> list of (seed, folder) holding ROW.json + SCORES.npz."""
    out = {}
    for row in sorted((SUITE / "E3").glob("*/seed_*/ROW.json")):
        out.setdefault(row.parent.parent.name, []).append((int(row.parent.name[5:]), row.parent))
    for sub in ("E6", "E7"):
        for row in sorted((SUITE / sub / "equal_pfa").glob("*/ROW.json")):
            fam = FOLDER_LABELS.get(row.parent.name)
            if fam is None:      # BCS-oracle is kept on disk but excluded at the user's request
                continue
            out.setdefault(fam, []).append((None, row.parent))
    return out


def detect(folder: Path):
    z = np.load(folder / "SCORES.npz")
    thr = float(np.quantile(z["cal_h0"].reshape(-1), 1 - PFA, method="higher"))
    return {"ids": z["eval_parent_id"], "thr": thr,
            "pd": (z["eval_target"] > thr).mean(axis=1), "pfa": (z["eval_h0"] > thr).mean(axis=1),
            "pfa_noise": (z["eval_h0_noise"] > thr).mean(axis=1),
            "e0": z["eval_e0_modification"].mean(axis=1),
            "damage": (z["eval_gain_error"] + z["eval_orth_error"]).mean(axis=1),
            "resid_peak": np.median(z["eval_resid_peak_noise_units"], axis=1)}


def equal_pfa_block():
    sources = equal_pfa_sources()
    table, per_parent = {}, {}
    for fam, items in sources.items():
        rows = [detect(folder) for _, folder in items]
        ids = rows[0]["ids"]
        if any(not np.array_equal(r["ids"], ids) for r in rows):
            raise RuntimeError(f"evaluation parent order differs within {fam}")
        seeds = [s for s, _ in items]
        entry = {"seeds": seeds, "n_seeds": len(rows)}
        for k in ("pd", "pfa", "pfa_noise", "e0", "damage", "resid_peak"):
            means = np.array([r[k].mean() for r in rows])
            entry[k] = {"seed_mean": float(means.mean()),
                        "seed_sd": float(means.std(ddof=1)) if len(means) > 1 else None}
        entry["thresholds"] = [r["thr"] for r in rows]
        table[fam] = entry
        per_parent[fam] = {k: np.mean([r[k] for r in rows], axis=0) for k in ("pd", "pfa", "pfa_noise", "e0", "damage")}
        per_parent[fam]["ids"] = ids
    paired = {}
    ref = per_parent.get("ReCoVP-Hybrid")
    if ref is not None:
        for i, (fam, pp) in enumerate(sorted(per_parent.items())):
            if fam == "ReCoVP-Hybrid":
                continue
            if not np.array_equal(pp["ids"], ref["ids"]):
                raise RuntimeError(f"evaluation parents differ: {fam}")
            paired[fam] = {}
            for j, k in enumerate(("pd", "pfa", "e0", "damage")):
                d = ref[k] - pp[k]
                paired[fam][k] = {"mean_diff": float(d.mean()), "ci95": boot(d, 4100000 + 100 * i + j)}
    return table, paired


def dev_block():
    """Hybrid (all seeds with DEV arrays, averaged per parent) minus classical/CVEDNet on DEV256."""
    ref_arrays, used = [], []
    for s in HYBRID_SEEDS:
        p = ROOT / "outputs" / f"recovp_hybrid_seed{s}" / f"seed_{s}" / "dev_epoch_256_parent_values.npz"
        if p.is_file():
            z = np.load(p)
            ref_arrays.append(z)
            used.append(s)
    if not ref_arrays:
        return {}
    pid = ref_arrays[0]["parent_id"]
    hyb = {m: np.mean([z[f"hybrid__{m}"] for z in ref_arrays], axis=0) for m in DEV_METRICS}
    out = {"hybrid_seeds": used, "methods": {}}
    files = sorted((SUITE / "E6" / "dev256").glob("*/dev_parent_values.npz")) + \
        sorted((SUITE / "E7" / "dev256").glob("*/dev_parent_values.npz"))
    for i, f in enumerate(files):
        prefix = f.parent.name
        if prefix not in FOLDER_LABELS:
            continue
        z = np.load(f)
        if not np.array_equal(z["parent_id"], pid):
            raise RuntimeError(f"DEV parent order mismatch: {f}")
        res = {}
        for j, m in enumerate(DEV_METRICS):
            v = np.asarray(z[f"{prefix}__{m}"], np.float64)
            d = hyb[m] - v
            better = (d < 0) if m in LOWER else (d > 0)
            res[m] = {"hybrid_mean": float(hyb[m].mean()), "method_mean": float(v.mean()),
                      "mean_diff": float(d.mean()), "ci95": boot(d, 4200000 + 100 * i + j),
                      "fraction_parents_hybrid_better": float(better.mean())}
        out["methods"][FOLDER_LABELS[prefix]] = res
    return out


def fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.2f}" if pct else f"{x:.4g}"


def main():
    table, paired = equal_pfa_block()
    dev = dev_block()
    summary = {"target_frame_pfa": PFA, "bootstrap_reps": REPS, "equal_pfa": table,
               "paired_equal_pfa_hybrid_minus_method": paired, "paired_dev256_hybrid_minus_method": dev}
    (SUITE / "SUITE_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = ["# 等虚警（帧 Pfa 1%）检测汇总", "",
             "| 方法 | 种子数 | Pd (%) | Pd 种子SD | 实测帧Pfa (%) | 无干扰帧Pfa (%) | 无干扰改动 E0 | 目标损伤 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fam, e in sorted(table.items(), key=lambda kv: -kv[1]["pd"]["seed_mean"]):
        lines.append(f"| {fam} | {e['n_seeds']} | {fmt(e['pd']['seed_mean'], True)} | {fmt(e['pd']['seed_sd'], True)} | "
                     f"{fmt(e['pfa']['seed_mean'], True)} | {fmt(e['pfa_noise']['seed_mean'], True)} | "
                     f"{fmt(e['e0']['seed_mean'])} | {fmt(e['damage']['seed_mean'])} |")
    if paired:
        lines += ["", "## Hybrid 减各方法（等虚警，逐场景配对，95% bootstrap）", "",
                  "| 方法 | ΔPd (百分点) | 区间 | ΔE0 | 区间 | Δ损伤 | 区间 |", "|---|---:|---|---:|---|---:|---|"]
        for fam, p in sorted(paired.items()):
            lines.append(f"| {fam} | {100 * p['pd']['mean_diff']:+.2f} | [{100 * p['pd']['ci95'][0]:+.2f}, {100 * p['pd']['ci95'][1]:+.2f}] | "
                         f"{p['e0']['mean_diff']:+.3g} | [{p['e0']['ci95'][0]:+.3g}, {p['e0']['ci95'][1]:+.3g}] | "
                         f"{p['damage']['mean_diff']:+.3g} | [{p['damage']['ci95'][0]:+.3g}, {p['damage']['ci95'][1]:+.3g}] |")
    if dev:
        lines += ["", f"## DEV256：Hybrid（种子 {dev['hybrid_seeds']}）减传统方法与 CVEDNet", "",
                  "| 方法 | 指标 | Hybrid | 方法 | 差值 | 95% 区间 | Hybrid 更好的场景比例 |", "|---|---|---:|---:|---:|---|---:|"]
        for name, res in dev["methods"].items():
            for m, r in res.items():
                lines.append(f"| {name} | {m} | {r['hybrid_mean']:.4g} | {r['method_mean']:.4g} | {r['mean_diff']:+.4g} | "
                             f"[{r['ci95'][0]:+.4g}, {r['ci95'][1]:+.4g}] | {r['fraction_parents_hybrid_better']:.2f} |")
    (SUITE / "SUITE_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
