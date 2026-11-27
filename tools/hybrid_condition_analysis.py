"""Summaries and publication-size figures for the registered condition benchmark."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.hybrid_condition_benchmark import OUT, REPORT, ROOT, DATA, JOBS, PRIMARY_SNR, PRIMARY_JSR, EVAL_PARENTS, regime, cell_name, write_json, now, sha

METHODS = ["ReCoVP-Hybrid", "RSDR-Net", "UNet-Tail", "CorrectionNet-Tail"]
LABELS = {"ReCoVP-Hybrid": "Hybrid (5 seeds)", "RSDR-Net": "RSDR (frozen)", "UNet-Tail": "U-Net Tail (3 seeds)", "CorrectionNet-Tail": "CorrectionNet Tail (3 seeds)"}
COLORS = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
JSR_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
METRICS = ["pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "amplitude", "pd_eq_0.01", "pfa_eq_0.01", "pd_eq_0.05", "pfa_eq_0.05", "pd_eq_0.1", "pfa_eq_0.1", "pd_no_jammer", "frame_pfa_noise", "noise_nmse", "background_nmse", "false_peak_suppression_db", "phase_deg"]
TITLES = {"pd": "Detection probability (%)", "frame_pfa_jammer": "Jammed-frame false alarms (%)", "u_nmse": "$E_U$ (lower is better)", "damage": "$E_S$ (lower is better)", "sjrif_db": "SJRI-F (dB)", "amplitude": "Target amplitude ratio", "pd_eq_0.01": "Detection, 1% calibration (%)", "pfa_eq_0.01": "Actual false alarms, 1% calibration (%)"}


def savefig(fig, name):
    folder = REPORT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                         "pdf.fonttype": 42, "figure.dpi": 120})


def validate_evidence():
    from tools.paper2_baseline_suite import equal_pfa_suite as E, wp5b_suite as W
    from tools.rsdr_adapter import ASSETS, EXPECTED
    W.SUITE = ROOT / "outputs/paper2_baseline_suite_20261126"
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    revision = json.loads((REPORT / "SOURCE_REVISION.json").read_text(encoding="utf-8"))
    assert sha(REPORT / revision["evaluated_source"]) == revision["evaluated_sha256"]
    assert sha(ROOT / "tools/hybrid_condition_benchmark.py") == revision["current_sha256"]
    registration = json.loads((DATA / "REGISTRATION.json").read_text(encoding="utf-8"))
    assert sha(DATA / "components/test_0_0.npz") == manifest["component_sha256"]
    assert sha(DATA / "REGISTRATION.json") == manifest["registration_sha256"]
    for path, digest in registration["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    for name, digest in EXPECTED.items():
        assert sha(ASSETS / name) == digest, name
    structural = json.loads((DATA / "STRUCTURAL_METADATA_AUDIT.json").read_text(encoding="utf-8"))
    assert structural["exact_replay_all_arrays"]
    assert sha(DATA / "STRUCTURAL_METADATA.json.gz") == structural["sha256"]
    assert structural["component_sha256"] == manifest["component_sha256"]
    reference_done = json.loads((OUT / "references/DONE.json").read_text(encoding="utf-8"))
    assert reference_done["master_sha256"] == manifest["component_sha256"]
    reference_cells = {}
    with np.load(DATA / "components/test_0_0.npz", allow_pickle=False) as master:
        expected_ids = master["parent_id"][:EVAL_PARENTS]
    for label in ["Observed", "Ideal J removal"]:
        for snr in PRIMARY_SNR:
            for jsr in PRIMARY_JSR:
                path = OUT / "references" / label / f"{cell_name(snr, jsr)}.npz"
                with np.load(path, allow_pickle=False) as values:
                    assert np.array_equal(values["parent_id"], expected_ids)
                    assert all(np.isfinite(values[key]).all() for key in values.files)
                reference_cells[str(path.relative_to(OUT))] = sha(path)
    write_json(REPORT / "REFERENCE_VALIDATION.json", {
        "passed": True, "master_sha256": manifest["component_sha256"],
        "reference_done": reference_done, "cell_sha256": reference_cells})
    jobs = []
    for method, seed in JOBS:
        tag = f"seed_{seed}" if seed is not None else "single"
        folder = OUT / method / tag
        binding, done, parity = [json.loads((folder / f"{name}.json").read_text(encoding="utf-8")) for name in ["BINDING", "DONE", "PARITY"]]
        assert parity["passed"] and parity["cached_state_restoration"]
        assert binding["master_sha256"] == manifest["component_sha256"]
        assert binding["registration_sha256"] == manifest["registration_sha256"]
        rowpath = W.row_folder(method, seed) / "ROW.json"
        assert sha(rowpath) == binding["threshold_row_sha256"]
        row = json.loads(rowpath.read_text(encoding="utf-8"))
        if seed is not None:
            identity = E.checkpoint_binding(Path(binding["model"]["checkpoint"]), seed)
            assert identity["checkpoint_sha256"] == row["checkpoint_sha256"] == binding["model"]["checkpoint_sha256"]
        else:
            assert row["model"]["native_seed"] == binding["model"]["native_seed"] == 40312102
            assert row["model"]["epoch"] == binding["model"]["epoch"] == 32
        assert len(done["cell_sha256"]) == done["cells"] == 48
        assert done["runner_sha256"] == revision["evaluated_sha256"]
        for name, digest in done["cell_sha256"].items():
            assert sha(folder / name) == digest, name
        destination = REPORT / "bindings" / method / tag
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["BINDING", "DONE", "PARITY"]:
            shutil.copy2(folder / f"{name}.json", destination / f"{name}.json")
        jobs.append({"method": method, "seed": seed, "cells": done["cells"], "parity_passed": True,
                     "runner_sha256": done["runner_sha256"], "checkpoint": binding["model"],
                     "calibration_row_sha256": binding["threshold_row_sha256"]})
    write_json(REPORT / "VALIDATION.json", {"created_utc": now(), "passed": True, "jobs": jobs,
        "total_checkpoint_cells": sum(j["cells"] for j in jobs), "master_sha256": manifest["component_sha256"],
        "registration_sha256": manifest["registration_sha256"], "rsdr_asset_sha256": EXPECTED,
        "checks": ["master components", "registered data and radar sources", "fixed checkpoint SHA and calibration identity", "RSDR frozen assets and native identity", "all cell SHA", "frozen metric and frame-score parity", "cached restoration parity"]})


def old_seed_evidence():
    analysis = json.loads((ROOT / "reports/Hybrid_5seed_20261127/ANALYSIS.json").read_text(encoding="utf-8"))
    eq = analysis["equal_pfa"]["ReCoVP-Hybrid"]
    rows = []
    for i, seed in enumerate((17, 29, 41, 53, 67)):
        means = analysis["dev"][str(seed)]["means"]
        rows.append({"seed": seed, "epoch": 256, **{k: means[f"main__{k}"] for k in ["u_nmse", "damage", "pd", "frame_pfa_jammer", "amplitude", "sjrif_db", "noise_nmse"]},
                     "pd_eq_1pct": eq["curves"]["0.01"]["pd"]["values"][i],
                     "eqset_e0": eq["diagnostics"]["e0_modification"]["seed_means"]["values"][i]})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / "PREVIOUS_DEV_SEED_DETAILS.csv", index=False)
    old = pd.read_csv(ROOT / "reports/Hybrid_capability_status_20261127/SNR_JSR_METRICS.csv")
    old[old.method.isin(METHODS)].to_csv(REPORT / "PREVIOUS_32CELL_DETAIL.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.5))
    for ax, method in zip(axs, METHODS[:2]):
        d = old[old.method == method]
        for jsr, color in zip(PRIMARY_JSR, JSR_COLORS):
            c = d[d.jsr_db == jsr].sort_values("snr_db")
            ax.plot(c.snr_db, 100 * c.pd, "o-", color=color, label=f"JSR {jsr} dB")
        ax.set(title=LABELS[method], xlabel="SNR (dB)", ylabel="Detection probability (%)", ylim=(-2, 103))
        ax.legend(fontsize=8)
    fig.suptitle("Previous 32-cell grid: 8 SNR x 4 JSR, 64 physical parents per cell")
    fig.tight_layout()
    savefig(fig, "00_previous_grid_detection")
    return frame


def collect():
    arrays, cells, bytype, seeds = {}, [], [], []
    reference_ids = None
    for method in METHODS:
        method_jobs = [seed for m, seed in JOBS if m == method]
        checkpoints = []
        for seed in method_jobs:
            folder = OUT / method / (f"seed_{seed}" if seed is not None else "single")
            done = json.loads((folder / "DONE.json").read_text(encoding="utf-8"))
            if done["cells"] != len(PRIMARY_SNR) * len(PRIMARY_JSR):
                raise RuntimeError("incomplete method")
            condition_parts = []
            for snr in PRIMARY_SNR:
                for jsr in PRIMARY_JSR:
                    with np.load(folder / f"{cell_name(snr, jsr)}.npz", allow_pickle=False) as z:
                        if reference_ids is None:
                            reference_ids = z["parent_id"].copy()
                        if not np.array_equal(reference_ids, z["parent_id"]):
                            raise RuntimeError("unpaired result parents")
                        condition_parts.append(np.stack([z[k] for k in METRICS], axis=0))
            checkpoints.append(np.stack(condition_parts, axis=0))
        a = np.stack(checkpoints, axis=0)  # seed, cell, metric, physical parent, jammer type
        if not np.isfinite(a).all():
            raise RuntimeError("nonfinite results")
        arrays[method] = a
        seed_cell = a.mean(axis=(-2, -1))
        for c, (snr, jsr) in enumerate((s, j) for s in PRIMARY_SNR for j in PRIMARY_JSR):
            row = {"method": method, "n_seeds": len(method_jobs), "snr_db": snr, "jsr_db": jsr, "regime": regime(snr), "physical_parents": EVAL_PARENTS}
            for k, metric in enumerate(METRICS):
                values = seed_cell[:, c, k]
                row[metric] = float(values.mean())
                row[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
                row[f"{metric}_seed_min"] = float(values.min())
                row[f"{metric}_seed_max"] = float(values.max())
            cells.append(row)
            for jt, label in enumerate(["direct", "repeat", "cyclic"]):
                bytype.append({"method": method, "snr_db": snr, "jsr_db": jsr, "regime": regime(snr), "jammer_type": label,
                               **{metric: float(a[:, c, k, :, jt].mean()) for k, metric in enumerate(METRICS)}})
        for i, seed in enumerate(method_jobs):
            for group in ["extreme", "ordinary"]:
                mask = [regime(s) == group for s in PRIMARY_SNR for _ in PRIMARY_JSR]
                seeds.append({"method": method, "seed": seed, "regime": group,
                              **{metric: float(a[i, mask, k].mean()) for k, metric in enumerate(METRICS)}})
    frames = [pd.DataFrame(x) for x in (cells, bytype, seeds)]
    for f, name in zip(frames, ["CONDITION_METRICS.csv", "JAMMER_TYPE_METRICS.csv", "SEED_REGIME_METRICS.csv"]):
        f.to_csv(REPORT / name, index=False)
    return arrays, *frames


def aggregate(arrays):
    rows, contrasts = [], []
    rng = np.random.default_rng(2026112902)
    draws = rng.integers(EVAL_PARENTS, size=(4000, EVAL_PARENTS))
    parent = {}
    for group in ["extreme", "ordinary"]:
        mask = np.array([regime(s) == group for s in PRIMARY_SNR for _ in PRIMARY_JSR])
        for method, a in arrays.items():
            # Cells and jammer variants are repeated conditions of the same parent.
            by_seed_parent = a[:, mask].mean(axis=(1, 4))
            seed_means = by_seed_parent.mean(axis=-1)
            parent[(method, group)] = by_seed_parent.mean(axis=0)
            for k, metric in enumerate(METRICS):
                x = parent[(method, group)][k]
                low, high = np.quantile(x[draws].mean(axis=1), [.025, .975])
                rows.append({"method": method, "regime": group, "metric": metric, "mean": float(x.mean()),
                             "seed_sd": float(seed_means[:, k].std(ddof=1)) if len(a) > 1 else None,
                             "seed_min": float(seed_means[:, k].min()), "seed_max": float(seed_means[:, k].max()),
                             "parent_ci95_low": float(low), "parent_ci95_high": float(high),
                             "independent_parents": EVAL_PARENTS, "conditions": int(mask.sum())})
        for other in METHODS[1:]:
            delta = parent[(METHODS[0], group)] - parent[(other, group)]
            for k, metric in enumerate(METRICS):
                low, high = np.quantile(delta[k][draws].mean(axis=1), [.025, .975])
                contrasts.append({"regime": group, "contrast": f"Hybrid minus {other}", "metric": metric,
                                  "difference": float(delta[k].mean()), "paired_parent_ci95_low": float(low), "paired_parent_ci95_high": float(high)})
    summary, paired = pd.DataFrame(rows), pd.DataFrame(contrasts)
    summary.to_csv(REPORT / "REGIME_SUMMARY.csv", index=False)
    paired.to_csv(REPORT / "PAIRED_REGIME_DIFFERENCES.csv", index=False)
    return summary, paired


def detection_curves(cells, metric, name):
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 5))
    for ax in axs:
        for method, style, marker in [(METHODS[0], "-", "o"), (METHODS[1], "--", "s")]:
            d = cells[cells.method == method]
            for jsr, color in zip(PRIMARY_JSR, JSR_COLORS):
                c = d[d.jsr_db == jsr].sort_values("snr_db")
                x, y = c.snr_db.to_numpy(), 100 * c[metric].to_numpy()
                ax.plot(x, y, style, marker=marker, ms=3.5, color=color, lw=1.8,
                        label=f"{'Hybrid' if method == METHODS[0] else 'RSDR'} / JSR {jsr} dB")
                if method == METHODS[0]:
                    ax.fill_between(x, 100 * c[f"{metric}_seed_min"], 100 * c[f"{metric}_seed_max"], color=color, alpha=.09)
        ax.axvline(-20, color=".4", ls=":", lw=1)
        upper = 102 if metric != "frame_pfa_jammer" else max(5, 115 * cells[metric].max())
        ax.set(xlabel="SNR (dB)", ylabel=TITLES[metric], ylim=(-.02 * upper, upper))
    axs[0].set(title="Extreme and ordinary conditions", xlim=(-30.5, .5))
    axs[1].set(title="Weak-signal detail", xlim=(-30.5, -19.5))
    axs[0].legend(fontsize=7.6, ncol=2, loc="lower right")
    fig.suptitle("128 paired parents per cell, shading = Hybrid training-seed range", fontsize=11)
    fig.tight_layout()
    savefig(fig, name)


def reference_and_condition_plots(cells):
    references = pd.read_csv(REPORT / "REFERENCE_CONDITION_METRICS.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric in zip(axs, ["pd", "frame_pfa_jammer"]):
        for method in METHODS:
            d = cells[(cells.method == method) & (cells.jsr_db == 20)].sort_values("snr_db")
            ax.plot(d.snr_db, 100 * d[metric], "o-", ms=3, color=COLORS[method], label=LABELS[method])
        for method, color, style in [("Observed", ".55", ":"), ("Ideal J removal", "black", "--")]:
            d = references[(references.method == method) & (references.jsr_db == 20)].sort_values("snr_db")
            ax.plot(d.snr_db, 100 * d[metric], style, color=color, lw=2, label=method)
        ax.axvline(-20, color=".4", ls=":", lw=1)
        ax.set(xlabel="SNR (dB)", ylabel=TITLES[metric], ylim=(-2, 102))
    axs[0].legend(fontsize=7.5)
    fig.suptitle("JSR = 20 dB, truth-known S+E reference retains noise, nominal CFAR")
    fig.tight_layout()
    savefig(fig, "08_detection_with_noise_reference")
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey="row")
    for column, jsr in enumerate(PRIMARY_JSR):
        for row, metric in enumerate(["pd_eq_0.01", "pfa_eq_0.01"]):
            ax = axs[row, column]
            for method in METHODS:
                d = cells[(cells.method == method) & (cells.jsr_db == jsr)].sort_values("snr_db")
                ax.plot(d.snr_db, 100 * d[metric], "o-", ms=3, color=COLORS[method], label=LABELS[method])
            ax.axvline(-20, color=".4", ls=":", lw=1)
            if row == 0:
                ax.set_ylim(-2, 102)
            else:
                maximum = 100 * cells[metric].max()
                ax.set_ylim(0, max(3, maximum * 1.15))
            ax.set(title=f"JSR {jsr} dB", xlabel="SNR (dB)")
            if column == 0:
                ax.set_ylabel(TITLES[metric])
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen global 1% calibration threshold, actual condition-specific false alarms below")
    fig.tight_layout()
    savefig(fig, "09_calibrated_detection_and_actual_false_alarms")
    fig, axs = plt.subplots(3, 4, figsize=(16, 10.8), sharex=True, sharey="row")
    for column, jsr in enumerate(PRIMARY_JSR):
        for row, metric in enumerate(["u_nmse", "damage", "sjrif_db"]):
            ax = axs[row, column]
            for method in METHODS:
                d = cells[(cells.method == method) & (cells.jsr_db == jsr)].sort_values("snr_db")
                ax.plot(d.snr_db, d[metric], "o-", ms=3, color=COLORS[method], label=LABELS[method])
            ax.axvline(-20, color=".4", ls=":", lw=1)
            ax.set(title=f"JSR {jsr} dB", xlabel="SNR (dB)")
            if column == 0:
                ax.set_ylabel(TITLES[metric])
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Recovery, target distortion and interference suppression by SNR and JSR")
    fig.tight_layout()
    savefig(fig, "10_error_and_suppression_by_jsr")
    detection_curves(cells, "frame_pfa_jammer", "11_false_alarms_snr_jsr")
    columns = [m for m in METRICS]
    cells.groupby(["method", "regime", "jsr_db"], as_index=False)[columns].mean().to_csv(
        REPORT / "REGIME_BY_JSR.csv", index=False)
    points = []
    for method in METHODS:
        for jsr in PRIMARY_JSR:
            d = cells[(cells.method == method) & (cells.jsr_db == jsr)].sort_values("snr_db")
            for metric in ["pd", "pd_eq_0.01", "pd_eq_0.05", "pd_eq_0.1"]:
                for goal in [.5, .9, .95]:
                    reached = d[d[metric] >= goal]
                    points.append({"method": method, "jsr_db": jsr, "metric": metric, "pd_goal": goal,
                                   "first_tested_snr_db": float(reached.snr_db.iloc[0]) if len(reached) else None,
                                   "measured_pd": float(reached[metric].iloc[0]) if len(reached) else None,
                                   "interpretation": "First registered SNR sample reaching goal; no interpolation"})
    pd.DataFrame(points).to_csv(REPORT / "DETECTION_OPERATING_POINTS.csv", index=False)


def plots(cells, bytype, seed_rows, summary, old_seeds):
    detection_curves(cells, "pd", "01_detection_snr_jsr")
    detection_curves(cells, "pd_eq_0.01", "02_detection_fixed_1pct_calibration")
    metrics = ["pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "amplitude"]
    h = cells[cells.method == METHODS[0]].set_index(["snr_db", "jsr_db"])
    r = cells[cells.method == METHODS[1]].set_index(["snr_db", "jsr_db"])
    fig, axs = plt.subplots(2, 3, figsize=(13.8, 9))
    for ax, metric in zip(axs.ravel(), metrics):
        scale = 100 if metric in ("pd", "frame_pfa_jammer") else 1
        delta = ((h[metric] - r[metric]) * scale).unstack("jsr_db").reindex(PRIMARY_SNR)
        limit = max(float(np.abs(delta.to_numpy()).max()), 1e-8)
        image = ax.imshow(delta, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(4), PRIMARY_JSR)
        ax.set_yticks(range(len(PRIMARY_SNR)), PRIMARY_SNR)
        ax.set(xlabel="JSR (dB)", ylabel="SNR (dB)", title=TITLES[metric].replace(" (%)", " (pp)"))
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.035)
    fig.suptitle("Hybrid minus RSDR, per condition (sign must be read with the metric)")
    fig.tight_layout()
    savefig(fig, "03_hybrid_minus_rsdr_heatmaps")
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
    x = np.arange(2)
    for ax, metric in zip(axs.ravel(), metrics):
        scale = 100 if metric in ("pd", "frame_pfa_jammer") else 1
        for i, method in enumerate(METHODS):
            d = summary[(summary.method == method) & (summary.metric == metric)].set_index("regime").reindex(["extreme", "ordinary"])
            y = d["mean"].to_numpy() * scale
            errors = np.maximum(0, np.vstack([y - d.parent_ci95_low.to_numpy() * scale, d.parent_ci95_high.to_numpy() * scale - y]))
            ax.bar(x + (i - 1.5) * .19, y, width=.18, color=COLORS[method], label=LABELS[method], yerr=errors, capsize=2)
        ax.set_xticks(x, ["Extreme\n[-30, -20)", "Ordinary\n[-20, 0]"])
        ax.set_title(TITLES[metric])
    axs[0, 0].legend(fontsize=7.6)
    fig.suptitle("Regime summaries, equal cell weighting, physical-parent bootstrap intervals")
    fig.tight_layout()
    savefig(fig, "04_regime_dashboard")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ids = old_seeds.seed.astype(str)
    axs[0].bar(ids, old_seeds.eqset_e0, color=[COLORS[METHODS[0]]] * 4 + ["#D55E00"])
    axs[0].set(yscale="log", title="Previous equal-PFA set, noise change $E_0$", xlabel="Hybrid training seed")
    axs[1].bar(ids, old_seeds.pd_eq_1pct * 100, color=COLORS[METHODS[0]])
    axs[1].set(title="Previous set, Pd with 1% calibration", ylabel="Detection (%)", xlabel="Hybrid training seed")
    for group, style in [("extreme", "o-"), ("ordinary", "s--")]:
        d = seed_rows[(seed_rows.method == METHODS[0]) & (seed_rows.regime == group)]
        axs[2].plot(d.seed.astype(int).astype(str), d.pd * 100, style, label=group)
    axs[2].set(title="New master, nominal detection", ylabel="Detection (%)", xlabel="Hybrid training seed")
    axs[2].legend()
    fig.tight_layout()
    savefig(fig, "05_training_seed_stability")
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for row, group in enumerate(["extreme", "ordinary"]):
        for ax, metric in zip(axs[row], ["pd", "frame_pfa_jammer", "u_nmse"]):
            scale = 100 if metric != "u_nmse" else 1
            for i, method in enumerate(METHODS):
                d = bytype[(bytype.method == method) & (bytype.regime == group)].groupby("jammer_type")[metric].mean().reindex(["direct", "repeat", "cyclic"])
                ax.bar(np.arange(3) + (i - 1.5) * .19, d.to_numpy() * scale, width=.18, color=COLORS[method], label=LABELS[method])
            ax.set_xticks(range(3), ["direct", "repeat", "cyclic"])
            ax.set_title(f"{group.capitalize()} / {TITLES[metric]}")
    axs[0, 0].legend(fontsize=7.3)
    fig.tight_layout()
    savefig(fig, "06_jammer_types_by_regime")
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, metric in zip(axs.ravel(), metrics):
        scale = 100 if metric in ("pd", "frame_pfa_jammer") else 1
        for method in METHODS:
            d = cells[(cells.method == method) & (cells.jsr_db == 20)].sort_values("snr_db")
            ax.plot(d.snr_db, d[metric] * scale, "o-", color=COLORS[method], ms=3, label=LABELS[method])
        ax.axvline(-20, color=".4", lw=1, ls=":")
        ax.set(xlabel="SNR (dB)", title=TITLES[metric])
    axs[0, 0].legend(fontsize=7.3)
    fig.suptitle("Four-model condition profiles at JSR = 20 dB")
    fig.tight_layout()
    savefig(fig, "07_six_metrics_at_jsr20")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    validate_evidence()
    setup()
    old = old_seed_evidence()
    arrays, cells, bytype, seed_rows = collect()
    summary, paired = aggregate(arrays)
    plots(cells, bytype, seed_rows, summary, old)
    reference_and_condition_plots(cells)
    write_json(REPORT / "ANALYSIS_SUMMARY.json", {"created_utc": now(), "all_jobs_complete": True,
        "jobs": JOBS, "primary_snr_db": PRIMARY_SNR, "primary_jsr_db": PRIMARY_JSR,
        "parents_per_cell": EVAL_PARENTS, "regime_rows": summary.astype(object).where(pd.notnull(summary), None).to_dict("records"),
        "paired_differences": paired.to_dict("records"), "bootstrap_replicates": 4000,
        "bootstrap_unit": "physical parent jointly across conditions and jammer types", "old_seed_rows": old.to_dict("records")})
    write_condition_tables(cells, bytype, seed_rows, summary, paired)
    print(summary[summary.metric.isin(["pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "pd_eq_0.01", "pfa_eq_0.01"])].to_string(index=False))


def write_condition_tables(cells, bytype, seed_rows, summary, paired):
    """Keep the report's complete numerical tables reproducible from CSV evidence."""
    lines = ["# 分工况完整数值表", "", "概率列使用百分数。EU、ES、幅度比使用线性值，SJRI-F 使用 dB。Hybrid 为五种子均值，两个卷积方法各为三种子均值，RSDR 为单个冻结训练实例。", ""]

    def table(frame, columns, headers, percentage=()):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, str):
                    values.append(value)
                elif pd.isna(value):
                    values.append("单实例")
                elif column in percentage:
                    values.append(f"{value * 100:.3f}%")
                else:
                    values.append(f"{value:.6g}")
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    for group, label in [("extreme", "极端"), ("ordinary", "普通")]:
        lines.extend([f"## {label}工况汇总", ""])
        rows = summary[summary.regime == group].pivot(index="method", columns="metric", values="mean").reset_index()
        columns = ["method", "pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "amplitude"]
        table(rows, columns, ["方法", "Pd", "受扰帧虚警", "EU", "ES", "SJRI-F", "幅度比"], ["pd", "frame_pfa_jammer"])
        lines.extend(["### 全局校准门限后的检测与本组实际虚警", ""])
        columns = ["method", "pd_eq_0.01", "pfa_eq_0.01", "pd_eq_0.05", "pfa_eq_0.05", "pd_eq_0.1", "pfa_eq_0.1"]
        table(rows, columns, ["方法", "1%校准 Pd", "实际虚警", "5%校准 Pd", "实际虚警", "10%校准 Pd", "实际虚警"], columns[1:])
        lines.extend(["### Hybrid 逐训练种子", ""])
        columns = ["seed", "pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "pd_eq_0.01", "noise_nmse"]
        table(seed_rows[(seed_rows.method == METHODS[0]) & (seed_rows.regime == group)], columns,
              ["seed", "Pd", "受扰帧虚警", "EU", "ES", "SJRI-F", "1%校准 Pd", "E0"], ["pd", "frame_pfa_jammer", "pd_eq_0.01"])
        lines.extend(["### 三类干扰", ""])
        rows = bytype[bytype.regime == group].groupby(["method", "jammer_type"], as_index=False)[["pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db"]].mean()
        table(rows, ["method", "jammer_type", "pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db"],
              ["方法", "干扰", "Pd", "受扰帧虚警", "EU", "ES", "SJRI-F"], ["pd", "frame_pfa_jammer"])

    for metric, label in [("pd", "检测概率"), ("frame_pfa_jammer", "受扰帧虚警"), ("u_nmse", "EU"), ("damage", "ES"), ("sjrif_db", "SJRI-F")]:
        lines.extend([f"## 逐 SNR×JSR 的{label}", ""])
        rows = cells.pivot(index=["snr_db", "jsr_db"], columns="method", values=metric).reindex(columns=METHODS).reset_index()
        table(rows, ["snr_db", "jsr_db"] + METHODS, ["SNR dB", "JSR dB"] + METHODS,
              METHODS if metric in ("pd", "frame_pfa_jammer") else [])

    lines.extend(["## 配对场景差值与区间", "", "差值按 Hybrid 减对照计算，概率列使用原始 0 至 1 数值。区间来自 4,000 次配对物理 parent bootstrap。训练种子差异另见 SEED_REGIME_METRICS.csv。", ""])
    columns = ["regime", "contrast", "metric", "difference", "paired_parent_ci95_low", "paired_parent_ci95_high"]
    table(paired[paired.metric.isin(["pd", "frame_pfa_jammer", "u_nmse", "damage", "sjrif_db", "pd_eq_0.01"])], columns,
          ["分组", "差值方向", "指标", "差值", "95%下限", "95%上限"])
    (REPORT / "CONDITION_TABLES.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
