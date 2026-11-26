#!/usr/bin/env python3
"""Plot Hybrid seed17 against first-paper RSDR by SNR and JSR strata.

Inputs are existing local audit artifacts. This script does no model inference.
Hybrid summaries use parent-level means and parent bootstrap CIs; RSDR summaries
use equal-weight means of condition-level means and have no pooled CI available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SNR_EDGES = np.array([-20, -15, -10, -5, 0, 5], dtype=float)
JSR_EDGES = np.array([-10, 0, 10, 20, 30, 40], dtype=float)
SEED = 20261125
BOOTSTRAPS = 3000

METRICS = {
    "u_nmse": ("u_nmse", "u_nmse (lower is better)", True),
    "damage": ("damage", "e_S / damage (lower is better)", True),
    "pd": ("pd", "Detection probability, P_d", False),
    "frame_pfa_jammer": ("frame_pfa_jammer", "Jammer-frame false alarm rate", False),
    "sjrif_db": ("sjrif_db", "SJRI-F (dB)", False),
}
JSR_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00"]
JSR_LABELS = ["[-10,0)", "[0,10)", "[10,20)", "[20,30)", "[30,40)"]


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Right-open bins, including the declared lower edge and excluding upper.
    ids = np.full(values.shape, -1, dtype=int)
    good = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
    ids[good] = np.digitize(values[good], edges, right=False) - 1
    return ids


def _bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def aggregate_hybrid(dev_npz: Path, parents_npz: Path) -> list[dict]:
    with np.load(dev_npz, allow_pickle=False) as d, np.load(parents_npz, allow_pickle=False) as p:
        snr_parent = np.nanmean(d["snr_db"], axis=1)
        # Match the evaluation code: average pulse-level JSR per jammer variant,
        # then average the three variants equally for each physical parent.
        jsr_variant = np.nanmean(d["jsr_db"], axis=2)
        jsr_parent = np.nanmean(jsr_variant, axis=1)
        snr_ids = _bin_ids(snr_parent, SNR_EDGES)
        jsr_ids = _bin_ids(jsr_parent, JSR_EDGES)
        rows: list[dict] = []
        for metric_index, (metric, _) in enumerate(METRICS.items()):
            values = p[f"hybrid__{metric}"]
            for si in range(len(SNR_EDGES) - 1):
                for ji in range(len(JSR_EDGES) - 1):
                    mask = (snr_ids == si) & (jsr_ids == ji) & np.isfinite(values)
                    n = int(mask.sum())
                    mean, lo, hi = _bootstrap_mean(
                        values[mask], SEED + metric_index * 1000 + si * 100 + ji
                    )
                    rows.append({
                        "model": "Hybrid seed17 epoch256 DEV",
                        "metric": metric,
                        "snr_bin": f"[{SNR_EDGES[si]:g},{SNR_EDGES[si+1]:g})",
                        "snr_mid_db": (SNR_EDGES[si] + SNR_EDGES[si+1]) / 2,
                        "jsr_bin": f"[{JSR_EDGES[ji]:g},{JSR_EDGES[ji+1]:g})",
                        "jsr_mid_db": (JSR_EDGES[ji] + JSR_EDGES[ji+1]) / 2,
                        "mean": mean,
                        "ci95_lower": lo,
                        "ci95_upper": hi,
                        "n_parents": n,
                        "n_conditions": 0,
                    })
    return rows


def aggregate_rsdr(condition_csv: Path) -> list[dict]:
    frame = pd.read_csv(condition_csv)
    frame = frame[(frame["suite"] == "main") & frame["metric"].isin(METRICS)]
    frame["snr_db"] = pd.to_numeric(frame["snr_db"], errors="coerce")
    frame["jsr_db"] = pd.to_numeric(frame["jsr_db"], errors="coerce")
    frame["rsdr_mean"] = pd.to_numeric(frame["rsdr_mean"], errors="coerce")
    snr_ids = _bin_ids(frame["snr_db"].to_numpy(), SNR_EDGES)
    jsr_ids = _bin_ids(frame["jsr_db"].to_numpy(), JSR_EDGES)
    frame = frame.assign(_snr_bin=snr_ids, _jsr_bin=jsr_ids)
    frame = frame[(frame["_snr_bin"] >= 0) & (frame["_jsr_bin"] >= 0)]
    rows: list[dict] = []
    for metric in METRICS:
        fm = frame[frame["metric"] == metric]
        for si in range(len(SNR_EDGES) - 1):
            for ji in range(len(JSR_EDGES) - 1):
                cell = fm[(fm["_snr_bin"] == si) & (fm["_jsr_bin"] == ji)]
                rows.append({
                    "model": "RSDR-Net seed40312102 epoch32 main scan",
                    "metric": metric,
                    "snr_bin": f"[{SNR_EDGES[si]:g},{SNR_EDGES[si+1]:g})",
                    "snr_mid_db": (SNR_EDGES[si] + SNR_EDGES[si+1]) / 2,
                    "jsr_bin": f"[{JSR_EDGES[ji]:g},{JSR_EDGES[ji+1]:g})",
                    "jsr_mid_db": (JSR_EDGES[ji] + JSR_EDGES[ji+1]) / 2,
                    "mean": float(cell["rsdr_mean"].mean()) if len(cell) else np.nan,
                    "ci95_lower": np.nan,
                    "ci95_upper": np.nan,
                    "n_parents": 0,
                    "n_conditions": int(len(cell)),
                })
    return rows


def plot_metric(rows: list[dict], metric: str, outdir: Path) -> None:
    _, ylabel, _ = METRICS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, layout="constrained")
    all_values = [r["mean"] for r in rows if r["metric"] == metric and np.isfinite(r["mean"])]
    for r in rows:
        if r["metric"] == metric:
            all_values.extend(v for v in (r["ci95_lower"], r["ci95_upper"]) if np.isfinite(v))
    ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
    if metric in ("pd", "frame_pfa_jammer"):
        shared_ylim = (0.0, 1.0)
    elif metric in ("u_nmse", "damage"):
        shared_ylim = (0.0, ymax * 1.12 if ymax > 0 else 1.0)
    else:
        pad = max((ymax - ymin) * 0.08, 0.5)
        shared_ylim = (ymin - pad, ymax + pad)
    model_specs = [
        (axes[0], "Hybrid seed17", "Hybrid seed17 epoch256 DEV"),
        (axes[1], "First-paper RSDR-Net (12 condition points/bin)", "RSDR-Net seed40312102 epoch32 main scan"),
    ]
    for ax, title, model in model_specs:
        subset = [r for r in rows if r["metric"] == metric and r["model"] == model]
        for ji, (label, color) in enumerate(zip(JSR_LABELS, JSR_COLORS)):
            points = sorted((r for r in subset if r["jsr_bin"] == label), key=lambda r: r["snr_mid_db"])
            # Missing cross-strata are kept as gaps; no interpolation.
            x = np.array([r["snr_mid_db"] for r in points], dtype=float)
            y = np.array([r["mean"] for r in points], dtype=float)
            n = np.array([r["n_parents"] if r["n_parents"] else r["n_conditions"] for r in points])
            y[n == 0] = np.nan
            linestyle = "-" if ax is axes[0] else "--"
            marker = "o" if ax is axes[0] else "s"
            ax.plot(x, y, color=color, linestyle=linestyle, marker=marker,
                    linewidth=1.8, markersize=5, label=f"JSR {label} dB")
            if ax is axes[0]:
                lower = np.array([r["ci95_lower"] for r in points], dtype=float)
                upper = np.array([r["ci95_upper"] for r in points], dtype=float)
                valid = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper) & (n > 1)
                if valid.any():
                    ax.fill_between(x, lower, upper, where=valid, color=color, alpha=0.12, linewidth=0)
            for xx, yy, nn in zip(x, y, n):
                if np.isfinite(yy) and ax is axes[0] and nn < 10:
                    ax.annotate(f"n={nn}", (xx, yy), xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=6.5, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_ylim(*shared_ylim)
        ax.set_xlabel("SNR (dB; 5 dB bins, midpoint shown)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([-17.5, -12.5, -7.5, -2.5, 2.5])
        ax.set_xticklabels(["−17.5", "−12.5", "−7.5", "−2.5", "2.5"])
        ax.grid(True, color="#D8DDE3", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="JSR bin (dB)", fontsize=8, title_fontsize=8,
               ncols=5, loc="outside lower center")
    fig.suptitle(f"{metric}: ReCoVP-Hybrid vs first-paper RSDR-Net", fontsize=13)
    stem = outdir / f"{metric}_by_snr_jsr"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-npz", type=Path, required=True)
    parser.add_argument("--parents-npz", type=Path, required=True)
    parser.add_argument("--rsdr-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_hybrid(args.dev_npz, args.parents_npz) + aggregate_rsdr(args.rsdr_csv)
    data_path = args.outdir / "SNR_JSR_metric_summary.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False, float_format="%.8g")
    for metric in METRICS:
        plot_metric(rows, metric, args.outdir)
    print(f"wrote {len(rows)} summary rows and {len(METRICS)} paired model figures to {args.outdir}")


if __name__ == "__main__":
    main()
