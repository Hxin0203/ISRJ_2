"""Export readable review figures from the completed, unchanged convergence CSVs."""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper2.io import ROOT, now, sha
from tools.durable_io import atomic_json

SOURCE = ROOT / "evidence/outputs/CONVERGENCE_256_8SEED_20261114"
OUT = ROOT / "evidence/outputs/PROGRESS_CHECK_20261115"
SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#E69F00", "#56B4E9", "#000000", "#777777")


def draw(filename, panels, shape, title, stem):
    with (SOURCE / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(*shape, figsize=(14, 8.5), squeeze=False)
    fig.subplots_adjust(top=0.81, bottom=0.08, left=0.07, right=0.98,
                        hspace=0.38, wspace=0.27)
    fig.suptitle(title, y=0.98, fontsize=15)
    fig.text(0.5, 0.935,
             "Solid: seeds continued from epoch 128; dashed: new seeds. Band: seed mean +/- sample SD.",
             ha="center", fontsize=10)
    for panel_index, (ax, (field, label, logarithmic)) in enumerate(zip(axes.flat, panels)):
        values = []
        for index, seed in enumerate(SEEDS):
            selected = sorted((r for r in rows if int(r["seed"]) == seed),
                              key=lambda r: int(r["epoch"]))
            x = np.array([int(r["epoch"]) for r in selected])
            y = np.array([float(r[field]) for r in selected])
            values.append(y)
            ax.plot(x, y, color=COLORS[index], lw=0.9, alpha=0.8,
                    linestyle="-" if index < 3 else "--",
                    marker="o" if len(x) == 8 else None, markersize=3,
                    label=f"Seed {seed}" if panel_index == 0 else None)
        matrix = np.stack(values)
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        ax.plot(x, mean, color="#222222", lw=2, label="Mean +/- SD" if panel_index == 0 else None)
        lower = np.maximum(mean-sd, np.finfo(float).tiny) if logarithmic else mean-sd
        ax.fill_between(x, lower, mean+sd, color="#555555", alpha=0.13)
        if logarithmic:
            ax.set_yscale("log")
        if len(x) == 8:
            ax.set_xticks(x)
        ax.set(title=label, xlabel="Epoch")
        ax.grid(True, which="both", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=5, frameon=False, fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw("DEV_METRICS_BY_CHECKPOINT.csv", (
        ("model__u_nmse__mean", "Restored-signal NMSE", True),
        ("model__damage__mean", "Target-increment error", True),
        ("model__no_jammer_pooled_nmse__mean", "No-jammer pooled NMSE", True),
        ("model__sjrif_db__mean", "SJRIF (dB)", False),
        ("model__pd__mean", "Detection probability", False),
        ("model__frame_pfa_jammer__mean", "Frame false alarm: jammer + noise", False),
    ), (2, 3), "Eight-seed DEV convergence: 32 to 256 epochs", "DEV_CONVERGENCE_REVIEW")
    draw("TRAINING_LOSS_BY_EPOCH.csv", (
        ("total", "Total B4 objective", True),
        ("endpoint", "Endpoint loss", True),
        ("pc", "Pulse-compression loss", True),
        ("damage", "Target-increment loss", True),
    ), (2, 2), "Eight-seed training convergence: 1 to 256 epochs", "TRAINING_CONVERGENCE_REVIEW")
    atomic_json(OUT / "FIGURE_MANIFEST.json", {
        "created_utc": now(), "source_data_changed": False,
        "inputs_sha256": {name: sha(SOURCE / name) for name in
                          ("DEV_METRICS_BY_CHECKPOINT.csv", "TRAINING_LOSS_BY_EPOCH.csv")},
        "script_sha256": sha(ROOT / "tools/plot_convergence_review_20261115.py"),
        "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.glob("*_REVIEW.*"))},
        "uncertainty": "Descriptive sample SD over eight training seeds, not a confidence interval.",
    })


if __name__ == "__main__":
    main()
