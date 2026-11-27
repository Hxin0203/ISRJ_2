"""Jammer-frame false alarm vs JNR = SNR + JSR (read-only, CPU).

Source: reports/Hybrid_condition_benchmark_20261127/CONDITION_METRICS.csv
(48 registered SNR x JSR cells, 128 physical parents, nominal CFAR).
For each method, cells with equal JNR are plotted individually so that the
spread within one JNR value is visible (zero for scale-equivariant or
RMS-normalized processors).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports/Hybrid_condition_benchmark_20261127/CONDITION_METRICS.csv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evidence/outputs/paper2_baseline_suite_20261126/FIG_PFA_JNR"
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
styles = {"ReCoVP-Hybrid": ("C3", "o"), "RSDR-Net": ("k", "s"),
          "UNet-Tail": ("C0", "^"), "CorrectionNet-Tail": ("C2", "v")}
fig, ax = plt.subplots(figsize=(7, 4.2))
summary = {}
for m, (c, mk) in styles.items():
    pts = defaultdict(list)
    for r in rows:
        if r["method"] == m:
            pts[int(r["snr_db"]) + int(r["jsr_db"])].append(100 * float(r["frame_pfa_jammer"]))
    xs = sorted(pts)
    ax.plot(xs, [sum(pts[x]) / len(pts[x]) for x in xs], color=c, lw=1.2, label=m)
    for x in xs:
        ax.scatter([x] * len(pts[x]), pts[x], color=c, marker=mk, s=14, zorder=3)
    summary[m] = {str(x): {"mean": sum(v) / len(v), "range": max(v) - min(v), "cells": len(v)} for x, v in pts.items()}
ax.set_xlabel("JNR = SNR + JSR (dB)")
ax.set_ylabel("Jammer-frame false alarm (%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_pfa_vs_jnr.{ext}", dpi=200)
(OUT / "PFA_VS_JNR.json").write_text(json.dumps({"source": str(SRC.relative_to(ROOT)), "methods": summary}, indent=1), encoding="utf-8")
print("wrote", OUT)
