"""WP5b local plotting from SUMMARY.json (pulled from server B). Repeat jammer type (primary).
Row 1: slice width w (M=2, period (M+1)w+1). Row 2: unaffected fraction (w=36, period 109/145/217).
Columns: E_U, E_S, Pd at each method's own 5% calibration threshold, jammer-frame Pfa (nominal CFAR).
Hybrid: mean of 5 seeds with seed-SD band; conv nets: mean of 3 seeds; others single model.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[2] / "evidence/outputs/paper2_baseline_suite_20261126/WP5b"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9})
S = json.loads((HERE / "SUMMARY.json").read_text(encoding="utf-8"))
T, eta = S["table"], S["actual_eta_repeat"]
SHOW = [("ReCoVP-Hybrid", "#c0392b", "-", "o"), ("ReCoVP-Stage1", "#e59866", "--", "o"),
        ("UNet-Tail", "#3A8A8A", "-", "s"), ("CN-NoDechirp", "#7fb3b3", "-", "^"),
        ("CorrectionNet", "#8FAFC4", "-", "v"), ("RSDR-Net", "#4F86A6", "-", "D"),
        ("Max-TF", "#8e7cc3", ":", "x"), ("CVEDNet", "#9aa5ad", ":", "+")]
COLS = [("repeat:u_nmse", r"$E_U$", True), ("repeat:damage", r"$E_S$", True),
        ("repeat:pd_eq0.05", r"$P_d$ (own 5% threshold)", False), ("repeat:frame_pfa_jammer", "jammer-frame $P_{fa}$ (nominal)", False)]
ROWS = [("width", ["w06", "w18", "w36", "w54", "w72"], lambda c: S["cells"][c]["width"], "slice width $w$ (samples)"),
        ("eta", ["w36", "eta_p145", "eta_p217"], lambda c: eta[c], "unaffected fraction (repeat, measured)")]
fig, ax = plt.subplots(2, 4, figsize=(7.4, 3.9), constrained_layout=True)
for r, (_, cells, xf, xlab) in enumerate(ROWS):
    x = np.array([xf(c) for c in cells])
    for name, col, ls, mk in SHOW:
        if name not in T:
            continue
        for k, (metric, ylab, logy) in enumerate(COLS):
            m = np.array([T[name]["cells"][c][metric]["mean"] for c in cells])
            a = ax[r, k]
            a.plot(x, m, color=col, ls=ls, marker=mk, ms=3.5, lw=1.1, label=f"{name} ({T[name]['seeds']})")
            if name == "ReCoVP-Hybrid":
                sd = np.array([T[name]["cells"][c][metric]["seed_sd"] or 0 for c in cells])
                a.fill_between(x, m - sd, m + sd, color=col, alpha=.15, lw=0)
            if logy:
                a.set_yscale("log")
            if r == 0:
                a.set_title(ylab, fontsize=9)
            a.set_xlabel(xlab, fontsize=8)
    ax[0, 0].axvspan(5, 6.5, color="#dddddd", alpha=.5, lw=0)
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.13))
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig10_structure_sweep.{ext}", dpi=300, bbox_inches="tight")
rows = []
for name, *_ in SHOW:
    if name in T:
        for c in ROWS[0][1] + ROWS[1][1][1:]:
            v = T[name]["cells"][c]
            rows.append(f"| {name} | {c} | {v['repeat:u_nmse']['mean']:.4f} | {v['repeat:damage']['mean']:.4f} | "
                        f"{100 * v['repeat:pd_eq0.05']['mean']:.1f} | {100 * v['repeat:pfa_eq0.05']['mean']:.1f} | "
                        f"{100 * v['repeat:frame_pfa_jammer']['mean']:.1f} |")
(HERE / "TABLE_repeat.md").write_text("| method | cell | E_U | E_S | Pd@5%thr | Pfa@5%thr | Pfa nominal |\n|---|---|---:|---:|---:|---:|---:|\n"
                                      + "\n".join(rows) + "\n", encoding="utf-8")
print("saved fig10_structure_sweep and TABLE_repeat.md")
