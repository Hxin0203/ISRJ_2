"""WP6 Fig. 5 plotting (local, from FIG5_INPUT.npz). Usage: python wp6_fig5_plot.py [jt]
jt: 1 = repeat (main text, default), 0 = direct, 2 = cyclic (supplement).
Rows: DEV parents at P10/P50/P90 of the Hybrid - UNet-Tail damage difference (pre-registered rule).
(a) magnitude of the first jammed pulse over the target support; (b) pulse-compression magnitude in
dB above the reference (S+E) median noise level, target range bin dotted; (c-e) range-Doppler power
of the J+S+E frame within +/-120 range bins of the target, dB above the reference median.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper2.radar_eval import pulse_compress  # noqa: E402

HERE = ROOT / "evidence/outputs/paper2_baseline_suite_20261126/WP6"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5})
jt = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TYPES = ("direct", "repeat", "cyclic")
z = np.load(HERE / "FIG5_INPUT.npz")
picks = json.loads((HERE / "FIG5_INPUT.json").read_text(encoding="utf-8"))["picks"]
LINES = [("Observation", "Observation", "#b8c0c6", 0.6), ("Reference_SpE", r"Reference $S\!+\!E$", "#22313F", 1.0),
         ("Final", "ReCoVP-Hybrid", "#c0392b", 0.9), ("UNet-Tail", "UNet-Tail", "#3A8A8A", 0.8),
         ("RSDR-Net", "RSDR-Net", "#4F86A6", 0.8)]
RDS = [("Observation", "Observation"), ("Final", "ReCoVP-Hybrid"), ("UNet-Tail", "UNet-Tail"), ("RSDR-Net", "RSDR-Net")]
HALF = 120
fig, ax = plt.subplots(3, 6, figsize=(7.4, 5.0), constrained_layout=True,
                       gridspec_kw={"width_ratios": [1.35, 1.35, 1, 1, 1, 1]})
for row, (q, _) in enumerate(picks.items()):
    tgt = z["target"][row, jt]
    active = np.flatnonzero(np.abs(tgt) > 0); a, b = active[0], active[-1] + 1
    tbin = int(np.argmax(np.abs(pulse_compress(tgt[None])[0][0])))
    for key, lab, col, lw in LINES[:4]:
        ax[row, 0].plot(np.arange(a, b), np.abs(z[f"{key}__wave"][row, jt, a:b]), color=col, lw=lw, label=lab)
    ref = z["Reference_SpE__pc_db"][row, jt]; floor = float(np.median(ref))
    for key, lab, col, lw in LINES:
        ax[row, 1].plot(z[f"{key}__pc_db"][row, jt] - floor, color=col, lw=lw, label=lab)
    ax[row, 1].axvline(tbin, color="#22313F", lw=0.5, ls=":")
    ax[row, 1].set_xlim(max(0, tbin - 300), min(len(ref), tbin + 300)); ax[row, 1].set_ylim(-15, 45)
    rfloor = float(np.median(z["Reference_SpE__rd_db"][row, jt]))
    lo, hi = max(0, tbin - HALF), min(len(ref), tbin + HALF)
    for c, (key, lab) in enumerate(RDS, start=2):
        rd = z[f"{key}__rd_db"][row, jt][:, lo:hi] - rfloor
        im = ax[row, c].imshow(rd, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=40,
                               extent=(lo, hi, -0.5, rd.shape[0] - 0.5), interpolation="nearest")
        ax[row, c].axvline(tbin, color="white", lw=0.4, ls=":")
        ax[row, c].set_yticks([]); ax[row, c].set_xticks([tbin]); ax[row, c].set_xticklabels([str(tbin)], fontsize=7)
        if row == 0:
            ax[row, c].set_title(lab, fontsize=8)
    ax[row, 0].set_ylabel(f"{q}  (DEV {int(z['parent_id'][row]) % 10000})", fontsize=8)
    ax[row, 1].set_ylabel("dB above floor", fontsize=8)
ax[0, 0].set_title("(a) |waveform|, jammed pulse", loc="left", fontsize=8.5)
ax[0, 1].set_title("(b) pulse compression", loc="left", fontsize=8.5)
ax[0, 2].set_title("(c) Observation", fontsize=8)
ax[2, 0].set_xlabel("fast-time sample"); ax[2, 1].set_xlabel("range bin")
for c in range(2, 6):
    ax[2, c].set_xlabel("range bin", fontsize=7.5)
handles, labels = ax[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.36, -0.045))
fig.colorbar(im, ax=ax[:, 2:], shrink=0.55, label="range-Doppler power, dB above floor")
fig.suptitle(f"Jammer type: {TYPES[jt]}.  Rows: P10 / P50 / P90 of the Hybrid $-$ UNet-Tail target-damage difference",
             fontsize=8.5)
name = "fig5_typical" if jt == 1 else f"figS_typical_{TYPES[jt]}"
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"{name}.{ext}", dpi=300, bbox_inches="tight")
print("saved", name, "target bins", flush=True)
