"""Operating-point figure for the ReCoVP-Hybrid manuscript (read-only, CPU).

Reads reports/Hybrid_5seed_20261127/ANALYSIS.json (equal-PFA evaluation set,
2,048 calibration-disjoint parents) and plots, per method:
  (a) background modification E0 (log) vs target damage D_S = gain + shape error
  (b) target damage D_S vs Pd at the 2 % calibrated frame-PFA target.
Seed means are markers; horizontal/vertical bars span seed min-max.
Hybrid seeds are also drawn individually so the seed-67 E0 outlier is visible.
CVEDNet, GMM-OMP and Max-TF are off-scale (E0 ~ 1e-3..1, D_S 0.6..19) and are
listed in the caption instead of plotted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports/Hybrid_5seed_20261127/ANALYSIS.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evidence/outputs/paper2_baseline_suite_20261126/FIG_OPERATING_POINT"
OUT.mkdir(parents=True, exist_ok=True)

a = json.loads(SRC.read_text(encoding="utf-8"))
ep = a["equal_pfa"]
SKIP = {"CVEDNet", "GMM-OMP", "Max-TF", "BCS-oracle"}


def vals(d):
    v = d["values"]
    if isinstance(v, str):
        v = [float(x) for x in v.split()]
    return np.asarray(v, float)


rows = {}
for m, d in ep.items():
    if m in SKIP:
        continue
    g = d["diagnostics"]
    e0 = vals(g["e0_modification"]["seed_means"])
    ds = vals(g["gain_error"]["seed_means"]) + vals(g["orth_error"]["seed_means"])
    pd2 = vals(d["curves"]["0.02"]["pd"]) if "values" in d["curves"]["0.02"]["pd"] else np.array([d["curves"]["0.02"]["pd"]["mean"]])
    rows[m] = dict(e0=e0, ds=ds, pd2=pd2 * 100)


def style(m):
    if m == "ReCoVP-Hybrid":
        return dict(c="C3", mk="*", ms=16, z=5)
    if m == "ReCoVP-Stage1":
        return dict(c="C1", mk="D", ms=8, z=4)
    if m == "RSDR-Net":
        return dict(c="k", mk="s", ms=8, z=4)
    if m.startswith("UNet"):
        return dict(c="C0", mk="o", ms=6, z=3)
    return dict(c="C2", mk="^", ms=6, z=3)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for m, r in rows.items():
    s = style(m)
    for k, (xk, yk) in enumerate((("e0", "ds"), ("ds", "pd2"))):
        x, y = r[xk], r[yk]
        ax[k].errorbar(x.mean(), y.mean(),
                       xerr=[[x.mean() - x.min()], [x.max() - x.mean()]],
                       yerr=[[y.mean() - y.min()], [y.max() - y.mean()]],
                       fmt=s["mk"], color=s["c"], ms=s["ms"], zorder=s["z"],
                       elinewidth=0.8, capsize=2)
        if m in ("ReCoVP-Hybrid", "ReCoVP-Stage1", "RSDR-Net", "UNet-Tail",
                 "CorrectionNet-Tail", "CN-NoDamage", "UNet-Freq"):
            ax[k].annotate(m, (x.mean(), y.mean()), textcoords="offset points",
                           xytext=(6, 4), fontsize=8)
h = rows["ReCoVP-Hybrid"]
ax[0].scatter(h["e0"], h["ds"], s=14, color="C3", alpha=0.6, zorder=6)
ax[0].set_xscale("log")
ax[0].set_xlabel("Background modification $E_0$ (log)")
ax[0].set_ylabel("Target damage $D_S$")
ax[0].set_title("(a) Background vs target")
ax[1].set_xlabel("Target damage $D_S$")
ax[1].set_ylabel("$P_d$ at 2 % calibrated frame PFA (%)")
ax[1].set_title("(b) Target vs detection")
for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_operating_point.{ext}", dpi=200)

summary = {m: {k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}
               for k, v in r.items()} for m, r in rows.items()}
(OUT / "OPERATING_POINT.json").write_text(json.dumps({"source": str(SRC.relative_to(ROOT)), "methods": summary}, indent=1), encoding="utf-8")
print("wrote", OUT)
