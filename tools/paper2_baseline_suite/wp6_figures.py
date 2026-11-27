"""WP6 local step: Fig. 6 (stage deltas, DEV256, 5 seeds) and the Fig. 5 parent selection.

Inputs (pulled from server B, light): WP6/FIG6_INPUT.npz (from WP8 PARENT_VALUES) and
WP6/E1_SEEDAVG_PARENT_VALUES.npz. Rule (task book 02, WP6): per parent, five-seed mean
E_S(Hybrid) - three-seed mean E_S(UNet-Tail); choose the parents nearest the 10th/50th/90th
percentiles, ties to the smaller index. Seed for Fig. 5 = median DEV E_U seed (67).
Deviation: Fig. 6(c) plots per-parent amplitude change vs damage change, because the WP8
arrays carry no per-parent gain/shape decomposition.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[2] / "evidence/outputs/paper2_baseline_suite_20261126/WP6"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10})
SEEDS = (17, 29, 41, 53, 67)
z = np.load(HERE / "FIG6_INPUT.npz")
e1 = np.load(HERE / "E1_SEEDAVG_PARENT_VALUES.npz")
pid = z["parent_id"]
if "parent_id" in e1.files and not np.array_equal(e1["parent_id"], pid):
    raise SystemExit("E1 and WP8 parent order differ")


def avg(stage, key):
    return np.mean([z[f"s{s}__{stage}__{key}"] for s in SEEDS], axis=0)


s1u, fu = avg("stage1_K16_learned", "u_nmse"), avg("final_K16_learned", "u_nmse")
s1d, fd = avg("stage1_K16_learned", "damage"), avg("final_K16_learned", "damage")
s1a, fa = avg("stage1_K16_learned", "amplitude"), avg("final_K16_learned", "amplitude")
du, dd, da = fu - s1u, fd - s1d, fa - s1a

# Fig. 5 selection
diff = fd - e1["UNet-Tail__damage__seedavg"]
picks = {}
for q in (10, 50, 90):
    target = np.percentile(diff, q)
    order = np.lexsort((np.arange(len(diff)), np.abs(diff - target)))
    picks[q] = int(order[0])

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
for a, d, lab in ((ax[0], du, r"$\Delta E_U$ (Final $-$ Stage 1)"), (ax[1], dd, r"$\Delta E_S$ (Final $-$ Stage 1)")):
    lin = 0.01
    lo, hi = float(d.min()), float(d.max())
    neg = np.logspace(np.log10(lin), np.log10(-lo * 1.05), 14) if lo < -lin else np.array([])
    pos = np.logspace(np.log10(lin), np.log10(hi * 1.05), 6) if hi > lin else np.array([])
    edges = np.unique(np.concatenate((-neg[::-1], np.linspace(-lin, lin, 9), pos)))
    a.hist(d, bins=edges, color="#8FAFC4", edgecolor="#22313F", linewidth=0.4)
    a.set_xscale("symlog", linthresh=lin)
    ticks = [t for t in (-10, -1, -0.1, -0.01, 0, 0.01, 0.1) if lo * 1.1 <= t <= max(hi * 1.1, 0.01)]
    a.set_xticks(ticks)
    a.set_xticklabels([f"{t:g}" for t in ticks])
    a.set_xlim(lo * 1.15, max(hi * 1.3, 0.012))
    a.axvline(0, color="#22313F", lw=0.8)
    a.axvline(np.mean(d), color="#c0392b", lw=1.0, ls="-", label=f"mean {np.mean(d):+.3f}")
    a.axvline(np.median(d), color="#3A8A8A", lw=1.0, ls="--", label=f"median {np.median(d):+.3f}")
    a.set_xlabel(lab); a.set_ylabel("parents"); a.legend(frameon=False, fontsize=7.5)
ax[2].scatter(da, dd, s=7, color="#3A8A8A", alpha=0.7, edgecolor="none")
ax[2].set_yscale("symlog", linthresh=0.01)
ax[2].set_yticks([-10, -1, -0.1, -0.01, 0, 0.01, 0.1])
ax[2].set_yticklabels(["-10", "-1", "-0.1", "-0.01", "0", "0.01", "0.1"])
ax[2].axhline(0, color="#22313F", lw=0.6); ax[2].axvline(0, color="#22313F", lw=0.6)
ax[2].set_xlabel(r"$\Delta|g|$ (Final $-$ Stage 1)"); ax[2].set_ylabel(r"$\Delta E_S$")
for a, t in zip(ax, "abc"):
    a.set_title(f"({t})", loc="left", fontsize=10)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig6_stage_delta.{ext}", dpi=300)

frac = lambda x: float(np.mean(x))
stats = {"parents": int(len(pid)), "seeds": SEEDS,
         "dE_U": {"mean": float(du.mean()), "median": float(np.median(du)), "frac_improved": frac(du < 0)},
         "dE_S": {"mean": float(dd.mean()), "median": float(np.median(dd)), "p95": float(np.percentile(dd, 95)),
                  "frac_improved": frac(dd < 0)},
         "d_amplitude": {"mean": float(da.mean()), "median": float(np.median(da)), "frac_decreased": frac(da < 0)},
         "quadrants": {"damage_down_amp_down": frac((dd < 0) & (da < 0)), "damage_down_amp_up": frac((dd < 0) & (da >= 0)),
                       "damage_up_amp_down": frac((dd >= 0) & (da < 0)), "damage_up_amp_up": frac((dd >= 0) & (da >= 0))},
         "corr_dd_da": float(np.corrcoef(dd, da)[0, 1]),
         "fig5_rule": "per-parent 5-seed Hybrid E_S minus 3-seed UNet-Tail E_S; nearest to P10/P50/P90, ties -> smaller index",
         "fig5_seed": 67, "fig5_picks": {str(q): {"index": i, "parent_id": int(pid[i]), "diff": float(diff[i])}
                                         for q, i in picks.items()}}
(HERE / "FIG6_STATS.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps(stats, indent=1))
