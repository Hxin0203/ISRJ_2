"""Fig. 7 (equal-Pfa Pfa-Pd curves) and E10 (residual coherence) from existing E3/E6/E7 outputs.

CPU only, read-only on all inputs. Writes PNG/PDF figures and a light JSON with the
plotted numbers into outputs/paper2_baseline_suite_20261126/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUITE = Path("D:/code/Xin/ISRJ_222/outputs/paper2_baseline_suite_20261126")
OUT = SUITE / "figures"
FOLDER_LABELS = {"max_tf_isrj1": "Max-TF", "gmm_omp": "GMM-OMP", "cvednet": "CVEDNet", "rsdr_net": "RSDR-Net"}
SHOW = [("ReCoVP-Hybrid", "#c0392b", "-", "o"), ("ReCoVP-Stage1", "#e59866", "--", "o"),
        ("CorrectionNet", "#2e86c1", "-", "s"), ("CN-NoDamage", "#85c1e9", "--", "s"),
        ("UNet", "#239b56", "-", "^"), ("UNet-Tail", "#82e0aa", "--", "^"),
        ("RSDR-Net", "#7d3c98", "-", "D"), ("CVEDNet", "#a6acaf", "-", "v"),
        ("Max-TF", "#b7950b", ":", "x"), ("GMM-OMP", "#5d6d7e", ":", "+")]


def sources():
    out = {}
    for row in sorted((SUITE / "E3").glob("*/seed_*/ROW.json")):
        if (row.parent / "SCORES.npz").is_file():
            out.setdefault(row.parent.parent.name, []).append(row.parent)
    for sub in ("E6", "E7"):
        for row in sorted((SUITE / sub / "equal_pfa").glob("*/ROW.json")):
            if row.parent.name in FOLDER_LABELS:
                out.setdefault(FOLDER_LABELS[row.parent.name], []).append(row.parent)
    return out


def curves(src):
    res = {}
    for fam, folders in src.items():
        rows = [json.loads((f / "ROW.json").read_text(encoding="utf-8"))["curve"] for f in folders]
        grid = [r["target_frame_pfa"] for r in rows[0]]
        pfa = np.mean([[r["eval_frame_pfa_jammer"] for r in rr] for rr in rows], axis=0)
        pd = np.array([[r["eval_pd"] for r in rr] for rr in rows])
        res[fam] = {"seeds": len(rows), "target": grid, "pfa": pfa.tolist(), "pd": pd.mean(0).tolist(),
                    "pd_seed_sd": (pd.std(0, ddof=1).tolist() if len(rows) > 1 else None)}
    return res


def coherence(src):
    res = {}
    for fam, folders in src.items():
        peak, conc = [], []
        for f in folders:
            z = np.load(f / "SCORES.npz")
            peak.append(z["eval_resid_peak_noise_units"].reshape(-1))
            conc.append(z["eval_resid_concentration"].reshape(-1))
        peak, conc = np.concatenate(peak), np.concatenate(conc)
        res[fam] = {"frames": int(peak.size),
                    "peak_db_quantiles": {str(q): float(10 * np.log10(max(np.quantile(peak, q / 100), 1e-12)))
                                          for q in (10, 50, 90, 99)},
                    "fraction_peak_above_13dB": float((peak > 10 ** 1.3).mean()),
                    "concentration_quantiles": {str(q): float(np.quantile(conc, q / 100)) for q in (10, 50, 90, 99)},
                    "_peak": peak}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = sources()
    cv = curves(src)
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
    for fam, color, ls, mk in SHOW:
        if fam not in cv:
            continue
        c = cv[fam]
        ax.plot(np.array(c["pfa"]) * 100, np.array(c["pd"]) * 100, color=color, ls=ls, marker=mk, ms=3, lw=1.1,
                label=f"{fam} (×{c['seeds']})" if c["seeds"] > 1 else fam)
    ax.set_xscale("log")
    ax.set_xlabel("Measured jammed frame $P_{fa}$ (%)")
    ax.set_ylabel("$P_d$ (%)")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig7_pfa_pd_curves.{ext}")
    plt.close(fig)

    co = coherence(src)
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    for fam, color, ls, _ in SHOW:
        if fam not in co or fam == "CVEDNet":
            continue
        x = np.sort(10 * np.log10(np.maximum(co[fam]["_peak"], 1e-6)))
        ax.plot(x, 1 - np.arange(1, x.size + 1) / x.size, color=color, ls=ls, lw=1.1, label=fam)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xlim(-10, 40)
    ax.set_xlabel("Residual RD peak / mean noise RD power (dB)")
    ax.set_ylabel("Fraction of jammed frames exceeding")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=5.5, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_residual_coherence_ccdf.{ext}")
    plt.close(fig)
    for v in co.values():
        v.pop("_peak")
    (OUT / "FIGURE_DATA.json").write_text(json.dumps({"fig7_curves": cv, "residual_coherence": co,
        "note": "CVEDNet omitted from the coherence plot: it removes receiver noise, so its residual u-e is dominated by -e."},
        indent=1), encoding="utf-8")
    for fam in sorted(co):
        print(fam, co[fam]["frames"], co[fam]["peak_db_quantiles"], round(co[fam]["fraction_peak_above_13dB"], 4))
    print("curves", {k: (v["seeds"], [round(p, 3) for p in v["pd"]]) for k, v in cv.items() if k in dict((s[0], 1) for s in SHOW)})


if __name__ == "__main__":
    main()
