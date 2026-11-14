"""Generate review plots from completed 1103 TEST receipts.

The script only reads light receipts and regenerates the fixed typical scene from
the frozen scene namespace. Missing conditions remain marked partial; no values
are imputed and no TEST arrays are written to the repository.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper2.evaluation_1103_v4 import restore_model
from paper2.io import ROOT, read, sha
from paper2.network import CorrectionNet
from paper2.radar_eval import doppler_process, pulse_compress
from paper2.scenes_1103 import scene

RUNS = (ROOT / "outputs/paper2_1103_test_20261114", ROOT / "outputs/paper2_1103_test_classical_20261114")
FREEZE = ROOT / "configs/paper2_1103_scan_freeze.json"
OUT = ROOT / "outputs/plots_1103_20261114"


def receipts():
    rows = []
    for run in RUNS:
        for p in sorted(run.glob("*/*/result.json")):
            r = read(p)
            if r.get("status") != "COMPLETE":
                continue
            identity = r["identity"]
            row = dict(run=run.name, suite=identity["suite"], index=identity["index"], method=identity["method"])
            row.update(identity["condition"])
            row.update({k: v["mean"] for k, v in r["metrics"].items()})
            rows.append(row)
    return rows


def save_table(rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "completed_receipts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def plot_grid(rows, suite, metric, title, filename, x="snr_db", y="jsr_db"):
    sub = [r for r in rows if r["suite"] == suite and metric in r]
    if not sub:
        return
    # Facet every non-axis intervention so no distinct conditions overwrite.
    facets = [k for k in ("method", "jam_type", "snr_db", "jsr_db", "width", "repeats")
              if k not in (x,y) and any(k in r for r in sub)]
    groups = sorted({tuple(r.get(k) for k in facets) for r in sub}, key=str)
    cols = min(3, len(groups))
    fig, axes = plt.subplots((len(groups)+cols-1)//cols, cols,
                             figsize=(5*cols,4*((len(groups)+cols-1)//cols)), squeeze=False)
    vmin, vmax = min(r[metric] for r in sub), max(r[metric] for r in sub)
    for ax, group in zip(axes.flat, groups):
        vals = [r for r in sub if tuple(r.get(k) for k in facets)==group]
        xs, ys = sorted({r.get(x) for r in vals}), sorted({r.get(y) for r in vals})
        a = np.full((len(ys), len(xs)), np.nan)
        for r in vals:
            a[ys.index(r.get(y)), xs.index(r.get(x))] = r[metric]
        im = ax.imshow(a, aspect="auto", origin="lower", interpolation="nearest",vmin=vmin,vmax=vmax)
        ax.set_xticks(range(len(xs)), [str(v) for v in xs], rotation=60)
        ax.set_yticks(range(len(ys)), [str(v) for v in ys])
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(", ".join(str(v) for v in group))
        fig.colorbar(im, ax=ax, shrink=.8)
    for ax in list(axes.flat)[len(groups):]: ax.set_visible(False)
    fig.suptitle(f"{title} (completed cells only)")
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=180); plt.close(fig)


def coherent_table(rows):
    sub = [r for r in rows if r["suite"] == "coherent"]
    if not sub: return
    keys = list(dict.fromkeys(k for r in sub for k in r))
    with (OUT / "coherent_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(sub)


def typical_scene():
    import torch
    torch.set_num_threads(2)
    freeze = read(FREEZE)
    checkpoint = ROOT / freeze["selected_checkpoint"]
    if sha(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint SHA mismatch")
    s, j, e, meta = scene("test", "coherent", 0, jam_type="repeat", snr_db=0, jsr_db=20)
    obs = np.stack((e, s + e, j + e, s + j + e), axis=0)[None]
    model = CorrectionNet().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state"])
    restored = restore_model(model, obs, "cpu")[0]
    before, lags = pulse_compress(obs[0,3])
    after, _ = pulse_compress(restored)
    rd_before, fd = doppler_process(before)
    rd_after, _ = doppler_process(after[3])
    from paper2.radar_eval import target_peak_range
    target_range = int(target_peak_range(s))
    jammer_range = int(np.argmax(np.mean(np.abs(before)**2, axis=0)))
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes[0, 0].plot(np.abs(obs[0,3,0])); axes[0, 0].set_title("Typical parent 0: |S+J+E|")
    axes[0, 0].set_xlabel("sample"); axes[0, 0].set_ylabel("amplitude")
    panorama = 10*np.log10(np.mean(np.abs(before)**2, axis=0)+1e-18)
    restored_pc = 10*np.log10(np.mean(np.abs(after[3])**2, axis=0)+1e-18)
    axes[0, 1].plot(lags, panorama, label="input")
    axes[0, 1].plot(lags, restored_pc, label="CorrectionNet")
    axes[0, 1].set_title("Pulse compression panorama"); axes[0, 1].legend()
    lo, hi = max(0,target_range-20), min(len(lags),target_range+21)
    axes[1, 0].plot(lags[lo:hi], panorama[lo:hi], label="input")
    axes[1, 0].plot(lags[lo:hi], restored_pc[lo:hi], label="CorrectionNet")
    axes[1, 0].axvline(target_range, color="k", linestyle=":", label="target-only peak")
    axes[1, 0].set_title("Target-range detail (±20 cells)"); axes[1, 0].legend()
    slow = np.arange(8)*1/30000
    for pulse_data, label in ((before[:,target_range],"target range"),
                              (before[:,jammer_range],"strongest input range")):
        axes[1, 1].plot(slow*1e6,np.unwrap(np.angle(pulse_data)),marker="o",label=label)
    axes[1, 1].set_title("Slow-time phase at selected range cells"); axes[1, 1].set_xlabel("slow time (µs)"); axes[1, 1].legend()
    axes[2, 0].imshow(10*np.log10(np.abs(rd_before)**2+1e-18), aspect="auto", origin="lower", extent=[lags[0], lags[-1], fd[0], fd[-1]])
    axes[2, 0].set_title("Input range-Doppler"); axes[2, 0].set_xlabel("lag"); axes[2, 0].set_ylabel("Hz")
    axes[2, 1].imshow(10*np.log10(np.abs(rd_after)**2+1e-18), aspect="auto", origin="lower", extent=[lags[0], lags[-1], fd[0], fd[-1]])
    axes[2, 1].set_title("CorrectionNet range-Doppler"); axes[2, 1].set_xlabel("lag"); axes[2, 1].set_ylabel("Hz")
    fig.suptitle("Fixed typical scene: parent 0, SNR 0 dB, JSR 20 dB")
    fig.tight_layout(); fig.savefig(OUT / "typical_parent0_snr0_jsr20.png", dpi=180); plt.close(fig)
    from paper2.evaluation_1103_v4 import coherent_metrics
    coherent = coherent_metrics(s[None], after[3][None], fft_points=4096)
    with (OUT / "typical_coherent_metrics.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f);w.writerow(["parent","phase_rmse_deg","coherent_power_db","peak_error_hz"])
        w.writerow([0,float(np.mean(coherent["phase_rmse_deg"])),float(np.mean(coherent["coherent_power_db"])),float(np.mean(coherent["peak_error_hz"]))])
    (OUT / "typical_scene_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = receipts(); save_table(rows); coherent_table(rows)
    plot_grid(rows, "main", "u_nmse", "Main-grid restored-waveform NMSE", "main_u_nmse.png")
    plot_grid(rows, "main", "sjrif_db", "Main-grid SJRIF (dB)", "main_sjrif_db.png")
    plot_grid(rows, "main", "pd", "Main-grid detection probability", "main_pd.png")
    plot_grid(rows, "pd", "pd", "Pd scan", "pd_scan.png")
    plot_grid(rows, "retention", "no_jammer_pooled_nmse", "No-jammer retention error", "retention_e0.png", x="snr_db", y="jam_type")
    plot_grid(rows, "sliceA", "u_nmse", "Slice-A width sensitivity", "sliceA_u_nmse.png", x="width", y="jsr_db")
    plot_grid(rows, "sliceB", "u_nmse", "Slice-B width sensitivity", "sliceB_u_nmse.png", x="width", y="repeats")
    typical_scene()
    print(json.dumps({"completed_receipts": len(rows), "output": str(OUT), "partial": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
