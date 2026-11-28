"""Representative full-view waveform, pulse-compression and Doppler figures."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
MASTER = ROOT / "data/hybrid_condition_master_20261127/components/test_0_0.npz"
OUT = ROOT / "outputs/original_paper_replication_256_v1/R1_waveforms"
SEEDS = (67, 97, 53, 41, 109)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _restore(method: str, seed: int | None, states, jammer):
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    if method == "ReCoVP-Hybrid":
        model, binding = E.load(method, seed)
        return E.restore(model, "hybrid", states)["main"], binding
    if method == "CVEDNet":
        from tools.paper2_baseline_suite import external_suite as X
        model, adapter, binding = X.load_model()
        return X.restore(model, adapter, states), binding
    from tools.paper2_baseline_suite import classical_suite as C
    name = {"Max-TF": "max_tf_isrj1", "GMM-OMP": "gmm_omp"}[method]
    return C.restore(name, states, jammer), {"config": name}


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("R1 runs only on server B")
    from tools.server_gpu_admission import require_visible_gpu
    admission = require_visible_gpu()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import doppler_process, pulse_compress
    manifest = json.loads((MASTER.parents[1] / "DATA_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if sha(MASTER) != manifest["component_sha256"]:
        raise RuntimeError("R1 master component SHA mismatch")
    selection = json.loads((ROOT / "reports/Original_paper_replication_256/SELECTED_SEEDS.json").read_text(encoding="utf-8-sig"))
    if tuple(selection["selected_seeds"]) != SEEDS or selection["epoch"] != 256:
        raise RuntimeError("R1 frozen seed selection differs")
    with np.load(MASTER, allow_pickle=False) as z:
        s = z["s"][0:1].copy()
        j = z["j"][0:1].copy() * np.float32(10 ** (20 / 20))
        e = z["epsilon"][0:1].copy()
        parent_id = z["parent_id"][0:1].copy()
        fd = z["doppler_hz"][0:1].copy()
    states = states_from_components(s, j, e)
    # Components use [parent, mechanism, pulse, fast time].
    selected_state = states[0, :, 1]
    selected_jammer = j[:, 1]
    pulse = int(np.argmax(np.sum(np.abs(selected_jammer[0]) ** 2, axis=-1) > 0))
    records = {}
    for method in ("Max-TF", "GMM-OMP", "CVEDNet"):
        u, binding = _restore(method, None, states, j)
        records[method] = {"waveform": u[0, :, 1].astype(np.complex64), "binding": binding}
    for seed in SEEDS:
        u, binding = _restore("ReCoVP-Hybrid", seed, states, j)
        records[f"ReCoVP-Hybrid_seed_{seed}"] = {"waveform": u[0, :, 1].astype(np.complex64), "binding": binding}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "R1_ISRRJ_full_view.npz", parent_id=parent_id, s=s[0], j=j[0], epsilon=e[0],
                        observed=selected_state, **{key: value["waveform"] for key, value in records.items()})
    # One image per method keeps every output independently inspectable and
    # preserves full time, range and Doppler axes without an inset.
    for label, record in records.items():
        wave = record["waveform"]
        pc, ranges = pulse_compress(wave)
        rd, doppler = doppler_process(pc)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
        axes[0].plot(np.arange(1800), np.abs(selected_state[3, pulse]), color="#7f8c8d", lw=.7, label="before")
        axes[0].plot(np.arange(1800), np.abs(wave[3, pulse]), color="#0072B2", lw=.7, label="after")
        axes[0].set_xlabel("Fast-time sample")
        axes[0].set_ylabel("Magnitude")
        axes[0].legend(frameon=False, fontsize=7)
        axes[1].plot(ranges, 20 * np.log10(np.maximum(np.abs(pc[3]).mean(axis=0), 1e-12)), color="#0072B2", lw=.8)
        axes[1].set_xlabel("Range lag (sample)")
        axes[1].set_ylabel("Pulse-compressed magnitude (dB)")
        extent = [ranges[0], ranges[-1], doppler[0] / 1e3, doppler[-1] / 1e3]
        axes[2].imshow(20 * np.log10(np.maximum(np.abs(rd[3]), 1e-12)), origin="lower", aspect="auto", extent=extent, cmap="magma")
        axes[2].set_xlabel("Range lag (sample)")
        axes[2].set_ylabel("Doppler (kHz)")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(OUT / f"{label}_full_view.{ext}", dpi=600 if ext == "png" else None,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
    write_json(OUT / "R1_PRELIMINARY.json", {
        "status": "PRELIMINARY_OUTPUTS_AWAITING_FULL_R1_VALIDATION", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_id": int(parent_id[0]), "jammer_variant": "ISRRJ", "snr_db": 0, "jsr_db": 20,
        "first_jammed_pulse_index": pulse,
        "pending": ["reference curves and full Doppler spectra", "three-mechanism extension",
                    "16-parent coherence diagnostics", "same-data epoch256 CVED-Net", "rendered figure QA"],
        "selected_hybrid_seeds": list(SEEDS), "methods": list(records),
        "master_sha256": sha(MASTER), "gpu_admission": admission,
        "outputs": "R1_ISRRJ_full_view.npz and one full-view PNG/PDF/SVG triplet per method",
    })
    print("R1_PRELIMINARY", flush=True)


if __name__ == "__main__":
    from tools.server_gpu_admission import NoGpuAvailable, choose_gpu
    while True:
        try:
            admission = choose_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(admission["physical_gpu_index"])
            write_json(OUT / "DISPATCH_ADMISSION.json", admission)
            main()
            break
        except NoGpuAvailable as exc:
            write_json(OUT / "RESOURCE_WAIT.json", {"state": "WAITING_RESOURCES", "reason": str(exc),
                                                   "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            time.sleep(30)
