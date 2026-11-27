"""Fixed, paired SNR/JSR benchmark. Evaluation only, with resumable cell outputs.

The master stores actual complex components at SNR=JSR=0 dB. Registered
amplitude transforms define every condition without duplicating the components.
Native-generator equivalence is checked before any model is evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hybrid_condition_master_20261127"
OUT = ROOT / "outputs/hybrid_condition_benchmark_20261127"
REPORT = ROOT / "reports/Hybrid_condition_benchmark_20261127"
SEED = 2026112901
MASTER_PARENTS = 512
EVAL_PARENTS = 128
SNR_GRID = [-30, -28, -26, -25, -24, -22, -20, -18, -16, -15, -14, -12, -10, -8, -6, -5, -4, -2, 0, 5, 10]
JSR_GRID = [-10, 0, 10, 20, 30, 35, 40]
PRIMARY_SNR = [-30, -28, -26, -25, -24, -22, -20, -18, -15, -10, -5, 0]
PRIMARY_JSR = [0, 10, 20, 30]
JOBS = [("ReCoVP-Hybrid", s) for s in (17, 29, 41, 53, 67)] + [("RSDR-Net", None)] + [("UNet-Tail", s) for s in (17, 29, 41)] + [("CorrectionNet-Tail", s) for s in (17, 29, 41)]


def sha(path):
    from tools.paper2_baseline_suite.equal_pfa_suite import sha as digest
    return digest(Path(path))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cell_name(snr, jsr):
    return f"snr_{snr:g}_jsr_{jsr:g}"


def regime(snr):
    if snr < -20:
        return "extreme"
    if snr <= 0:
        return "ordinary"
    return "extension"


def prepare():
    from paper2.data_1103 import generate
    registration = DATA / "REGISTRATION.json"
    if not registration.exists():
        write_json(registration, {
            "registered_utc": now(), "version": 1, "seed": SEED,
            "master_parents": MASTER_PARENTS, "primary_parent_indices": [0, EVAL_PARENTS],
            "reserve_parent_indices": [EVAL_PARENTS, MASTER_PARENTS],
            "snr_grid_db": SNR_GRID, "jsr_grid_db": JSR_GRID,
            "primary_snr_db": PRIMARY_SNR, "primary_jsr_db": PRIMARY_JSR,
            "regimes": {"extreme": "-30 <= SNR < -20", "ordinary": "-20 <= SNR <= 0", "extension": "SNR > 0"},
            "components": "Frozen paper2.data_1103 generator, actual saved S,J,E at SNR=JSR=0 dB",
            "transform": {"s": "S0", "epsilon": "E0 * 10**(-snr_db/20)", "j": "J0 * 10**(jsr_db/20)"},
            "states": ["E", "S+E", "J+E", "S+J+E"],
            "jammer_types": ["direct", "repeat", "cyclic"],
            "inference_jobs": JOBS, "checkpoint_policy": "Five completed Hybrid epoch-256 checkpoints; three epoch-256 seeds per convolutional baseline; frozen first-paper RSDR F-R7",
            "threshold_policy": "Reuse each checkpoint's independent 1024-parent calibration at frame-PFA targets 1%,5%,10%; no fitting on master",
            "primary_summary": "Equal weight per registered SNR x JSR cell within each regime, types averaged within physical parent",
            "uncertainty": "Training-seed SD separately; paired parent bootstrap resamples physical IDs jointly across all conditions and types, conditional on checkpoints and frozen calibration",
            "purpose": "Synthetic development benchmark for condition characterization and future registered subsets",
            "independent_confirmation": "Remains separate; this master is not an independent final confirmation set",
            "source_sha256": {p: sha(ROOT / p) for p in ["paper2/data_1103.py", "paper2/vendor/generate_v2.py", "paper2/evaluation_1103_v4.py", "paper2/radar_eval.py"]},
        })
    reg = json.loads(registration.read_text(encoding="utf-8"))
    for p, digest in reg["source_sha256"].items():
        if sha(ROOT / p) != digest:
            raise RuntimeError(f"registered source changed: {p}")
    path, receipt = generate("test", count=MASTER_PARENTS, seed=SEED, snr_db=0, jsr_db=0, out_dir=DATA / "components")
    with np.load(path, allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    for key in ("s", "j", "epsilon"):
        if base[key].shape != (MASTER_PARENTS, 3, 8, 1800) or not np.isfinite(base[key]).all():
            raise RuntimeError(f"invalid master component {key}")
    # Direct generation at two distant conditions checks the scale construction.
    equivalence = []
    for snr, jsr in [(-30, 40), (-20, -10)]:
        native, _ = generate("engineering", count=4, seed=SEED, snr_db=snr, jsr_db=jsr, out_dir=DATA / "validation")
        with np.load(native, allow_pickle=False) as z:
            errors = {}
            for key, scale in [("s", 1), ("j", 10 ** (jsr / 20)), ("epsilon", 10 ** (-snr / 20))]:
                scaled = base[key][:4] * np.float32(scale)
                errors[key] = float(np.linalg.norm((scaled - z[key]).ravel()) / max(np.linalg.norm(z[key].ravel()), 1e-30))
            if max(errors.values()) > 3e-7:
                raise RuntimeError(f"native-generator scale validation failed {errors}")
            equivalence.append({"snr_db": snr, "jsr_db": jsr, "relative_errors": errors, "native_sha256": sha(native)})
    protected = [ROOT / "data/paper2_1103/train.npz", ROOT / "data/paper2_1103/dev.npz"]
    protected += sorted((ROOT / "data").glob("**/equal_pfa/*/test.npz"))
    protected += sorted((ROOT / "data").glob("**/confirmation/test.npz"))
    protected += sorted((ROOT / "data/paper2_baseline_suite_20261126/sweep").glob("**/*.npz"))
    overlaps = {}
    for p in dict.fromkeys(protected):
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                if "parent_id" in z:
                    overlaps[str(p.relative_to(ROOT))] = int(np.intersect1d(base["parent_id"], z["parent_id"]).size)
    if not overlaps or any(overlaps.values()):
        raise RuntimeError(f"protected ID overlap check failed {overlaps}")
    cells = [{"name": cell_name(s, j), "snr_db": s, "jsr_db": j, "regime": regime(s)} for s in SNR_GRID for j in JSR_GRID]
    manifest = {
        "created_utc": now(), "registration_sha256": sha(registration), "component_path": str(path), "component_sha256": sha(path),
        "component_receipt_sha256": sha(receipt), "parents": MASTER_PARENTS, "cells": cells,
        "logical_condition_parents": MASTER_PARENTS * len(cells), "physical_independent_parents": MASTER_PARENTS,
        "jammed_pulse_count_histogram": {str(int(v)): int(np.sum(base["jammed_pulse_count"] == v)) for v in np.unique(base["jammed_pulse_count"])},
        "native_generator_equivalence": equivalence, "protected_id_overlap": overlaps,
        "pairing": "Same physical parent at every SNR/JSR; target and normalized noise identical across types",
        "storage": "Factorized actual complex components plus deterministic registered scale transforms; observations materialized when selected",
        "bytes_on_disk_components": path.stat().st_size,
    }
    write_json(DATA / "DATA_MANIFEST.json", manifest)
    write_json(REPORT / "DATA_MANIFEST.json", manifest)
    write_json(REPORT / "REGISTRATION.json", reg)
    print("MASTER_READY", path, manifest["logical_condition_parents"], flush=True)


def load_components(count=EVAL_PARENTS):
    manifest = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    p = Path(manifest["component_path"])
    if sha(p) != manifest["component_sha256"]:
        raise RuntimeError("master SHA mismatch")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k][:count] for k in z.files if k != "jam_types"}, manifest


def select(snr, jsr, start=0, stop=EVAL_PARENTS):
    """Materialize a registered subset from the saved master; never generate anew."""
    if snr not in SNR_GRID or jsr not in JSR_GRID or not 0 <= start < stop <= MASTER_PARENTS:
        raise ValueError("selection outside registered master")
    b, _ = load_components(stop)
    return {"s": b["s"][start:stop], "j": b["j"][start:stop] * np.float32(10 ** (jsr / 20)),
            "epsilon": b["epsilon"][start:stop] * np.float32(10 ** (-snr / 20)),
            "parent_id": b["parent_id"][start:stop], "doppler_hz": b["doppler_hz"][start:stop]}


def score(s, j, e, u, fd, threshold_rows):
    """Reuse frozen energy metrics; compute matched-filter/CFAR statistics once.

    This combined scorer is checked against frozen radar_metrics and frame_scores
    on the first chunk of every checkpoint, before persisting any cell.
    """
    from paper2.evaluation_1103_v4 import group_metrics, nearest_doppler_bin, states_from_components
    from paper2.radar_eval import pulse_compress, target_peak_range, doppler_process, ca_cfar, sjrif, mainlobe_guard
    n = len(s)
    flat_s, flat_j, flat_e = [x.reshape(n * 3, 8, 1800) for x in (s, j, e)]
    flat_u = np.moveaxis(u, 2, 1).reshape(n * 3, 4, 8, 1800)
    m = {k: np.asarray(v).reshape(n, 3) for k, v in group_metrics(flat_s, flat_j, flat_e, flat_u).items()}
    pc, _ = pulse_compress(u.astype(np.complex128))
    before, _ = pulse_compress(s.astype(np.complex128) + j.astype(np.complex128) + e.astype(np.complex128))
    peaks = target_peak_range(s[:, 0])
    names = ["pd", "pd_no_jammer", "frame_pfa_jammer", "frame_pfa_noise", "cell_pfa_jammer", "cell_pfa_noise", "sjrif_db", "false_peak_suppression_db", "score_h0", "score_target"]
    for pfa in threshold_rows:
        names += [f"pd_eq_{pfa}", f"pfa_eq_{pfa}"]
    m.update({key: np.empty((n, 3), dtype=np.float64) for key in names})
    for p in range(n):
        for jt in range(3):
            m_sjri = sjrif(before[p, jt], pc[p, 3, jt], int(peaks[p]), mainlobe_guard=mainlobe_guard())
            for key in ("sjrif_db", "false_peak_suppression_db"):
                m[key][p, jt] = m_sjri[key]
            for st in range(4):
                rd, bins = doppler_process(pc[p, st, jt])
                power = np.abs(rd) ** 2
                detections, threshold, valid, nr, alpha = ca_cfar(power)
                if nr != 436 or valid.sum() != 9032:
                    raise RuntimeError("CFAR geometry changed")
                ratio = np.divide(power, threshold / alpha, out=np.zeros_like(power), where=valid & (threshold > 0))
                if st in (0, 2):
                    suffix = "noise" if st == 0 else "jammer"
                    m[f"frame_pfa_{suffix}"][p, jt] = (detections & valid).any()
                    m[f"cell_pfa_{suffix}"][p, jt] = (detections & valid).sum() / valid.sum()
                    if st == 2:
                        m["score_h0"][p, jt] = ratio[valid].max()
                else:
                    db = nearest_doppler_bin(fd[p], bins)
                    tr = int(peaks[p])
                    if not valid[db, tr]:
                        raise RuntimeError("target outside CFAR support")
                    m["pd" if st == 3 else "pd_no_jammer"][p, jt] = detections[db, max(0, tr - 1):tr + 2].any()
                    if st == 3:
                        m["score_target"][p, jt] = ratio[db, max(0, tr - 1):tr + 2].max()
            for pfa, threshold in threshold_rows.items():
                m[f"pd_eq_{pfa}"][p, jt] = m["score_target"][p, jt] > threshold
                m[f"pfa_eq_{pfa}"][p, jt] = m["score_h0"][p, jt] > threshold
    if not all(np.isfinite(a).all() for a in m.values()):
        raise RuntimeError("nonfinite metric")
    return m


def run(method, seed):
    import torch
    from tools.paper2_baseline_suite import equal_pfa_suite as E, wp5b_suite as W
    from paper2.evaluation_1103_v4 import states_from_components
    E.ROOT, E.OUTPUTS = ROOT, ROOT / "outputs"
    W.SUITE = ROOT / "outputs/paper2_baseline_suite_20261126"
    E.configure_precision()
    torch.set_num_threads(4)
    b, manifest = load_components()
    tag = f"seed_{seed}" if seed is not None else "single"
    folder = OUT / method / tag
    if (folder / "DONE.json").exists():
        print("already complete", method, seed, flush=True)
        return
    rowpath = W.row_folder(method, seed) / "ROW.json"
    row = json.loads(rowpath.read_text(encoding="utf-8"))
    thresholds = {str(p): next(float(c["threshold_ratio"]) for c in row["curve"] if abs(c["target_frame_pfa"] - p) < 1e-12) for p in (.01, .05, .1)}
    if method == "RSDR-Net":
        from tools.paper2_baseline_suite.rsdr_suite import load
        model, binding = load()
    else:
        model, binding = E.load(method, seed)
        if row.get("checkpoint_sha256") != binding["checkpoint_sha256"]:
            raise RuntimeError("calibration/checkpoint identity mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "BINDING.json", {"model": binding, "threshold_row_sha256": sha(rowpath), "thresholds": thresholds,
        "master_sha256": manifest["component_sha256"], "registration_sha256": manifest["registration_sha256"],
        "host": platform.node(), "gpu": torch.cuda.get_device_name(0), "policy": "Local evaluation authorized 2026-11-27 when server training GPUs busy", "created_utc": now()})
    started = time.perf_counter()
    validated = False
    for snr in PRIMARY_SNR:
        e = b["epsilon"] * np.float32(10 ** (-snr / 20))
        nojam_cache = {}
        for jsr in PRIMARY_JSR:
            target = folder / f"{cell_name(snr, jsr)}.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as z:
                    if not np.array_equal(z["parent_id"], b["parent_id"]) or not all(np.isfinite(z[k]).all() for k in z.files):
                        raise RuntimeError(f"invalid prior cell {target}")
                continue
            j = b["j"] * np.float32(10 ** (jsr / 20))
            parts = []
            for lo in range(0, EVAL_PARENTS, 8):
                hi = min(lo + 8, EVAL_PARENTS)
                s0, j0, e0 = b["s"][lo:hi], j[lo:hi], e[lo:hi]
                st = states_from_components(s0, j0, e0)
                u = np.empty_like(st)
                if lo not in nojam_cache:
                    if method == "RSDR-Net":
                        nojam_cache[lo] = model.restore(np.ascontiguousarray(st[:, :, 0]), state_indexes=[0, 1])
                    else:
                        nojam_cache[lo] = E.restore(model, E.FAMILIES[method][1], st[:, :2, 0])["main"]
                u[:, :2] = nojam_cache[lo][:, :, None]
                if method == "RSDR-Net":
                    for jt in range(3):
                        u[:, 2:, jt] = model.restore(np.ascontiguousarray(st[:, :, jt]), state_indexes=[2, 3])
                else:
                    u[:, 2:] = E.restore(model, E.FAMILIES[method][1], st[:, 2:])["main"]
                m = score(s0, j0, e0, u, b["doppler_hz"][lo:hi], thresholds)
                if not validated:
                    reference = W.dev_metrics_by_type(s0, j0, e0, u, b["doppler_hz"][lo:hi])
                    for key, value in reference.items():
                        if not np.allclose(value, m[key], rtol=2e-6, atol=2e-9):
                            raise RuntimeError(f"combined metric parity failed {key}")
                    from paper2.radar_eval import target_peak_range
                    scores = E.frame_scores(u, s0, e0, b["doppler_hz"][lo:hi], target_peak_range(s0[:, 0]), True)
                    for a, c in [("h0", "score_h0"), ("target", "score_target")]:
                        if not np.allclose(scores[a], m[c], rtol=2e-6, atol=2e-9):
                            raise RuntimeError(f"frame score parity failed {a}")
                    if method == "RSDR-Net":
                        from tools.paper2_baseline_suite.rsdr_suite import restore
                        full = restore(model, st[:1])
                    else:
                        full = E.restore(model, E.FAMILIES[method][1], st[:1])["main"]
                    if not np.allclose(full, u[:1], rtol=2e-5, atol=3e-5):
                        raise RuntimeError("cached state restoration parity failed")
                    write_json(folder / "PARITY.json", {"passed": True, "frozen_metric_keys": list(reference), "frame_scores": ["h0", "target"], "cached_state_restoration": True, "at": now()})
                    validated = True
                parts.append(m)
            merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            tmp = target.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, parent_id=b["parent_id"], **merged)
            os.replace(tmp, target)
            print(method, seed, snr, jsr, f"{time.perf_counter()-started:.1f}s", flush=True)
    write_json(folder / "DONE.json", {"finished_utc": now(), "method": method, "seed": seed, "cells": len(PRIMARY_SNR) * len(PRIMARY_JSR),
        "parents_per_cell": EVAL_PARENTS, "seconds": time.perf_counter() - started, "runner_sha256": sha(Path(__file__)),
        "cell_sha256": {p.name: sha(p) for p in sorted(folder.glob("snr_*.npz"))}})


def queue():
    for method, seed in JOBS:
        args = [os.sys.executable, "-m", "tools.hybrid_condition_benchmark", "run", "--method", method]
        if seed is not None:
            args += ["--seed", str(seed)]
        result = subprocess.run(args, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"evaluation failed {method} {seed} exit {result.returncode}")
    write_json(OUT / "ALL_COMPLETE.json", {"finished_utc": now(), "jobs": JOBS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "queue"])
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "queue":
        queue()
    else:
        run(args.method, args.seed)
