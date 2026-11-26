"""E6: classical baselines on full DEV256 and on the equal-Pfa sets (server CPU only).

Configurations are the historical 1103 defaults in ``paper2/baselines_1103.py``
(Max-TF window 51 / quantile 0.7; GMM-OMP K=8; BCS lambda 0.01 with the true
jammer-free support, an oracle reference). None of them was tuned on DEV; the
17-candidate DEV pilot of 2026-11-16 is cited separately as sensitivity.
Per-parent DEV arrays follow the Hybrid evaluator: the three jammer variants are
scored separately and averaged inside each physical parent. Equal-Pfa scores use
exactly the E3 statistic (imported), so classical rows sit in the same table.
Everything is resumable per 16-parent chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "paper2_baseline_suite_20261126" / "E6"
DEV = ROOT / "data" / "paper2_1103" / "dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
CHUNK = 16
METHODS = ("max_tf", "max_tf_isrj1", "gmm_omp", "bcs_oracle")
# max_tf: 1103 adapter (window 51, hop 12, smoothing 3, q 0.7).
# max_tf_isrj1: exactly the first-paper comparison (max_tf_waveform_v7_7.suppress):
#   same max_tf.py, MaxTFConfig(sample_rate_hz=60e6) defaults = window 65, hop 16,
#   n_fft 128, min_clean 2, smoothing 5, q 0.7; same reference and inverse.
LABELS = {"max_tf": "Max-TF (1103 cfg)", "max_tf_isrj1": "Max-TF", "gmm_omp": "GMM-OMP", "bcs_oracle": "BCS-oracle"}


def run_max_tf_isrj1(observed: np.ndarray) -> np.ndarray:
    from paper2.baselines_1103 import tx_reference
    from paper2.vendor.baselines_1103.max_tf import MaxTFConfig, max_tf_suppress
    y = np.asarray(observed, dtype=np.complex128)
    ref = tx_reference(y.shape[-1])
    result = max_tf_suppress(y, ref, MaxTFConfig(sample_rate_hz=60e6))
    return (np.conj(np.fft.ifft(result.pc_after, axis=-1)) * ref).astype(np.complex64)


def restore(method: str, states: np.ndarray, j: np.ndarray) -> np.ndarray:
    """states [n,4,3,8,T]; j [n,3,8,T] -> restored [n,4,3,8,T]."""
    from paper2.baselines_1103 import run_bcs_oracle, run_gmm_omp, run_max_tf
    out = np.empty_like(states, dtype=np.complex64)
    n = states.shape[0]
    for st in range(states.shape[1]):
        flat = states[:, st].reshape(-1, states.shape[-1])
        if method == "max_tf":
            out[:, st] = run_max_tf(flat).reshape(states[:, st].shape)
        elif method == "max_tf_isrj1":
            out[:, st] = run_max_tf_isrj1(flat).reshape(states[:, st].shape)
        elif method == "gmm_omp":
            out[:, st] = run_gmm_omp(flat).reshape(states[:, st].shape)
        else:
            jf = j.reshape(-1, j.shape[-1])
            rec, _ = run_bcs_oracle(flat, jf)
            out[:, st] = rec.reshape(states[:, st].shape)
    return out


def dev_metrics(s, j, e, u, doppler):
    from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, states_from_components
    n = len(s)
    s4, j4, e4 = (x.reshape(n * 3, 8, 1800).astype(np.complex128) for x in (s, j, e))
    uu = np.moveaxis(u, 2, 1).reshape(n * 3, 4, 8, 1800).astype(np.complex128)  # [n,3,4,8,T]
    m = group_metrics(s4, j4, e4, uu)
    m.update(radar_metrics(s4, states_from_components(s4, j4, e4), uu, np.repeat(doppler, 3)))
    return {k: np.asarray(v).reshape(n, 3).mean(axis=1) for k, v in m.items()}


def run_dev(method: str) -> None:
    import hashlib
    from paper2.evaluation_1103_v4 import states_from_components
    h = hashlib.sha256(DEV.read_bytes()).hexdigest()
    if h != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    folder = OUT / "dev256" / method
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k] for k in ("s", "j", "epsilon"))
        pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
    t0 = time.perf_counter()
    for lo in range(0, len(pid), CHUNK):
        part = folder / f"chunk_{lo:03d}.npz"
        if part.is_file():
            continue
        hi = min(lo + CHUNK, len(pid))
        states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])  # [n,4,3,8,T]
        u = restore(method, states, j[lo:hi])
        vals = dev_metrics(s[lo:hi], j[lo:hi], e[lo:hi], u, fd[lo:hi])
        np.savez(part.with_suffix(".tmp.npz"), parent_id=pid[lo:hi], **vals)
        os.replace(part.with_suffix(".tmp.npz"), part)
        print(method, "dev chunk", lo, round(time.perf_counter() - t0, 1), flush=True)
    parts = [np.load(folder / f"chunk_{lo:03d}.npz") for lo in range(0, len(pid), CHUNK)]
    keys = [k for k in parts[0].files if k != "parent_id"]
    merged = {f"{method}__{k}": np.concatenate([p[k] for p in parts]) for k in keys}
    ids = np.concatenate([p["parent_id"] for p in parts])
    if not np.array_equal(ids, pid):
        raise RuntimeError("chunk parent order mismatch")
    np.savez_compressed(folder / "dev_parent_values.npz", parent_id=ids, **merged)
    summary = {"method": method, "parents": int(len(ids)), "dev_sha256": DEV_SHA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "config_source": ("first-paper max_tf_waveform_v7_7 (MaxTFConfig defaults, fs=60 MHz)" if method == "max_tf_isrj1"
                                 else "paper2/baselines_1103.py historical 1103 defaults"), "means": {k: float(v.mean()) for k, v in merged.items()},
               "oracle_support": method == "bcs_oracle", "server": platform.node()}
    (folder / "SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("dev done", method, flush=True)


def run_equal_pfa(method: str) -> None:
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite.equal_pfa_suite import CAL, EVAL, MAIN_PFA, curve, frame_scores, sha, digest_ids, atomic_json
    folder = OUT / "equal_pfa" / method
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for role, path, full in (("cal", CAL, False), ("eval", EVAL, True)):
        with np.load(path, allow_pickle=False) as z:
            s, j, e = (z[k] for k in ("s", "j", "epsilon"))
            ids, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
        tidx = target_peak_range(s[:, 0])
        for lo in range(0, len(ids), CHUNK):
            part = folder / f"{role}_{lo:04d}.npz"
            if part.is_file():
                continue
            hi = min(lo + CHUNK, len(ids))
            states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
            if not full:   # calibration needs only the target-absent J+E state
                sub = restore(method, states[:, 2:3], j[lo:hi])
                u = np.zeros_like(states, dtype=np.complex64)
                u[:, 2] = sub[:, 0]
            else:
                u = restore(method, states, j[lo:hi])
            sc = frame_scores(u, s[lo:hi], e[lo:hi], fd[lo:hi], tidx[lo:hi], full)
            np.savez(part.with_suffix(".tmp.npz"), **sc)
            os.replace(part.with_suffix(".tmp.npz"), part)
            print(method, role, lo, flush=True)
        chunks = [np.load(folder / f"{role}_{lo:04d}.npz") for lo in range(0, len(ids), CHUNK)]
        results[role] = ({k: np.concatenate([c[k] for c in chunks]) for k in chunks[0].files}, ids)
    (cal, cal_ids), (ev, ev_ids) = results["cal"], results["eval"]
    np.savez_compressed(folder / "SCORES.npz", cal_parent_id=cal_ids, eval_parent_id=ev_ids, cal_h0=cal["h0"],
                        **{f"eval_{k}": v for k, v in ev.items()})
    rows = curve(cal["h0"], ev)
    diag = {k: float(np.mean(v)) for k, v in ev.items() if k not in ("h0", "h0_noise", "target", "target_nojam")}
    atomic_json(folder / "ROW.json", {"family": LABELS[method],
                                      "seed": None, "calibration_sha256": sha(CAL), "evaluation_sha256": sha(EVAL),
                                      "calibration_parent_ids_sha256": digest_ids(cal_ids),
                                      "evaluation_parent_ids_sha256": digest_ids(ev_ids), "curve": rows,
                                      "main_pfa": MAIN_PFA, "diagnostics_mean": diag,
                                      "config_source": "paper2/baselines_1103.py historical 1103 defaults"})
    print("equal-pfa done", method, flush=True)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only; never touch the GPUs
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dev", "equal-pfa"))
    parser.add_argument("method", choices=METHODS)
    a = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    (run_dev if a.action == "dev" else run_equal_pfa)(a.method)


if __name__ == "__main__":
    main()
