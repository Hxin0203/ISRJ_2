"""E7b: frozen first-paper RSDR-Net (native seed 40312102, epoch 32) on DEV256 and the equal-Pfa sets.

Weights and vendor sources are SHA-checked by ``tools.rsdr_adapter.RSDRNet``. The
model is evaluated as-is (no training on second-paper data). It processes the
four statewise groups of one jammer variant jointly, as in the first paper, so
each jammer variant is restored separately. Metrics, statistic and threshold
policy are imported from the other suite modules unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "paper2_baseline_suite_20261126" / "E7"
DEV = ROOT / "data" / "paper2_1103" / "dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
CHUNK = 16
LABEL = "rsdr_net"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load():
    from tools.rsdr_adapter import RSDRNet
    net = RSDRNet(device="cuda:0", batch_pulses=256, verify_vendor=True)
    params = sum(p.numel() for m in (*net.experts, *net.routers, net.gate) for p in m.parameters())
    meta = {"model": "RSDR-Net (first paper, F-R7)", "native_seed": 40312102, "epoch": 32,
            "parameters": int(params), "assets": "outputs/rsdr_unified_20261115/assets (SHA-checked by adapter)"}
    return net, meta


def restore(net, states: np.ndarray) -> np.ndarray:
    """[n,4,3,8,T] -> [n,4,3,8,T]; each jammer variant restored as one four-state group."""
    out = np.empty(states.shape, dtype=np.complex64)
    for jt in range(states.shape[2]):
        out[:, :, jt] = np.asarray(net.restore(np.ascontiguousarray(states[:, :, jt])), dtype=np.complex64)
    return out


def run_dev(net, meta) -> None:
    from paper2.evaluation_1103_v4 import states_from_components
    from tools.paper2_baseline_suite.classical_suite import dev_metrics
    folder = OUT / "dev256" / LABEL
    if (folder / "SUMMARY.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(DEV.read_bytes()).hexdigest() != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k] for k in ("s", "j", "epsilon"))
        pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
    t0 = time.perf_counter()
    parts = []
    for lo in range(0, len(pid), CHUNK):
        hi = min(lo + CHUNK, len(pid))
        states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
        parts.append(dev_metrics(s[lo:hi], j[lo:hi], e[lo:hi], restore(net, states), fd[lo:hi]))
        print("rsdr dev", lo, round(time.perf_counter() - t0, 1), flush=True)
    merged = {f"{LABEL}__{k}": np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(folder / "dev_parent_values.npz", parent_id=pid, **merged)
    (folder / "SUMMARY.json").write_text(json.dumps({
        "method": "RSDR-Net", "parents": int(len(pid)), "dev_sha256": DEV_SHA, "created_utc": now(),
        "server": platform.node(), "model": meta, "seconds": time.perf_counter() - t0,
        "means": {k: float(v.mean()) for k, v in merged.items()},
        "training_data": "first-paper data and budget; no second-paper data used for training or selection"},
        indent=1, default=str), encoding="utf-8")


def run_equal_pfa(net, meta) -> None:
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite.equal_pfa_suite import (CAL, EVAL, MAIN_PFA, atomic_json, curve,
                                                              digest_ids, frame_scores, sha)
    folder = OUT / "equal_pfa" / LABEL
    if (folder / "ROW.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = {}
    for role, path, full in (("cal", CAL, False), ("eval", EVAL, True)):
        with np.load(path, allow_pickle=False) as z:
            s, j, e = (z[k] for k in ("s", "j", "epsilon"))
            ids, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
        tidx = target_peak_range(s[:, 0])
        parts = []
        for lo in range(0, len(ids), CHUNK):
            hi = min(lo + CHUNK, len(ids))
            states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
            parts.append(frame_scores(restore(net, states), s[lo:hi], e[lo:hi], fd[lo:hi], tidx[lo:hi], full))
            if lo % 256 == 0:
                print("rsdr", role, lo, round(time.perf_counter() - t0, 1), flush=True)
        results[role] = ({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, ids)
    (cal, cal_ids), (ev, ev_ids) = results["cal"], results["eval"]
    np.savez_compressed(folder / "SCORES.npz", cal_parent_id=cal_ids, eval_parent_id=ev_ids, cal_h0=cal["h0"],
                        **{f"eval_{k}": v for k, v in ev.items()})
    diag = {k: float(np.mean(v)) for k, v in ev.items() if k not in ("h0", "h0_noise", "target", "target_nojam")}
    atomic_json(folder / "ROW.json", {
        "family": "RSDR-Net", "seed": None, "created_utc": now(), "server": platform.node(), "model": meta,
        "calibration_sha256": sha(CAL), "evaluation_sha256": sha(EVAL),
        "calibration_parent_ids_sha256": digest_ids(cal_ids), "evaluation_parent_ids_sha256": digest_ids(ev_ids),
        "curve": curve(cal["h0"], ev), "main_pfa": MAIN_PFA, "diagnostics_mean": diag,
        "inference_and_scoring_seconds": time.perf_counter() - t0})


def work() -> None:
    from tools.paper2_baseline_suite.equal_pfa_suite import require_server
    admission = require_server()
    net, meta = load()
    meta = {**meta, "gpu_admission": admission}
    run_dev(net, meta)
    run_equal_pfa(net, meta)
    print("E7b rsdr done", flush=True)


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("work",))
    parser.parse_args()
    from tools.server_gpu_admission import NoGpuAvailable
    try:
        work()
    except NoGpuAvailable as exc:
        print("gpu busy", exc, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
