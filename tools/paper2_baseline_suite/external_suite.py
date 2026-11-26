"""E7: frozen external learned baseline CVEDNet on DEV256 and the equal-Pfa sets.

The model is the user's sealed first-paper CVEDNet (official chai-xuan/CVEDNet
@1f1dbf3, epoch 320, selected by DEV SI-SNR on the first-paper TRAIN512/DEV256;
no second-paper data took part in its training or selection). It is evaluated
here as-is (no fine-tuning). The vendored adapter and official sources are
byte-identical to the training copy on server B and are SHA-checked.

DEV per-parent metrics reuse ``classical_suite.dev_metrics``; the equal-Pfa
statistic, threshold policy and curve reuse ``equal_pfa_suite`` unchanged, so
CVEDNet rows sit in the same tables as every other method. CVEDNet outputs the
restored waveform directly (no correction subtraction).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs" / "paper2_baseline_suite_20261126" / "E7"
VENDOR = ROOT / "tools" / "paper2_baseline_suite" / "cvednet_vendor"
VENDOR_SHA = {
    "cvednet_adapter.py": "5775486a233aa8a4f9aeb1398897c534576a63c362b9807b1e98e22fa035dfea",
    "CVEDNet_official/models/CVEDNet.py": "f4f6d6da51d0950b3dedbf5032b8972374c985c0ba6ce2b3dc8b9aef8f90dc53",
    "CVEDNet_official/models/util/complex_util.py": "10acec07dde4175e5012d62cfc28ee26e5b75c6d90653af1dde9f65390ce545f",
}
CKPT = Path("D:/code/Xin/RSDR_Net/cvednet_baseline_20261125_r1/sealed_e320/cvednet_e320.pt")
CKPT_SHA = "8cbdad0225b315ba34d0f2b3ed8d456995760eba0d2498b515087f7da923663a"
DEV = ROOT / "data" / "paper2_1103" / "dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
CHUNK = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_model():
    import torch
    for rel, digest in VENDOR_SHA.items():
        if sha(VENDOR / rel) != digest:
            raise RuntimeError(f"CVEDNet vendor drift: {rel}")
    if sha(CKPT) != CKPT_SHA:
        raise RuntimeError("sealed CVEDNet checkpoint identity changed")
    sys.path.insert(0, str(VENDOR))
    import cvednet_adapter as ca
    model = ca.build_model("cuda")
    cp = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model"], strict=True)
    model.eval()
    params = ca.count_parameters(model)
    meta = {"checkpoint": str(CKPT), "checkpoint_sha256": CKPT_SHA, "epoch": cp.get("epoch"),
            "val_sisnr_first_paper_dev": cp.get("val_sisnr"), "parameters": params,
            "vendor_sha256": VENDOR_SHA, "official_commit": ca.OFFICIAL_COMMIT}
    return model, ca, meta


def restore(model, ca, states: np.ndarray) -> np.ndarray:
    """[n,4,3,8,T] observations -> restored waveforms (CVEDNet returns the waveform)."""
    return ca.infer_numpy(model, states, "cuda", batch=64).astype(np.complex64)


def run_dev(model, ca, meta) -> None:
    from paper2.evaluation_1103_v4 import states_from_components
    from tools.paper2_baseline_suite.classical_suite import dev_metrics
    folder = OUT / "dev256" / "cvednet"
    if (folder / "SUMMARY.json").is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    if sha(DEV) != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k] for k in ("s", "j", "epsilon"))
        pid, fd = z["parent_id"].copy(), z["doppler_hz"].copy()
    t0 = time.perf_counter()
    parts = []
    for lo in range(0, len(pid), CHUNK):
        hi = min(lo + CHUNK, len(pid))
        states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
        u = restore(model, ca, states)
        parts.append(dev_metrics(s[lo:hi], j[lo:hi], e[lo:hi], u, fd[lo:hi]))
        print("cvednet dev", lo, round(time.perf_counter() - t0, 1), flush=True)
    merged = {f"cvednet__{k}": np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(folder / "dev_parent_values.npz", parent_id=pid, **merged)
    summary = {"method": "CVEDNet", "parents": int(len(pid)), "dev_sha256": DEV_SHA, "created_utc": now(),
               "server": platform.node(), "model": meta, "seconds": time.perf_counter() - t0,
               "means": {k: float(v.mean()) for k, v in merged.items()},
               "training_data": "first-paper TRAIN512/DEV256 (rsdr-core-v1); no second-paper data used for training or selection"}
    (folder / "SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def run_equal_pfa(model, ca, meta) -> None:
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import target_peak_range
    from tools.paper2_baseline_suite.equal_pfa_suite import (CAL, EVAL, MAIN_PFA, atomic_json, curve,
                                                              digest_ids, frame_scores)
    folder = OUT / "equal_pfa" / "cvednet"
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
        for lo in range(0, len(ids), 32):
            hi = min(lo + 32, len(ids))
            states = states_from_components(s[lo:hi], j[lo:hi], e[lo:hi])
            u = restore(model, ca, states)
            parts.append(frame_scores(u, s[lo:hi], e[lo:hi], fd[lo:hi], tidx[lo:hi], full))
            print("cvednet", role, lo, round(time.perf_counter() - t0, 1), flush=True)
        results[role] = ({k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, ids)
    (cal, cal_ids), (ev, ev_ids) = results["cal"], results["eval"]
    np.savez_compressed(folder / "SCORES.npz", cal_parent_id=cal_ids, eval_parent_id=ev_ids, cal_h0=cal["h0"],
                        **{f"eval_{k}": v for k, v in ev.items()})
    diag = {k: float(np.mean(v)) for k, v in ev.items() if k not in ("h0", "h0_noise", "target", "target_nojam")}
    atomic_json(folder / "ROW.json", {
        "family": "CVEDNet", "seed": None, "created_utc": now(), "server": platform.node(), "model": meta,
        "calibration_sha256": sha(CAL), "evaluation_sha256": sha(EVAL),
        "calibration_parent_ids_sha256": digest_ids(cal_ids), "evaluation_parent_ids_sha256": digest_ids(ev_ids),
        "curve": curve(cal["h0"], ev), "main_pfa": MAIN_PFA, "diagnostics_mean": diag,
        "inference_and_scoring_seconds": time.perf_counter() - t0})


def work() -> None:
    from tools.paper2_baseline_suite.equal_pfa_suite import configure_precision, require_server
    admission = require_server()
    configure_precision()
    import torch
    torch.set_num_threads(4)
    model, ca, meta = load_model()
    meta = {**meta, "gpu_admission": admission}
    run_dev(model, ca, meta)
    run_equal_pfa(model, ca, meta)
    print("E7 cvednet done", flush=True)


def launch() -> None:
    """Wait for a GPU below 85% utilization, then run the worker on it."""
    from tools.server_gpu_admission import NoGpuAvailable, choose_gpu
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            gpu = choose_gpu()
        except NoGpuAvailable:
            time.sleep(60)
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["physical_gpu_index"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        code = subprocess.call([sys.executable, "-m", "tools.paper2_baseline_suite.external_suite", "work"],
                               cwd=str(ROOT), env=env)
        if code == 3:
            time.sleep(60)
            continue
        (OUT / "LAUNCH_RESULT.json").write_text(json.dumps({"exit_code": code, "gpu": gpu, "utc": now()}, indent=1),
                                                encoding="utf-8")
        return


def main() -> None:
    if platform.node().casefold() != "desktop-rfferm3":
        raise RuntimeError("server B only")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("launch", "work"))
    a = parser.parse_args()
    if a.action == "launch":
        launch()
        return
    from tools.server_gpu_admission import NoGpuAvailable
    try:
        work()
    except NoGpuAvailable as exc:
        print("gpu busy", exc, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
