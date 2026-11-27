"""WP7 rerun: interleaved (round-robin) GPU timing on the LOCAL workstation, evaluation only.

The first local run (wp7_local_timing.py) was flagged interfered: desktop applications changed
GPU load from 45% to 18% during a 10-minute sequential sweep, so methods measured at different
times saw different background load. Here every GPU configuration is measured in each of
ROUNDS rounds, in a freshly shuffled order per round (fixed seed), with all models resident.
Reported value: median over rounds of the per-round median; spread = min/max of round medians.
CPU methods (Max-TF, GMM-OMP) are not re-measured. Same inputs, precision and checkpoints.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL))
from tools.paper2_baseline_suite import wp7_local_timing as T  # noqa: E402

ROUNDS, WARM, REP = 5, 5, 40


def callables(y):
    from paper2.evaluation_1103_v4 import restore_model
    from tools.vp_1103_recovp_hybrid import _complex_input
    from tools.paper2_baseline_suite import external_suite as X
    from tools.rsdr_adapter import RSDRNet
    cfg, keep = {}, []
    for name in ("UNet-Tail", "CorrectionNet-Tail"):
        m, _ = T.load_net(name); keep.append(m)
        for n in T.PULSES:
            dev = T.to_iq(y[:n])
            cfg[(name, n, "end_to_end")] = (lambda m=m, n=n: restore_model(m, y[:n], torch.device("cuda")))
            cfg[(name, n, "network_only")] = (lambda m=m, dev=dev: m(dev))
    h, _ = T.load_net("ReCoVP-Hybrid"); keep.append(h)
    for n in T.PULSES:
        dev = T.to_iq(y[:n]); yc = _complex_input(dev)
        with torch.inference_mode():
            one = h.stage1(dev)
        cfg[("ReCoVP-Hybrid", n, "end_to_end")] = (lambda n=n: h.factor(T.to_iq(y[:n]))["restored"].cpu().numpy())
        cfg[("ReCoVP-Hybrid", n, "network_only")] = (lambda dev=dev: h(dev))
        cfg[("ReCoVP-Hybrid", n, "stage1_only")] = (lambda dev=dev: h.stage1(dev))
        cfg[("ReCoVP-Hybrid", n, "stage2_only")] = (lambda one=one, yc=yc: h.stage2(one["residual"], one["correction"], yc,
                                                                                  one["scale"], h.stage1.matched_spectrum))
    sys.path.insert(0, str(LOCAL / "tools/paper2_baseline_suite/cvednet_vendor"))
    import cvednet_adapter as ca
    for rel_v, d in X.VENDOR_SHA.items():
        if T.sha(LOCAL / "tools/paper2_baseline_suite/cvednet_vendor" / rel_v) != d:
            raise RuntimeError(f"CVEDNet vendor drift: {rel_v}")
    if T.sha(LOCAL / T.CVED[0]) != T.CVED[1]:
        raise RuntimeError("CVEDNet checkpoint SHA mismatch")
    cv = ca.build_model("cuda")
    cv.load_state_dict(torch.load(LOCAL / T.CVED[0], map_location="cpu", weights_only=False)["model"], strict=True)
    cv.eval(); keep.append(cv)
    for n in T.PULSES:
        x = torch.from_numpy(y[:n]).cuda(); s = ca.pulse_scale(x)
        cfg[("CVEDNet", n, "end_to_end")] = (lambda n=n: ca.infer_numpy(cv, y[:n], "cuda", batch=64))
        cfg[("CVEDNet", n, "network_only")] = (lambda x=x, s=s: ca.forward_complex(cv, x / s))
    net = RSDRNet(device="cuda:0", batch_pulses=256, verify_vendor=True); keep.append(net)
    for n in (8, 64):
        st = np.repeat(y[:n].reshape(n // 8, 1, 8, 1800), 4, axis=1)
        cfg[("RSDR-Net", n, "end_to_end")] = (lambda st=st: net.restore(st, state_indexes=[3]))
    return cfg, keep


def main():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    y, data_sha = T.inputs()
    cfg, _keep = callables(y)
    # The RSDR adapter enables deterministic algorithms at load time; Hybrid's CUDA median has no
    # deterministic kernel. Timing does not need bitwise determinism, so restore the default.
    det_was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    rng = random.Random(20261127)
    rounds = {k: [] for k in cfg}
    snaps = [T.gpu_snapshot()]
    t0 = time.time()
    with torch.inference_mode():
        for r in range(ROUNDS):
            order = list(cfg); rng.shuffle(order)
            for k in order:
                rounds[k].append(T.timed(cfg[k], warmup=WARM, repeats=REP)["median_ms"])
            snaps.append(T.gpu_snapshot())
            print("round", r + 1, "done", round(time.time() - t0), "s", snaps[-1], flush=True)
    res = {}
    for (m, n, kind), v in rounds.items():
        res.setdefault(m, {}).setdefault(str(n), {})[kind] = {"median_of_round_medians_ms": float(np.median(v)),
                                                              "round_min_ms": float(min(v)), "round_max_ms": float(max(v)),
                                                              "round_medians_ms": v}
    util = [int(s[0].split(",")[3].strip().rstrip(" %")) for s in snaps]
    doc = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": torch.cuda.get_device_name(0),
           "rounds": ROUNDS, "warmup_per_config_round": WARM, "repeats_per_config_round": REP,
           "gpu_snapshots_per_round": snaps, "gpu_util_range_percent": [min(util), max(util)],
           "deterministic_algorithms_enabled_by_rsdr_adapter": det_was, "deterministic_algorithms_during_timing": False,
           "input_sha256": data_sha, "script_sha256": T.sha(Path(__file__)), "seconds": time.time() - t0,
           "protocol": __doc__, "methods": res}
    T.EVID.mkdir(parents=True, exist_ok=True)
    (T.EVID / "TIMING_RR.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({m: {n: {k: round(x["median_of_round_medians_ms"], 2) for k, x in d.items()} for n, d in v.items()}
                      for m, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
