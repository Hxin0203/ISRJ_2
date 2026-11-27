"""WP7: same-machine inference timing on the LOCAL workstation GPU (evaluation only, no training).

User authorization 2026-11-27: when server-B GPUs are busy, the idle local GPU may be used for
evaluation/inference. Server GPUs were saturated by the resumed seed-79/97 training, so timing
there would be contaminated. All methods are timed here, on one GPU, same precision and inputs.

Identity: every checkpoint is SHA-checked against the server copy (values below, read on
server B); Hybrid source bindings are checked against local files (byte-identical to server).
Input: 4 parents from the frozen generator, role 'engineering', seed 2026112790, SNR -10 dB,
JSR 20 dB, operational observation y = S+J+E only (the four-state construction is an
evaluation device and is not timed). Unit: pulses (1800 complex samples); one CPI = 8 pulses.
Protocol: FP32, TF32 off, warm-up 20, measure 200, torch.cuda.synchronize() before/after each
call; end-to-end = numpy complex in -> numpy complex out; network-only = device tensors in/out.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

LOCAL = Path(__file__).resolve().parents[2]
OUT_DIR = LOCAL / "outputs" / "wp7_local"
EVID = LOCAL / "evidence" / "outputs" / "paper2_baseline_suite_20261126" / "WP7"
CKPT = {"ReCoVP-Hybrid": ("outputs/recovp_hybrid_seed67/seed_67/epoch_256.pt", "hybrid",
                          "de7f659217ef597dd4ff6b9da746099ccf5466cffd3489270a15bea0855817b2"),
        "UNet-Tail": ("outputs/unet_mf_tail_screen_20261117_v2/tail/seed_17/epoch_256.pt", "unet",
                      "ba7ce8f34d59ad7c071b23a9bd61c2f66a58bd78f675772f4ebbcbc549461803"),
        "CorrectionNet-Tail": ("outputs/correction_mf_tail_20261117_v1/arm/seed_17/epoch_256.pt", "cn",
                               "7ff216c12dc0e7cb4f1ce0df18aa41d7946797e7de475470997ddd25f3f0e24b")}
CVED = ("outputs/wp7_external/cvednet_e320.pt", "8cbdad0225b315ba34d0f2b3ed8d456995760eba0d2498b515087f7da923663a")
WARMUP, REPEATS = 20, 200
PULSES = (1, 8, 64)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gpu_snapshot() -> list[str]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    return q


def inputs():
    """-> ([64, 1800] complex64 operational observations = 8 CPIs of 8 pulses, data SHA)."""
    from paper2.data_1103 import generate
    path, _ = generate("engineering", count=4, seed=2026112790, snr_db=-10.0, jsr_db=20.0, out_dir=OUT_DIR / "input")
    with np.load(path, allow_pickle=False) as z:
        y = (z["s"] + z["j"] + z["epsilon"]).astype(np.complex64)       # [4, 3, 8, T]
    return y.reshape(-1, 8, 1800)[:8].reshape(64, 1800), sha(path)
def timed(fn, warmup=WARMUP, repeats=REPEATS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "p5_ms": float(np.percentile(ts, 5)),
            "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean()), "n": int(len(ts)),
            "warmup": warmup}


def flops(fn) -> int | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fn()
        return int(fc.get_total_flops())
    except Exception:
        return None


def load_net(name: str):
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    rel, key, digest = CKPT[name]
    cp = LOCAL / rel
    if sha(cp) != digest:
        raise RuntimeError(f"checkpoint SHA mismatch: {cp}")
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    if key == "hybrid":
        for src, d in payload["identity"]["source_sha256"].items():
            if sha(LOCAL / src) != d:
                raise RuntimeError(f"Hybrid source binding differs locally: {src}")
    with torch.random.fork_rng(devices=[]):
        model = E.build(key)
    model.load_state_dict(payload.get("model", payload.get("state")), strict=True)
    return model.cuda().eval(), {"checkpoint": rel, "checkpoint_sha256": digest}


def to_iq(y: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack((y.real, y.imag), 1).astype(np.float32)).cuda()


def bench(y: np.ndarray) -> dict:
    from paper2.evaluation_1103_v4 import restore_model
    from tools.vp_1103_recovp_hybrid import _complex_input
    res = {}
    # ---- pulse-wise correction networks
    for name in ("UNet-Tail", "CorrectionNet-Tail"):
        model, b = load_net(name)
        r = {"binding": b, "parameters": int(sum(p.numel() for p in model.parameters())), "by_pulses": {}}
        with torch.inference_mode():
            r["flops_per_pulse"] = flops(lambda: model(to_iq(y[:1])))
            for n in PULSES:
                dev = to_iq(y[:n])
                r["by_pulses"][n] = {"end_to_end": timed(lambda: restore_model(model, y[:n], torch.device("cuda"))),
                                     "network_only": timed(lambda: model(dev))}
        res[name] = r; del model; torch.cuda.empty_cache()
    # ---- ReCoVP-Hybrid with stage split
    model, b = load_net("ReCoVP-Hybrid")
    r = {"binding": b, "parameters_effective_real": model.parameter_report(), "by_pulses": {}}
    with torch.inference_mode():
        r["flops_per_pulse"] = flops(lambda: model(to_iq(y[:1])))
        r["flops_stage1_per_pulse"] = flops(lambda: model.stage1(to_iq(y[:1])))
        for n in PULSES:
            dev = to_iq(y[:n]); yc = _complex_input(dev); one = model.stage1(dev)
            st2 = lambda: model.stage2(one["residual"], one["correction"], yc, one["scale"], model.stage1.matched_spectrum)
            e2e = lambda: model.factor(to_iq(y[:n]))["restored"].cpu().numpy()
            r["by_pulses"][n] = {"end_to_end": timed(e2e), "network_only": timed(lambda: model(dev)),
                                 "stage1_only": timed(lambda: model.stage1(dev)), "stage2_only": timed(st2)}
    res["ReCoVP-Hybrid"] = r; del model; torch.cuda.empty_cache()
    # ---- CVEDNet (official vendor, SHA-checked)
    from tools.paper2_baseline_suite import external_suite as X
    for rel_v, d in X.VENDOR_SHA.items():
        if sha(LOCAL / "tools/paper2_baseline_suite/cvednet_vendor" / rel_v) != d:
            raise RuntimeError(f"CVEDNet vendor drift: {rel_v}")
    if sha(LOCAL / CVED[0]) != CVED[1]:
        raise RuntimeError("CVEDNet checkpoint SHA mismatch")
    sys.path.insert(0, str(LOCAL / "tools/paper2_baseline_suite/cvednet_vendor"))
    import cvednet_adapter as ca
    model = ca.build_model("cuda")
    model.load_state_dict(torch.load(LOCAL / CVED[0], map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    r = {"binding": {"checkpoint": CVED[0], "checkpoint_sha256": CVED[1]}, "parameters": ca.count_parameters(model), "by_pulses": {}}
    with torch.inference_mode():
        x1 = torch.from_numpy(y[:1]).cuda()
        r["flops_per_pulse"] = flops(lambda: ca.forward_complex(model, x1 / ca.pulse_scale(x1)))
        for n in PULSES:
            x = torch.from_numpy(y[:n]).cuda(); s = ca.pulse_scale(x)
            r["by_pulses"][n] = {"end_to_end": timed(lambda: ca.infer_numpy(model, y[:n], "cuda", batch=64)),
                                 "network_only": timed(lambda: ca.forward_complex(model, x / s))}
    res["CVEDNet"] = r; del model; torch.cuda.empty_cache()
    # ---- RSDR-Net (CPI unit; observed-only state)
    from tools.rsdr_adapter import RSDRNet
    net = RSDRNet(device="cuda:0", batch_pulses=256, verify_vendor=True)
    r = {"parameters": int(sum(p.numel() for m in (*net.experts, *net.routers, net.gate) for p in m.parameters())),
         "unit_note": "processes whole CPIs (8 pulses); restore(..., state_indexes=[3]) runs experts/routers on y only",
         "network_only": "not separated (adapter is numpy in/out)", "by_pulses": {}}
    for n in (8, 64):
        st = np.repeat(y[:n].reshape(n // 8, 1, 8, 1800), 4, axis=1)
        r["by_pulses"][n] = {"end_to_end": timed(lambda: net.restore(st, state_indexes=[3]))}
    res["RSDR-Net"] = r; del net; torch.cuda.empty_cache()
    # ---- classical methods on CPU (4 threads, fewer repeats: CPU-bound)
    from tools.paper2_baseline_suite.classical_suite import run_max_tf_isrj1
    from paper2.baselines_1103 import run_gmm_omp
    torch.set_num_threads(4)
    for name, fn in (("Max-TF", run_max_tf_isrj1), ("GMM-OMP", run_gmm_omp)):
        res[name] = {"device": "CPU", "by_pulses": {n: {"end_to_end": timed(lambda: fn(y[:n]), warmup=5, repeats=50)}
                                                   for n in PULSES}}
    return res


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one local CUDA device expected")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    y, data_sha = inputs()
    before = gpu_snapshot(); t0 = time.time()
    res = bench(y)
    after = gpu_snapshot()
    util = lambda rows: [int(r.split(",")[3].strip().rstrip(" %")) for r in rows]
    doc = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
           "cpu": platform.processor(), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "gpu_before": before, "gpu_after": after,
           "interfered": bool(abs(util(after)[0] - util(before)[0]) > 20),
           "note_concurrency": "local desktop applications (browser, Lightroom, wallpaper) share this GPU; no compute jobs",
           "input_sha256": data_sha, "precision": "FP32, TF32 off", "protocol": __doc__, "seconds": time.time() - t0,
           "script_sha256": sha(Path(__file__)), "flops_note": "torch FlopCounterMode; FFT/linalg/complex elementwise may be uncounted",
           "methods": res}
    EVID.mkdir(parents=True, exist_ok=True)
    (EVID / "TIMING.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: {n: round(v["end_to_end"]["median_ms"], 3) for n, v in m["by_pulses"].items()}
                      for k, m in res.items()}, indent=1))
    print("gpu before", before, "after", after)


if __name__ == "__main__":
    sys.path.insert(0, str(LOCAL))
    main()
