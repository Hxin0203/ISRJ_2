"""One synchronized end-to-end CPU/GPU benchmark for all Paper 2 models."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from vp_1103_common import WORK, atomic_json, now, provenance, read, sha, validate_role
from vp_1103_p2_smoothvp import SmoothVPNet, real_parameters
from tools.unet_ablation.models import make_model
from tools.unet_ablation.runtime import configure_precision

OUT = WORK / "evidence/outputs/p2_smoothvp_latency"
NAMES = {"UNet-Original": "unet", "UNet-Tail": "unet", "CorrectionNet-Original": "full",
         "CorrectionNet-Tail": "full", "P0-Best-Tail": "m2", "M3-SmoothVP": "vp"}


class Restore(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, iq):
        return iq - self.model(iq)


def model_for(name, coordinates, checkpoint_root=None):
    kind = NAMES[name]
    if kind == "vp":
        model = SmoothVPNet(coordinates)
        path = WORK / "outputs/p2_smoothvp_screen/M3-SmoothVP/17/epoch_256.pt"
    else:
        variant = "no_dechirp" if kind == "m2" and coordinates == "RAW_IQ" else "full" if kind == "m2" else kind
        model = make_model(variant)
        path = (WORK / "outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail/17/epoch_256.pt"
                if kind == "m2" and coordinates == "RAW_IQ" else
                (checkpoint_root or WORK / "outputs/vp_1103_checkpoint_cache") / f"{name}.pt")
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        binding = {"weights": "TRAINED_EPOCH_256", "checkpoint_sha256": sha(path)}
    else:
        if kind == "vp":
            raise FileNotFoundError(f"The neural VP benchmark needs its trained checkpoint: {path}")
        binding = {"weights": "CONSTRUCTOR_WEIGHTS",
                   "scope": "Frozen baseline architecture; optional --checkpoint-root loads trained weights"}
    return Restore(model.eval()), binding


def macs_count(model, x):
    total = 0

    def count(module, inputs, outputs):
        nonlocal total
        if isinstance(module, nn.Conv1d):
            total += outputs.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        else:
            total += outputs.numel() * module.in_features

    handles = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    try:
        with torch.inference_mode():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total


def effective_count(model, x):
    model.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    count = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return count


def measure(model, x, warmup, repeats):
    cuda = x.is_cuda
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            y = model(x)
            if cuda:
                torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if not torch.isfinite(y).all():
                raise FloatingPointError("Latency run produced nonfinite values")
            times.append(elapsed)
    p50, p95 = np.quantile(times, [.5, .95])
    return {"p50_ms": float(p50), "p95_ms": float(p95),
            "pulse_throughput_per_second": float(1000 * len(x) / np.mean(times)),
            "group_throughput_per_second": float(1000 / np.mean(times)),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "raw_repeats_ms": times}


def run(warmup=50, repeats=200, checkpoint_root=None, devices=("cpu", "cuda")):
    p0 = read(WORK / "evidence/outputs/p0_dechirp_tail_2x2/SUMMARY.json")
    coordinates = p0["coordinates"]
    if coordinates not in ("RAW_IQ", "DECHIRP"):
        raise RuntimeError("P0 coordinates must be locked before the unified benchmark")
    train_path, receipt = validate_role("train")
    with np.load(train_path, allow_pickle=False) as z:
        wave = (z["s"][0, 0] + z["j"][0, 0] + z["epsilon"][0, 0]).astype(np.complex64)
    pulse = torch.from_numpy(np.stack((wave.real, wave.imag), axis=1))
    precision = configure_precision()
    torch.set_num_threads(2)
    start = now()
    status = {**provenance(__file__, Path(__file__).with_name("vp_1103_p2_smoothvp.py")),
              "stage": "RUNNING", "started_utc": start, "ended_utc": None, "data": {"train": receipt},
              "seeds": [17], "batch_groups": 1, "precision": precision}
    atomic_json(OUT / "STATUS.json", status)
    rows = {}
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested GPU benchmark but CUDA is unavailable")
        rows[device] = {}
        for name in NAMES:
            model, binding = model_for(name, coordinates, checkpoint_root)
            model = model.to(device)
            row = {"binding": binding, "registered_real_parameters": real_parameters(model),
                   "effective_real_parameters": effective_count(model, pulse[:1].to(device)), "timings": {}}
            for count in (1, 8):
                x = pulse[:count].to(device)
                macs = macs_count(model, x)
                row["timings"][str(count)] = {**measure(model, x, warmup, repeats),
                                              "conv1d_linear_real_macs": macs}
            rows[device][name] = row
            atomic_json(OUT / "SUMMARY.json", {"coordinates": coordinates, "devices": rows,
                        "warmup": warmup, "repeats": repeats, "torch": torch.__version__,
                        "cpu_threads": 2, "precision": precision,
                        "mac_scope": "Real Conv1d and Linear only; excludes FFT, GroupNorm, complex dictionary synthesis, "
                                     "complex128 solve and elementwise operations. Latency includes the complete restored output.",
                        "cuda_device_name": torch.cuda.get_device_name() if device == "cuda" else None})
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    from tools.durable_io import atomic_write
    lines = ["# Unified inference benchmark", "",
             "Batch is one pulse group. Group sizes are one pulse and eight pulses. Full restoration is timed.",
             "Each model uses 50 warmups and 200 synchronized wall-clock repetitions by default.",
             "Historical constructor-weight timings are explicitly tagged in SUMMARY.json.",
             "", "| Device | Model | Pulses | Active real parameters | P50 ms | P95 ms | Pulse/s | Peak GPU bytes |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for device, models in rows.items():
        for name, row in models.items():
            for n, timing in row["timings"].items():
                lines.append(f"| {device} | {name} | {n} | {row['effective_real_parameters']} | "
                             f"{timing['p50_ms']:.4f} | {timing['p95_ms']:.4f} | "
                             f"{timing['pulse_throughput_per_second']:.1f} | {timing['peak_gpu_allocated_bytes']} |")
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "STATUS.json", {**status, "stage": "COMPLETE", "ended_utc": now(),
                                    "summary_sha256": sha(OUT / "SUMMARY.json")})


def run_v21(warmup=50, repeats=200):
    """Same complete restoration timing, with the locked v2.1 extractor included."""
    from vp_1103_p2_smoothvp_v21 import SmoothVPNetV21
    from vp_1103_snr_jsr import DEFINITION
    from tools.durable_io import atomic_write
    out = WORK / "evidence/outputs/p2_smoothvp_v21_latency"
    started = now()
    config = read(WORK / "outputs/p2_smoothvp_v21_screen/CONFIG.json")
    path, data_binding = validate_role("train")
    with np.load(path, allow_pickle=False) as z:
        wave = z["s"][0, 0] + z["j"][0, 0] + z["epsilon"][0, 0]
    pulse = torch.from_numpy(np.stack((wave.real, wave.imag), axis=1).astype(np.float32))
    precision = configure_precision()
    torch.set_num_threads(2)
    status = {**provenance(__file__, WORK / "tools/vp_1103_p2_smoothvp_v21.py", WORK / "tools/vp_1103_v21_extractors.py"),
              "started_utc": started, "ended_utc": None, "stage": "RUNNING", "data": data_binding,
              "warmup": warmup, "repeats": repeats, "precision": precision}
    atomic_json(out / "STATUS.json", status)
    rows = {}
    for device in ("cpu", "cuda"):
        rows[device] = {}
        for name, variant in (("UNet-Tail", "unet"), ("CorrectionNet-Tail", "full"),
                              ("P0-NoDechirp-Tail", "no_dechirp"), ("SmoothVP-v21", "vp21")):
            if variant == "vp21":
                model = SmoothVPNetV21(config["extractor"])
                cp = WORK / "outputs/p2_smoothvp_v21_screen/M3-SmoothVP-v21/17/epoch_256.pt"
            else:
                model = make_model(variant)
                cp = (WORK / "outputs/p0_dechirp_tail_2x2/P0-NoDechirp-Tail/17/epoch_256.pt"
                      if variant == "no_dechirp" else WORK / "outputs/vp_1103_checkpoint_cache" / f"{name}.pt")
            if cp.exists():
                model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model"])
                binding = {"weights": "TRAINED_EPOCH_256", "checkpoint_sha256": sha(cp)}
            elif variant in ("unet", "full"):
                binding = {"weights": "CONSTRUCTOR_ARCHITECTURE_ONLY", "scope": "Branch-free baseline CNN architecture timing; trained weights unavailable locally"}
            else:
                raise FileNotFoundError(cp)
            model = Restore(model.eval()).to(device)
            row = {"binding": binding, "effective_real_parameters": effective_count(model, pulse[:1].to(device)), "timings": {}}
            for count in (1, 8):
                x = pulse[:count].to(device)
                row["timings"][str(count)] = {**measure(model, x, warmup, repeats),
                                               "conv1d_linear_real_macs": macs_count(model, x)}
            rows[device][name] = row
            del model
            if device == "cuda": torch.cuda.empty_cache()
    summary = {"devices": rows, "warmup": warmup, "repeats": repeats, "cpu_threads": 2,
               "precision": precision, "cuda_name": torch.cuda.get_device_name(), "torch": torch.__version__,
               "scope": "Full y-j_hat restoration, including candidate FFT/CLEAN, refinement, windows and complex128 solve",
               "mac_scope": "Conv1d/Linear only; FFT, matrix solve and elementwise operations excluded from MAC count"}
    atomic_json(out / "SUMMARY.json", summary)
    lines = ["# v2.1 统一端到端计时", "", DEFINITION, "",
             "所有模型统一 50 次预热，batch=1 脉冲组；1/8 脉冲分别计时。基线权重绑定详见 SUMMARY.json。",
             "", "| 设备 | 模型 | 脉冲数 | 参数 | P50 ms | P95 ms | 脉冲/s | 峰值显存 bytes |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for device, models in rows.items():
        for name, row in models.items():
            for count, result in row["timings"].items():
                lines.append(f"| {device} | {name} | {count} | {row['effective_real_parameters']} | "
                             f"{result['p50_ms']:.5g} | {result['p95_ms']:.5g} | {result['pulse_throughput_per_second']:.5g} | {result['peak_gpu_allocated_bytes']} |")
    atomic_write(out / "REPORT.md", lambda f: f.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(out / "STATUS.json", {**status, "stage": "COMPLETE", "ended_utc": now(), "summary_sha256": sha(out / "SUMMARY.json")})


def run_recovp(warmup=50,repeats=200):
    """One entry, same timing implementation, with the complete ReCoVP path."""
    global OUT,NAMES,model_for
    from vp_1103_recovp import ReCoVPNet
    original_out,original_names,original_factory=OUT,NAMES,model_for
    config=read(WORK/"outputs/recovp_formal/CONFIG.json")
    OUT=WORK/"evidence/outputs/recovp_latency"
    unavailable={}
    NAMES={"UNet-Tail":"unet","CorrectionNet-Tail":"full","P0-NoDechirp-Tail":"m2","ReCoVP-Base":"recovp"}
    soft=WORK/"outputs/recovp_formal/ReCoVP-SoftShare/17/epoch_256.pt"
    if soft.exists():NAMES["ReCoVP-SoftShare"]="recovp"
    else:unavailable["ReCoVP-SoftShare"]="NOT_TRAINED_GATE"
    def factory(name,coordinates,checkpoint_root=None):
        if name.startswith("ReCoVP"):
            model=ReCoVPNet(config["extractor"],share_mode="soft_subspace" if name.endswith("SoftShare") else "none")
            cp=WORK/"outputs/recovp_formal"/name/"17/epoch_256.pt"
            model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["model"])
            return Restore(model.eval()),{"weights":"TRAINED_EPOCH_256","checkpoint_sha256":sha(cp)}
        value,binding=original_factory(name,coordinates,checkpoint_root)
        if binding["weights"]!="TRAINED_EPOCH_256":raise RuntimeError(f"Actual trained baseline weights required: {name}")
        return value,binding
    model_for=factory
    try:
        run(warmup,repeats)
        summary=read(OUT/"SUMMARY.json")
        summary.update(unavailable_models=unavailable,
            recovp_arithmetic="FP32 stored parameters, float64 real CNN/features/heads, complex128 solve, float32 output",
            complete_recovp_sources=provenance(WORK/"tools/vp_1103_recovp.py",WORK/"tools/vp_1103_recovp_extractors.py"))
        atomic_json(OUT/"SUMMARY.json",summary)
        status=read(OUT/"STATUS.json");status["summary_sha256"]=sha(OUT/"SUMMARY.json");atomic_json(OUT/"STATUS.json",status)
    finally:
        OUT,NAMES,model_for=original_out,original_names,original_factory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--suite", choices=("original", "v21", "recovp"), default="original")
    options = parser.parse_args()
    if options.suite == "recovp":
        run_recovp(options.warmup, options.repeats)
    elif options.suite == "v21":
        run_v21(options.warmup, options.repeats)
    else:
        run(options.warmup, options.repeats, options.checkpoint_root, options.devices)
