"""Fail-closed server preflight for ReCoVP-Hybrid seed 17.

This script reads only the frozen TRAIN and DEV NPZ/receipt pairs. It never
loads confirmation, test, or historical TEST data and never starts training.
The anchor-recall statistic is explicitly a jammer-only matched-filter peak
proxy: the dataset does not provide latent per-echo annotations.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper2.io import ROOT, now, sha
from paper2.train_1103 import batches, load_arrays, make_batch
from tools.durable_io import atomic_json
from tools.vp_1103_recovp_hybrid_objective import hybrid_objective
from tools.vp_1103_recovp_hybrid import (
    ANCHOR_COUNT, FFT_LENGTH, T, ReCoVPHybrid, _rect_k16_anchors,
)


RUN = "recovp_hybrid_seed17"
BASE = ROOT / "outputs" / RUN
EVIDENCE = ROOT / "evidence" / "outputs" / RUN
DATA = ROOT / "data" / "paper2_1103"
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
GPU_INDEX = "1"
GPU_UUID = "GPU-bb60dd3a-738e-2398-908f-7f77ac05549b"
MAX_PARAMS = 59_700
TRAIN_SHA = "6148d097fecb0381e56227eb80cf1379e5f2750816ba65ce12f5457ee678f0fd"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
TOL = 1e-5


def configure_precision() -> dict:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def guard_server() -> dict:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError(f"wrong host: {platform.node()}")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"wrong repository root: {ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the selected physical GPU 1 must be the only visible CUDA device")
    physical_status = subprocess.check_output(
        ["nvidia-smi", "-i", GPU_INDEX,
         "--query-gpu=uuid,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    fields = [part.strip() for part in physical_status.split(",")]
    if len(fields) != 4 or fields[0] != GPU_UUID:
        raise RuntimeError(f"physical GPU identity changed: {physical_status}")
    free_mib, total_mib, utilization = map(float, fields[1:])
    if utilization >= 85:
        raise RuntimeError(f"GPU-1 preflight resource admission failed: {physical_status}")
    props = torch.cuda.get_device_properties(0)
    torch_uuid = getattr(props, "uuid", None)
    if torch_uuid is not None and str(torch_uuid) not in (GPU_UUID, GPU_UUID.removeprefix("GPU-")):
        raise RuntimeError(f"CUDA device UUID mismatch: {torch_uuid}")
    return {"host": platform.node(), "root": str(ROOT.resolve()),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "physical_gpu_index": 1, "physical_gpu_uuid": fields[0],
            "gpu_name": torch.cuda.get_device_name(0), "free_mib_at_start": free_mib,
            "total_mib": total_mib, "utilization_percent_at_start": utilization,
            "admission_policy": "concurrent use allowed; start only when selected GPU utilization is below 85 percent"}


def source_bindings() -> dict[str, str]:
    names = (
        "tools/vp_1103_recovp_hybrid.py",
        "tools/vp_1103_recovp_hybrid_preflight.py",
        "tools/vp_1103_recovp_hybrid_objective.py",
        "paper2/train_1103.py", "paper2/data_1103.py", "paper2/network.py",
        "paper2/science.py", "paper2/train_256_8seed.py", "paper2/io.py",
        "tools/unet_tail_loss/objective.py", "tools/durable_io.py",
    )
    result = {name: sha(ROOT / name) for name in names}
    train_source = ROOT / "tools" / "vp_1103_recovp_hybrid_train.py"
    if train_source.exists():
        result[train_source.relative_to(ROOT).as_posix()] = sha(train_source)
    return result


def data_bindings() -> dict[str, str]:
    result = {}
    for role in ("train", "dev"):
        data_path = DATA / f"{role}.npz"
        receipt_path = DATA / f"{role}.receipt.json"
        digest = sha(data_path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("sha256") != digest:
            raise RuntimeError(f"{role} receipt does not bind the data file")
        expected = TRAIN_SHA if role == "train" else DEV_SHA
        if digest != expected:
            raise RuntimeError(f"{role} SHA differs from the frozen identity")
        result[f"{role}.npz"] = digest
        result[f"{role}.receipt.json"] = sha(receipt_path)
    return result


def as_iq(complex_rows: torch.Tensor) -> torch.Tensor:
    return torch.stack((complex_rows.real, complex_rows.imag), dim=1)


def as_complex(iq: torch.Tensor) -> torch.Tensor:
    return torch.complex(iq[:, 0], iq[:, 1])


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(expected).clamp_min(1e-20)
    return float((torch.linalg.vector_norm(actual - expected) / denom).detach().cpu())


def equivariance_checks(model: ReCoVPHybrid, device: torch.device) -> dict:
    model.eval()
    g = torch.Generator(device=device).manual_seed(170113)
    y = torch.complex(torch.randn((2, T), generator=g, device=device),
                      torch.randn((2, T), generator=g, device=device))
    baseline = model.factor(as_iq(y))
    tests = []
    for amplitude in (0.2, 0.5, 2.0, 5.0):
        phase = float(torch.rand((), generator=g, device=device) * (2 * math.pi))
        c = amplitude * complex(math.cos(phase), math.sin(phase))
        transformed = model.factor(as_iq(y * c))
        tests.append({
            "amplitude": amplitude, "phase_rad": phase,
            "stage1_residual_rel_error": rel_error(transformed["stage1_restored"], c * baseline["stage1_restored"]),
            "stage2_delta_rel_error": rel_error(transformed["stage2_delta"], c * baseline["stage2_delta"]),
            "final_rel_error": rel_error(transformed["restored"], c * baseline["restored"]),
        })
    maximum = max(max(row[k] for k in (
        "stage1_residual_rel_error", "stage2_delta_rel_error", "final_rel_error")) for row in tests)

    anchors = baseline["anchors"]
    perm = torch.randperm(ANCHOR_COUNT, generator=g, device=device)
    permuted = model.factor(as_iq(y), anchors_override=anchors[:, perm])
    column_error = rel_error(permuted["stage1_restored"], baseline["stage1_restored"])
    return {"tests": tests, "max_relative_error": maximum,
            "stage1_column_permutation_relative_error": column_error,
            "threshold": TOL,
            "pass": maximum < TOL and column_error < TOL}


def total_correction(model: ReCoVPHybrid, batch: dict) -> tuple[torch.Tensor, dict]:
    y_iq = batch["y"]
    y = as_complex(y_iq)
    result = model.factor(y_iq)
    correction = y - result["restored"]
    records = len(batch["s"]) // 8
    corr_iq = as_iq(correction).reshape(records * 8, 4, 2, T)
    return corr_iq, result


def objective(model: ReCoVPHybrid, batch: dict, *, need_details=False):
    losses, details = hybrid_objective(model, batch, need_details=True,
                                       verify_stage1_identity=True)
    if not need_details:
        return losses["total"]
    return losses["total"], {key: float(value.detach()) for key, value in
                             {**losses, **details}.items()}


def gradient_and_timing(model: ReCoVPHybrid, device: torch.device) -> dict:
    train_path = DATA / "train.npz"
    arrays = load_arrays(train_path)
    if arrays["s"].shape != (512, 3, 8, T):
        raise RuntimeError(f"unexpected TRAIN array shape: {arrays['s'].shape}")
    ids = next(iter(batches(512, 1, 17)))[:8]
    batch = make_batch(arrays, ids, device)
    if batch["y"].shape != (8 * 8 * 4, 2, T):
        raise RuntimeError(f"unexpected real B=8 microbatch shape: {batch['y'].shape}")

    model.train()
    model.zero_grad(set_to_none=True)
    loss, details = objective(model, batch, need_details=True)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite B=8 preflight loss")
    loss.backward()
    missing, zero, nonfinite = [], [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            nonfinite.append(name)
        elif float(param.grad.detach().abs().max()) == 0.0:
            zero.append(name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Time ten complete B=8 forward/backward/update microbatches, with no data
    # augmentation and no changes to any saved experiment checkpoint.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    durations = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        current = objective(model, batch)
        current.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    microbatch_seconds = float(np.mean(durations))
    epoch_updates = math.ceil(512 * 3 / 32)
    microbatches_per_update = 4
    total_microbatches = 256 * epoch_updates * microbatches_per_update
    return {
        "records": 8, "input_shape": list(batch["y"].shape), "loss": float(loss.detach()),
        "loss_components": details,
        "trainable_parameter_tensors": sum(1 for p in model.parameters() if p.requires_grad),
        "gradient_tensors_with_finite_nonzero_max": sum(1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
            and float(p.grad.detach().abs().max()) > 0),
        "missing_gradient_names": missing, "zero_gradient_names": zero,
        "nonfinite_gradient_names": nonfinite,
        "all_trainable_tensors_finite_nonzero": not (missing or zero or nonfinite),
        "ten_step_seconds": durations, "mean_microbatch_seconds": microbatch_seconds,
        "estimated_256_epoch_hours": microbatch_seconds * total_microbatches / 3600,
        "estimated_updates": 256 * epoch_updates,
        "estimated_microbatches": total_microbatches,
    }


def legacy_e0_rect_anchors(wave: torch.Tensor, spectrum: torch.Tensor, count: int) -> torch.Tensor:
    """Exact one-round E0_rect selector from archived fp32_anchors.py."""
    with torch.no_grad():
        wave = wave.detach().to(torch.complex64)
        wave = wave / torch.sqrt(wave.abs().square().mean(-1, keepdim=True)).clamp_min(1e-12)
        envelope = torch.fft.ifft(torch.fft.fft(wave, n=FFT_LENGTH) * spectrum.conj(),
                                  n=FFT_LENGTH)[..., :T].abs()
        local = envelope >= F.max_pool1d(envelope[:, None], 13, 1, 6)[:, 0]
        position = torch.arange(T, device=wave.device)[None]
        available = ((position >= 2) & (position < T - 2)).expand_as(envelope).clone()
        selected = []
        for _ in range(count):
            scores = envelope.masked_fill(~(available & local), -torch.inf)
            missing = ~torch.isfinite(scores).any(-1)
            scores = torch.where(missing[:, None], envelope.masked_fill(~available, -torch.inf), scores)
            index = scores.argmax(-1)
            selected.append(index)
            available &= (position - index[:, None]).abs() >= 6
        return torch.stack(selected, -1)


def dev_anchor_recall(model: ReCoVPHybrid, device: torch.device) -> dict:
    """DEV-only jammer-only MF peak energy proxy for K12 vs K16.

    Each clean jammer's own E0_rect top-16 response peaks serve as observable
    proxy components. Their matched-filter peak energy is counted when an
    observed j+e K candidate falls within +/-2 samples. This is not latent
    simulator component truth and is labelled accordingly in every record.
    """
    path = DATA / "dev.npz"
    totals = {}
    all_weight = all_covered12 = all_covered16 = 0.0
    with np.load(path, allow_pickle=False) as z:
        jammer_types = [str(v) for v in z["jam_types"].tolist()]
        # Read only the allowed DEV scientific arrays and native metadata.
        j_all = z["j"]
        e_all = z["epsilon"]
        if j_all.shape != (256, 3, 8, T) or e_all.shape != j_all.shape:
            raise RuntimeError("DEV jammer/noise shapes differ from frozen protocol")
        spectrum = model.stage1.anchor_spectrum.to(device=device, dtype=torch.complex64)
        for jt, label in enumerate(jammer_types):
            type_total = {"weight": 0.0, "covered12": 0.0, "covered16": 0.0,
                          "records": 0, "reference_peaks": 0}
            for parent0 in range(0, 256, 32):
                parent1 = min(parent0 + 32, 256)
                pure = torch.as_tensor(j_all[parent0:parent1, jt].reshape(-1, T), device=device)
                observed = torch.as_tensor((j_all[parent0:parent1, jt] + e_all[parent0:parent1, jt]).reshape(-1, T), device=device)
                # Candidate definition matches frozen archived E0_rect_k12 and
                # the new model's K=16 extension; the first 12 must coincide.
                clean_peaks = legacy_e0_rect_anchors(pure, spectrum, 16)
                cand12 = legacy_e0_rect_anchors(observed, spectrum, 12)
                cand16 = _rect_k16_anchors(observed, spectrum)
                if not torch.equal(cand12, cand16[:, :12].long()):
                    raise RuntimeError("E0_rect K12 and K16 prefixes do not match")
                response = torch.fft.ifft(torch.fft.fft(pure.to(torch.complex64), n=FFT_LENGTH) *
                                          spectrum.conj(), n=FFT_LENGTH)[..., :T].abs()
                weight = response.gather(1, clean_peaks).square()
                covered12 = ((clean_peaks[:, :, None] - cand12[:, None, :]).abs() <= 2).any(-1)
                covered16 = ((clean_peaks[:, :, None] - cand16[:, None, :]).abs() <= 2).any(-1)
                type_total["weight"] += float(weight.sum().cpu())
                type_total["covered12"] += float((weight * covered12).sum().cpu())
                type_total["covered16"] += float((weight * covered16).sum().cpu())
                type_total["records"] += int(pure.shape[0])
                type_total["reference_peaks"] += int(clean_peaks.numel())
            totals[label] = {
                "records": type_total["records"], "proxy_components": type_total["reference_peaks"],
                "k12_proxy_energy_recall": type_total["covered12"] / max(type_total["weight"], 1e-30),
                "k16_proxy_energy_recall": type_total["covered16"] / max(type_total["weight"], 1e-30),
                "absolute_change": (type_total["covered16"] - type_total["covered12"]) / max(type_total["weight"], 1e-30),
            }
            all_weight += type_total["weight"]
            all_covered12 += type_total["covered12"]
            all_covered16 += type_total["covered16"]
    return {"definition": "jammer-only E0_rect matched-filter top-16 peak energy covered by observed j+epsilon candidate within +/-2 samples; proxy, not latent component ground truth",
            "window": "rect", "tolerance_samples": 2, "types": totals,
            "global_proxy_energy_recall_k12": all_covered12 / max(all_weight, 1e-30),
            "global_proxy_energy_recall_k16": all_covered16 / max(all_weight, 1e-30),
            "global_absolute_change": (all_covered16 - all_covered12) / max(all_weight, 1e-30)}


def run() -> dict:
    server = guard_server()
    precision = configure_precision()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    device = torch.device("cuda:0")
    sources, data = source_bindings(), data_bindings()
    model = ReCoVPHybrid().to(device)
    params = model.parameter_report()
    if params["total_effective_real_parameters"] > MAX_PARAMS:
        raise RuntimeError(f"parameter limit exceeded: {params}")
    dummy = torch.randn((1, 2, T), device=device)
    with torch.inference_mode():
        output = model(dummy)
    if output.shape != dummy.shape or not torch.isfinite(output).all():
        raise RuntimeError("1800-point output shape or finiteness failed")

    equiv = equivariance_checks(model, device)
    if not equiv["pass"]:
        raise RuntimeError(f"C-star/column permutation check failed: {equiv}")
    gradients = gradient_and_timing(model, device)
    if not gradients["all_trainable_tensors_finite_nonzero"]:
        raise RuntimeError("B=8 gradients are not finite/nonzero for every trainable tensor")
    recall = dev_anchor_recall(model, device)
    result = {
        "format": "recovp-hybrid-preflight-v1", "stage": "PREFLIGHT_PASSED",
        "status": "PASS", "scope": "TRAIN_AND_DEV_ONLY",
        "created_utc": now(), "seed": 17, "epochs": 256,
        "server": server, "precision": precision,
        "source_sha256": sources, "data_sha256": data,
        "parameter_report": params, "output_shape": list(output.shape),
        "equivariance": equiv, "real_train_microbatch": gradients,
        "dev_k12_k16_recall": recall,
        "guard": {"confirmation_access": "PROHIBITED", "historical_test_access": "PROHIBITED",
                  "training_started": False},
        "limitations": ["Finite numerical tests do not prove equivariance for all floating-point inputs.",
                        "The DEV anchor metric is a matched-filter peak proxy because latent echo annotations are absent."],
    }
    atomic_json(EVIDENCE / "PREFLIGHT.json", result)
    atomic_json(BASE / "PREFLIGHT.json", result)
    return result


def main() -> None:
    try:
        result = run()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except BaseException as exc:
        failure = {"format": "recovp-hybrid-preflight-v1", "stage": "PREFLIGHT_FAILED",
                   "status": "FAIL", "created_utc": now(), "error": repr(exc),
                   "host": platform.node(), "root": str(ROOT),
                   "training_started": False}
        # Fail closed; this script is never the training dispatcher.
        try:
            atomic_json(EVIDENCE / "PREFLIGHT.json", failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
