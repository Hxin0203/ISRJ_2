"""Adapter around the unmodified official CVEDNet (chai-xuan/CVEDNet @ 1f1dbf3).

Only interface changes live here; every change is listed in ADAPTATION_REPORT.md.
  A1  ptflops/pandas are imported by the official model file for profiling only;
      inert stubs are used when the packages are absent (no effect on the network).
  A2  Input length 1800 (official 768). The network is length-agnostic:
      encoder 1800 -> 224 frames, decoder 224 -> 1800.
  A3  Per-pulse input-RMS scaling (official data loader unpublished). The scale is
      computed from the observed input only and the output is rescaled by it.
  A4  Pulse-compression reference rebuilt for this project's LFM (600 samples,
      k = 3e12 Hz/s, Fs = 60 MHz) using the official construction
      fft(fftshift(conj(flip(Sref)))) on the frame length.
  A5  The official loss code places tensors on "cuda" at import time; here the
      device is a parameter. Loss formulas are unchanged.
"""
from pathlib import Path
import hashlib
import importlib
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "CVEDNet_official"
OFFICIAL_COMMIT = "1f1dbf38269ef805d4d6924624776d2306c26fa3"
FS = 60e6
N = 1800
L_TX = 600
K_LFM = 3e12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official_sources():
    return {str(p.relative_to(OFFICIAL)).replace("\\", "/"): sha(p)
            for p in sorted(OFFICIAL.rglob("*.py"))}


def _stub_profiling_imports():
    added = []
    for name in ("ptflops", "pandas"):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
        if name == "ptflops":
            def _unavailable(*a, **k):
                raise RuntimeError("ptflops stub: profiling only")
            stub.get_model_complexity_info = _unavailable
        sys.modules[name] = stub
        added.append(name)
    return added


def load_official_class():
    """Import models/CVEDNet.py unchanged under a private package name."""
    stubs = _stub_profiling_imports()
    clash = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "models" or k.startswith("models.")}
    sys.path.insert(0, str(OFFICIAL))
    try:
        module = importlib.import_module("models.CVEDNet")
    finally:
        sys.path.remove(str(OFFICIAL))
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            sys.modules["cvednet_official_" + k] = sys.modules.pop(k)
        sys.modules.update(clash)
        for name in stubs:  # keep stubs out of other libraries' import checks
            sys.modules.pop(name, None)
    return module.CVEDNet


def build_model(device):
    cls = load_official_class()
    model = cls().to(device)  # official defaults A=512,B=16,C=128,D=512,E=8,F=3
    return model


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- A3 scaling
def pulse_scale(x):
    """x: complex tensor [B, N]; returns real [B, 1] input RMS (no labels)."""
    return torch.sqrt(torch.mean(x.real ** 2 + x.imag ** 2, dim=-1, keepdim=True)).clamp_min(1e-30)


def forward_complex(model, x):
    """x: complex64 [B, N] already scaled. Returns (real, imag) each [B, 1, N].

    Uses the official [B, 2, N] real/imag input format (CVEDNet.py __main__);
    Encoder_Block converts it to complex exactly as for a complex input.
    """
    return model(torch.stack((x.real, x.imag), dim=1))


@torch.no_grad()
def infer_numpy(model, z, device, batch=64):
    """z: complex ndarray [..., N] in physical units -> complex128 same shape."""
    shape = z.shape
    flat = torch.from_numpy(np.ascontiguousarray(z.reshape(-1, shape[-1]))).to(torch.complex64)
    out = []
    for i in range(0, flat.shape[0], batch):
        x = flat[i:i + batch].to(device)
        s = pulse_scale(x)
        re, im = forward_complex(model, x / s)
        y = torch.complex(re.squeeze(1), im.squeeze(1)) * s
        out.append(y.to(torch.complex128).cpu())
    return torch.cat(out).numpy().reshape(shape)


# ---------------------------------------------------------------- A4/A5 loss
def pc_reference(device, n=N):
    t = (np.arange(L_TX) + 0.0) / FS - 5e-6
    tx = np.exp(1j * np.pi * K_LFM * t * t)
    sref = np.zeros(n, np.complex128)
    start = (n - L_TX) // 2
    sref[start:start + L_TX] = tx
    s = torch.from_numpy(sref).to(torch.complex64)
    spec = torch.fft.fft(torch.fft.fftshift(torch.conj(torch.flip(s, dims=(-1,)))))
    return spec.to(device)


class PulseCompressionLoss(torch.nn.Module):
    """Formula of official sub/pc_loss1.py::pulse_compression_loss (l1 0.1, l2 1)."""

    def __init__(self, device, n=N):
        super().__init__()
        self.l1_factor = 0.1
        self.l2_factor = 1
        self.register_buffer("Sref", pc_reference(device, n))

    def forward(self, output_signal, gt_signal):
        if output_signal.dim() == 3:
            output_signal = output_signal.squeeze(1)
        if gt_signal.dim() == 3:
            gt_signal = gt_signal.squeeze(1)
        x_pc_t = abs(torch.fft.fftshift(torch.fft.ifft(torch.fft.fft(torch.fft.fftshift(output_signal)) * self.Sref))).flatten()
        y_pc_t = abs(torch.fft.fftshift(torch.fft.ifft(torch.fft.fft(torch.fft.fftshift(gt_signal)) * self.Sref))).flatten()
        x_db = 20 * torch.log10(torch.clip(x_pc_t / torch.max(x_pc_t), 1e-8, None))
        y_db = 20 * torch.log10(torch.clip(y_pc_t / torch.max(y_pc_t), 1e-8, None))
        l1_loss = F.l1_loss(y_db, x_db)
        l2_loss = torch.norm(y_pc_t - x_pc_t, p="fro") / torch.norm(y_pc_t, p="fro")
        return l2_loss * self.l2_factor + l1_loss * self.l1_factor


def official_total_loss(output, gt, pc_loss):
    """train.py: MSE(real)+MSE(imag)+PC_loss(complex)."""
    recon = F.mse_loss(output[0], gt.real) + F.mse_loss(output[1], gt.imag)
    pc = pc_loss(output[0] + 1j * output[1], gt.real + 1j * gt.imag)
    return recon + pc, recon, pc


def si_snr(est, ref, eps=1e-8):
    """torchmetrics ScaleInvariantSignalNoiseRatio (zero-mean), per row."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(((target ** 2).sum(-1) + eps) / ((noise ** 2).sum(-1) + eps))
