"""Change only PlainUNet's normalization; keep all trainable tensors identical."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tools.unet_ablation.models import PlainUNet


def observed_scale(y):
    """Robust complex-noise RMS proxy, not an oracle noise measurement.

    Unitary DFT white complex Gaussian noise has exponential power, whose
    population median is log(2) times its mean. Finite windows and interrupted
    chirps leak into the fixed outer quarter-band; record this contamination.
    The relative floor preserves scale equivariance away from the 1e-8 floor.
    No labels, SNR/JNR metadata, neighboring pulses or fitted parameters enter.
    """
    z = torch.complex(y[:, 0], y[:, 1])
    frequencies = torch.fft.fftfreq(y.shape[-1], device=y.device)
    power = torch.fft.fft(z, norm="ortho").abs().square()
    selected = power[:, frequencies.abs() > 0.375]
    noise = (selected.median(dim=-1).values / math.log(2)).clamp_min(0).sqrt()
    rms = y.square().sum(1).mean(-1).sqrt()
    return torch.maximum(noise, rms * 1e-4).clamp_min(1e-8)[:, None, None]


class NoiseReferenceUNet(PlainUNet):
    def forward(self, y):
        scale = observed_scale(y)
        f0 = self.enc0(y / scale)
        f1 = self.enc1(F.max_pool1d(f0, 2))
        f2 = self.enc2(F.max_pool1d(f1, 2))
        d1 = self.dec1(torch.cat((F.interpolate(f2, size=f1.shape[-1], mode="linear", align_corners=False), f1), 1))
        d0 = self.dec0(torch.cat((F.interpolate(d1, size=f0.shape[-1], mode="linear", align_corners=False), f0), 1))
        return self.out(d0) * scale
