"""Single-stage K=8 real-masked shifted-LFM variable projection model."""
import math

import torch
from torch import nn
import torch.nn.functional as F

import vp_1103_common
from tools.unet_ablation.models import DoubleConv

N = 1800
FFT_LENGTH = 4096
K = 8
MIN_PRECISION = 1e-4


def real_parameters(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


class SmoothVPNet(nn.Module):
    """Predict correction j_hat; physical solver receives no truth metadata."""

    def __init__(self, coordinates="RAW_IQ", fixed_lambda=False):
        super().__init__()
        if coordinates not in ("RAW_IQ", "DECHIRP"):
            raise ValueError(coordinates)
        self.coordinates = coordinates
        self.fixed_lambda = fixed_lambda
        a, b, c = 18, 38, 68
        self.enc0 = DoubleConv(3, a)
        self.enc1 = DoubleConv(a, b)
        self.enc2 = DoubleConv(b, c)
        self.dec1 = DoubleConv(c + b, b)
        self.dec0 = DoubleConv(b + a, a)
        self.spatial_head = nn.Sequential(nn.Conv1d(a, 32, 3, padding=1),
                                          nn.SiLU(), nn.Conv1d(32, K * 2, 1))
        if not fixed_lambda:
            self.precision_head = nn.Sequential(nn.Linear(c, 64), nn.SiLU(), nn.Linear(64, K))
            # A deliberately high-damping start while retaining nonzero gradients.
            nn.init.normal_(self.precision_head[-1].weight, std=.01)
            nn.init.constant_(self.precision_head[-1].bias, 22.)
        t = torch.arange(600, dtype=torch.float64) / 60e6 - 5e-6
        pulse = torch.zeros(FFT_LENGTH, dtype=torch.complex128)
        pulse[:600] = torch.exp(1j * math.pi * (30e6 / 10e-6) * t.square())
        self.register_buffer("pulse_spectrum", torch.fft.fft(pulse))
        self.register_buffer("frequency", torch.fft.fftfreq(FFT_LENGTH, dtype=torch.float64))
        self.register_buffer("positions", torch.arange(N, dtype=torch.float32))
        carrier = math.pi / 1200 * torch.arange(N, dtype=torch.float64).square()
        self.register_buffer("dechirp", torch.exp(-1j * carrier))
        # The full, non-circular template fits for delays in [0,1200].
        anchors = torch.linspace(150, 1050, K // 2).repeat_interleave(2)
        self.register_buffer("delay_anchor_bias", -((torch.arange(1201)[None] - anchors[:, None]) / 180)**2)
        self.register_buffer("local_delay_offsets", torch.tensor([-1., 1.] * (K // 2)))

    def encode(self, iq):
        scale = iq.square().sum(1, keepdim=True).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        y = torch.complex(iq[:, 0], iq[:, 1]) / scale[:, 0]
        xf = torch.fft.fft(y, n=FFT_LENGTH)
        matched = torch.fft.ifft(xf * self.pulse_spectrum.to(xf).conj())[..., :N].abs()
        matched = matched / matched.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        coordinate = y * self.dechirp.to(y) if self.coordinates == "DECHIRP" else y
        x = torch.stack((coordinate.real, coordinate.imag, matched), 1)
        f0 = self.enc0(x)
        f1 = self.enc1(F.avg_pool1d(f0, 2))
        f2 = self.enc2(F.avg_pool1d(f1, 2))
        d1 = self.dec1(torch.cat((F.interpolate(f2, size=f1.shape[-1], mode="linear",
                                               align_corners=False), f1), 1))
        d0 = self.dec0(torch.cat((F.interpolate(d1, size=N, mode="linear",
                                               align_corners=False), f0), 1))
        return y, scale, f2, d0, matched

    def dictionary(self, delay, masks):
        # Zero padding to 4096 avoids circular wrap into the receive window.
        phase = -2j * math.pi * self.frequency[None, None] * delay.double()[..., None]
        shifted = torch.fft.ifft(self.pulse_spectrum[None, None] * torch.exp(phase))[..., :N]
        b = masks.double() * shifted
        b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min(1e-12)
        return b.transpose(1, 2)

    def factor(self, iq):
        y, scale, f2, d0, matched = self.encode(iq)
        spatial = self.spatial_head(d0)
        logits, raw_mask = spatial.chunk(2, 1)
        # Sigmoid expected position is a continuous absolute delay in [0,1200].
        # Fixed broad priors keep the eight initial atoms distinct.
        guided = logits[..., :1201] + self.delay_anchor_bias.to(logits)
        guided = guided + 12. * torch.log1p(matched[:, None, :1201])
        probabilities = torch.softmax(guided, -1)
        center = (probabilities * self.positions[:1201].to(logits)).sum(-1)
        # Paired atoms are initialized on adjacent delays around each soft peak.
        delay = center + self.local_delay_offsets
        masks = F.avg_pool1d(torch.sigmoid(raw_mask), 9, stride=1, padding=4, count_include_pad=False)
        b = self.dictionary(delay, masks)
        yd = y.to(torch.complex128)
        bh = b.mH
        projection = (bh @ yd.unsqueeze(-1)).squeeze(-1)
        # Median is a robust observed-only scale. It is piecewise differentiable.
        noise = (y.abs().square().median(dim=-1).values / math.log(2)).double().clamp_min(1e-12)
        rho = projection.abs().square() / noise[:, None]
        if self.fixed_lambda:
            log_precision = torch.zeros_like(rho)
            precision = torch.full_like(rho, MIN_PRECISION)
        else:
            log_precision = self.precision_head(f2.mean(-1)).double()
            exponent = log_precision + 2. * (math.log(9.) - torch.log1p(rho))
            # Smooth bounded exponent avoids overflow without a hard clamp.
            precision = MIN_PRECISION + torch.exp(30. * torch.tanh(exponent / 30.))
        gram = bh @ b
        normal = .5 * (gram + gram.mH) + torch.diag_embed(precision).to(torch.complex128)
        # Diagonal precision is strictly positive, including collinear columns.
        right = torch.linalg.solve(normal, bh)
        beta = (right @ yd.unsqueeze(-1)).squeeze(-1)
        correction = (b @ beta.unsqueeze(-1)).squeeze(-1) * scale[:, 0]
        return {"correction": correction, "b": b, "right": right, "delay": delay,
                "mask": masks, "log_precision": log_precision, "precision": precision,
                "rho": rho, "scale": scale, "gram": gram}

    def forward(self, iq):
        correction = self.factor(iq)["correction"].to(torch.complex64)
        return torch.stack((correction.real, correction.imag), 1)


def noise_gain(parts):
    right = parts["right"]
    return torch.einsum("bij,bji->b", parts["gram"], right @ right.mH).real


def operator_apply(parts, waveform):
    return (parts["b"] @ (parts["right"] @ waveform.to(torch.complex128).unsqueeze(-1))).squeeze(-1)
