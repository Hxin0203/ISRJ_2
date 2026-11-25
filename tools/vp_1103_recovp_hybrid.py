"""ReCoVP-Hybrid: K=16 variable projection with an equivariant refiner.

This file is intentionally self-contained.  The Stage-1 equations and feature
heads follow the frozen ReCoVP FP32 implementation recovered from commit
7137814, while its real U-Net widths are narrowed to (13, 26, 44), and the
observation-only rectangular matched-filter anchor selector is extended to
K=16.  No historical source or checkpoint is imported or modified.

Input is RAW_IQ with shape [B, 2, 1800] (or a complex [B, 1800] tensor); the
reference LFM pulse is 600 samples.  For finite float32/complex64 input whose
per-pulse RMS is in [1e-12, 1e12], normalization uses the exact RMS with no
absolute floor.  The all-zero pulse is handled separately with scale 1 and
maps to an all-zero correction/restoration.  These are numerical-domain
conditions for the implementation; finite preflight samples do not prove a
mathematical universal quantification over floating-point inputs.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


T = 1800
PULSE_LENGTH = 600
FFT_LENGTH = 4096
ANCHOR_COUNT = 16
STAGE1_WIDTHS = (13, 26, 44)
STAGE2_WIDTHS = (9, 13, 17, 21)
MAX_EFFECTIVE_REAL_PARAMETERS = 59_700


def effective_real_parameters(module: nn.Module) -> int:
    """Count trainable complex scalars as two real parameters."""
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in module.parameters() if p.requires_grad)


def _complex_input(iq: torch.Tensor) -> torch.Tensor:
    if iq.is_complex():
        if iq.ndim != 2 or iq.shape[-1] != T:
            raise ValueError(f"complex input must have shape [B,{T}]")
        return iq.to(torch.complex64)
    if iq.ndim != 3 or iq.shape[1:] != (2, T):
        raise ValueError(f"RAW_IQ input must have shape [B,2,{T}]")
    return torch.complex(iq[:, 0].float(), iq[:, 1].float())


def _rms_scale(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exactly homogeneous RMS and normalized input; zero is explicit."""
    rms = y.abs().square().mean(-1, keepdim=True).sqrt()
    scale = torch.where(rms > 0, rms, torch.ones_like(rms))
    return scale, y / scale


def _complex_pool(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(F.avg_pool1d(x.real, 2), F.avg_pool1d(x.imag, 2))


def _complex_interpolate(x: torch.Tensor, length: int) -> torch.Tensor:
    return torch.complex(F.interpolate(x.real, size=length, mode="linear", align_corners=False),
                         F.interpolate(x.imag, size=length, mode="linear", align_corners=False))


def _lag_product(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(x[..., :1]), x[..., 1:] * x[..., :-1].conj()), -1)


class _DoubleConv(nn.Sequential):
    """Frozen Base block shape: Conv/GroupNorm/ReLU twice."""
    def __init__(self, incoming: int, outgoing: int):
        groups = min(8, outgoing)
        while outgoing % groups:
            groups -= 1
        super().__init__(
            nn.Conv1d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing), nn.ReLU())


def _rect_k16_anchors(x: torch.Tensor, matched_spectrum: torch.Tensor) -> torch.Tensor:
    """Frozen E0_rect peak selection with K=16 and six-sample separation."""
    with torch.no_grad():
        y = x.detach().to(torch.complex64)
        response = torch.fft.ifft(torch.fft.fft(y, n=FFT_LENGTH) *
                                  matched_spectrum.to(device=y.device, dtype=y.dtype).conj(),
                                  n=FFT_LENGTH)[..., :T].abs()
        local = response >= F.max_pool1d(response[:, None], 13, 1, 6)[:, 0]
        pos = torch.arange(T, device=x.device)[None]
        available = ((pos >= 2) & (pos <= T - 3)).expand_as(response).clone()
        selected: list[torch.Tensor] = []
        for _ in range(ANCHOR_COUNT):
            scores = response.masked_fill(~(available & local), -torch.inf)
            empty = ~torch.isfinite(scores).any(-1)
            if empty.any():
                scores = torch.where(empty[:, None], response.masked_fill(~available, -torch.inf), scores)
            index = scores.argmax(-1)
            selected.append(index)
            available &= (pos - index[:, None]).abs() >= 6
        return torch.stack(selected, -1).to(x.real.dtype)


class ReCoVPStage1(nn.Module):
    """Narrowed frozen-Base FP32 feature network and complex128 K=16 VP solve."""
    def __init__(self):
        super().__init__()
        a, b, c = STAGE1_WIDTHS
        self.enc0 = _DoubleConv(4, a)
        self.enc1 = _DoubleConv(a, b)
        self.enc2 = _DoubleConv(b, c)
        self.dec1 = _DoubleConv(c + b, b)
        self.dec0 = _DoubleConv(b + a, a)
        self.refine_head = nn.Sequential(nn.Linear(a + c + 2, 64), nn.SiLU(), nn.Linear(64, 1))
        self.window_head = nn.Sequential(nn.Conv1d(a + 9, 12, 3, padding=1), nn.SiLU(),
                                         nn.Conv1d(12, 1, 3, padding=1))
        self.shared_precision_head = nn.Sequential(nn.Linear(a + c + 4, 16), nn.SiLU(),
                                                    nn.Linear(16, 1))
        nn.init.normal_(self.refine_head[-1].weight, std=.005)
        nn.init.zeros_(self.refine_head[-1].bias)
        nn.init.normal_(self.shared_precision_head[-1].weight, std=.01)
        nn.init.constant_(self.shared_precision_head[-1].bias, 22.)

        t = torch.arange(PULSE_LENGTH, dtype=torch.float64) / 60e6 - 5e-6
        pulse = torch.zeros(FFT_LENGTH, dtype=torch.complex128)
        pulse[:PULSE_LENGTH] = torch.exp(1j * math.pi * 3e12 * t.square())
        pulse_spectrum = torch.fft.fft(pulse)
        unit_spectrum = pulse_spectrum / math.sqrt(PULSE_LENGTH)
        normalized_pulse = torch.fft.ifft(unit_spectrum)[:PULSE_LENGTH]
        matched_spectrum = torch.fft.fft(normalized_pulse, n=FFT_LENGTH)
        frequency = torch.fft.fftfreq(FFT_LENGTH, dtype=torch.float64)
        self.register_buffer("pulse_spectrum", pulse_spectrum)
        self.register_buffer("unit_spectrum", unit_spectrum)
        self.register_buffer("matched_spectrum", matched_spectrum)
        self.register_buffer("anchor_spectrum", pulse_spectrum)
        self.register_buffer("frequency", frequency)
        self.register_buffer("positions", torch.arange(T, dtype=torch.float32))

    def _invariant_encode(self, x: torch.Tensor):
        matched = torch.fft.ifft(torch.fft.fft(x, n=FFT_LENGTH) *
                                 self.matched_spectrum.to(device=x.device, dtype=x.dtype).conj(),
                                 n=FFT_LENGTH)[..., :T].abs()
        lag = _lag_product(x)
        glob = torch.stack((x.abs(), matched, lag.real, lag.imag), 1)
        f0 = self.enc0(glob.float())
        f1 = self.enc1(F.avg_pool1d(f0, 2))
        f2 = self.enc2(F.avg_pool1d(f1, 2))
        d1 = self.dec1(torch.cat((_complex_interpolate(torch.complex(f2, torch.zeros_like(f2)), f1.shape[-1]).real,
                                  f1), 1))
        features = self.dec0(torch.cat((F.interpolate(d1, size=T, mode="linear", align_corners=False),
                                        f0), 1))
        return f2, features, matched

    @staticmethod
    def _sample(features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        grid_x = positions / (T - 1) * 2 - 1
        grid = torch.stack((grid_x, torch.zeros_like(grid_x)), -1)[:, None]
        sampled = F.grid_sample(features[:, :, None], grid, align_corners=True, mode="bilinear")
        return sampled[:, :, 0].transpose(1, 2)

    def forward(self, iq: torch.Tensor, anchors_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        y = _complex_input(iq)
        scale, x = _rms_scale(y)
        f2, features, matched = self._invariant_encode(x)
        anchors = _rect_k16_anchors(x, self.anchor_spectrum.to(device=x.device))
        if anchors_override is not None:
            if anchors_override.shape != anchors.shape:
                raise ValueError(f"anchors_override must have shape [B,{ANCHOR_COUNT}]")
            anchors = anchors_override.to(device=x.device, dtype=x.real.dtype)
        local = self._sample(features, anchors)
        peak = self._sample(matched[:, None], anchors)
        shared = f2.mean(-1)[:, None].expand(-1, ANCHOR_COUNT, -1)
        position = (anchors / T)[..., None]
        delta = 2 * torch.tanh(self.refine_head(torch.cat((local, shared, position,
                                                            torch.log1p(peak)), -1))[..., 0])
        delay = anchors + delta
        phase = -2j * math.pi * self.frequency.to(device=x.device)[None, None] * delay.double()[..., None]
        template = torch.fft.ifft(self.unit_spectrum.to(device=x.device)[None, None] *
                                  torch.exp(phase), n=FFT_LENGTH)[..., :T]

        z = x[:, None].to(torch.complex128) * template.conj()
        channels: list[torch.Tensor] = []
        for length in (5, 15, 31):
            smoothed = torch.complex(
                F.avg_pool1d(z.real, length, 1, length // 2, count_include_pad=False),
                F.avg_pool1d(z.imag, length, 1, length // 2, count_include_pad=False))
            lag_s = _lag_product(smoothed)
            channels.extend((smoothed.abs(), lag_s.real, lag_s.imag))
        physical = torch.stack(channels, 2).float()
        broadcast = features[:, None].expand(-1, ANCHOR_COUNT, -1, -1)
        window_input = torch.cat((broadcast, physical), 2).flatten(0, 1)
        masks = torch.sigmoid(self.window_head(window_input)).reshape(-1, ANCHOR_COUNT, T)
        masks = F.avg_pool1d(masks, 3, 1, 1, count_include_pad=False)
        mask_peak = masks.amax(-1, keepdim=True)
        masks = masks / mask_peak.clamp_min(.05)
        atom = masks.double() * template
        atom_norm = torch.linalg.vector_norm(atom, dim=-1) + 1e-12
        basis = (atom / atom_norm[..., None]).transpose(1, 2)
        # x has unit RMS for every nonzero input; its robust variance is invariant.
        center = x.mean(-1, keepdim=True)
        for _ in range(3):
            weight = (x - center).abs().clamp_min(1e-12).reciprocal()
            center = (weight * x).sum(-1, keepdim=True) / weight.sum(-1, keepdim=True)
        sigma = ((x - center).abs().median(-1).values.square() / math.log(2)).clamp_min(1e-12)
        rho = (basis.mH @ x.to(torch.complex128)[..., None])[..., 0].abs().square() / sigma[:, None]
        context = torch.cat((shared, local, torch.log1p(rho).float()[..., None],
                             masks.mean(-1)[..., None], position, torch.log1p(peak)), -1)
        log_precision = self.shared_precision_head(context)[..., 0].double()
        exponent = log_precision + 2 * (math.log(9) - torch.log1p(rho))
        precision = 1e-4 + torch.exp(30 * torch.tanh(exponent / 30))
        basis = basis.to(torch.complex128)
        yd = x.to(torch.complex128)
        gram = basis.mH @ basis
        normal = .5 * (gram + gram.mH) + torch.diag_embed(precision).to(torch.complex128)
        right = torch.linalg.solve(normal, basis.mH)
        correction_norm = (basis @ (right @ yd[..., None]))[..., 0]
        correction = correction_norm.to(torch.complex64) * scale
        restored = y - correction
        return {"correction": correction, "restored": restored, "residual": restored,
                "scale": scale, "normalized_input": x, "anchors": anchors, "delay": delay,
                "dictionary": basis, "right": right, "gram": gram, "mask": masks,
                "precision": precision, "rho": rho}


class BiasFreeComplexConv1d(nn.Module):
    """Complex-linear 1-D convolution with no affine offset."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,
                              dilation=dilation, bias=False)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,
                              dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real(x.real) - self.imag(x.imag),
                             self.real(x.imag) + self.imag(x.real))


class EquivComplexGatedBlock(nn.Module):
    """One complex-linear map followed by a scale-invariant real radial gate."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv = BiasFreeComplexConv1d(in_channels, out_channels, 3, dilation)
        self.gate = nn.Conv1d(out_channels + 6, out_channels, 3,
                              padding=dilation, dilation=dilation)
        self.dilation = dilation

    def forward(self, z: torch.Tensor, invariant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        u = self.conv(z)
        # z and scale have the same positive degree; the smooth modulus ratio is invariant.
        scale_3d = scale.unsqueeze(-1)
        delta = 1e-7 * scale_3d
        mag = torch.sqrt(u.real.square() + u.imag.square() + delta.square())
        logmag = torch.log(mag / scale_3d + 1e-3)
        gate = torch.sigmoid(F.silu(self.gate(torch.cat((logmag, invariant), 1))))
        return gate * u


class _Stage2Level(nn.Module):
    def __init__(self, incoming: int, outgoing: int, dilation: int = 1, residual: bool = False):
        super().__init__()
        self.block = EquivComplexGatedBlock(incoming, outgoing, dilation)
        self.residual = residual
        self.skip = (BiasFreeComplexConv1d(incoming, outgoing, 1)
                     if residual and incoming != outgoing else None)

    def forward(self, z: torch.Tensor, inv: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        out = self.block(z, inv, scale)
        if self.residual:
            skip = z if self.skip is None else self.skip(z)
            out = out + skip
        return out


class EquivComplexUNetRefiner(nn.Module):
    """Explicit 3-level complex U-Net: widths 9/13/17/21, dilation 1/2/4."""
    def __init__(self):
        super().__init__()
        a, b, c, d = STAGE2_WIDTHS
        self.stem = EquivComplexGatedBlock(2, a)
        self.enc0 = _Stage2Level(a, a, residual=True)
        self.enc1 = _Stage2Level(a, b)
        self.enc2 = _Stage2Level(b, c)
        self.bottleneck2 = _Stage2Level(c, d, dilation=1, residual=True)
        self.bottleneck_d2 = _Stage2Level(d, d, dilation=2, residual=True)
        self.bottleneck_d4 = _Stage2Level(d, d, dilation=4, residual=True)
        self.dec2 = _Stage2Level(d + c, c)
        self.dec1 = _Stage2Level(c + b, b)
        self.dec0 = _Stage2Level(b + a, a)
        self.out = BiasFreeComplexConv1d(a, 1, kernel_size=1)
        # Input: three summaries per invariant feature plus two physical scalars.
        self.alpha = nn.Sequential(nn.Linear(20, 4), nn.SiLU(), nn.Linear(4, 1))
        nn.init.constant_(self.alpha[-1].bias, -3.0)
        nn.init.normal_(self.out.real.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.out.imag.weight, mean=0.0, std=1e-3)

    @staticmethod
    def invariant_features(residual: torch.Tensor, correction: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
        delta = 1e-7 * scale
        rmag = torch.sqrt(residual.real.square() + residual.imag.square() + delta.square())
        jmag = torch.sqrt(correction.real.square() + correction.imag.square() + delta.square())
        f1 = torch.log(rmag / scale + 1e-3)
        f2 = torch.log(jmag / scale + 1e-3)
        lag = residual[..., 1:] * residual[..., :-1].conj()
        lag = F.pad(lag, (1, 0)) / scale.square()
        cross = residual * correction.conj() / scale.square()
        return torch.stack((f1, f2, lag.real, lag.imag, cross.real, cross.imag), 1)

    @staticmethod
    def _stats(inv: torch.Tensor) -> torch.Tensor:
        mean = inv.mean(-1)
        std = inv.std(-1, unbiased=False)
        p95 = torch.quantile(inv, .95, dim=-1)
        return torch.cat((mean, std, p95), 1)

    def forward(self, residual: torch.Tensor, correction: torch.Tensor,
                original: torch.Tensor, scale: torch.Tensor,
                matched_spectrum: torch.Tensor) -> dict[str, torch.Tensor]:
        # Work in normalized coordinates. This keeps all learned gates invariant.
        r = residual / scale
        j = correction / scale
        inv = self.invariant_features(r, j, torch.ones_like(scale))
        z = torch.cat((r[:, None], j[:, None]), 1)
        x0 = self.stem(z, inv, torch.ones_like(scale))
        e0 = self.enc0(x0, inv, torch.ones_like(scale))
        inv1, inv2, inv3 = F.avg_pool1d(inv, 2), F.avg_pool1d(F.avg_pool1d(inv, 2), 2), F.avg_pool1d(F.avg_pool1d(F.avg_pool1d(inv, 2), 2), 2)
        e1 = self.enc1(_complex_pool(e0), inv1, torch.ones_like(scale))
        e2 = self.enc2(_complex_pool(e1), inv2, torch.ones_like(scale))
        b = self.bottleneck2(_complex_pool(e2), inv3, torch.ones_like(scale))
        b = self.bottleneck_d2(b, inv3, torch.ones_like(scale))
        b = self.bottleneck_d4(b, inv3, torch.ones_like(scale))
        d2 = self.dec2(torch.cat((_complex_interpolate(b, e2.shape[-1]), e2), 1), inv2, torch.ones_like(scale))
        d1 = self.dec1(torch.cat((_complex_interpolate(d2, e1.shape[-1]), e1), 1), inv1, torch.ones_like(scale))
        d0 = self.dec0(torch.cat((_complex_interpolate(d1, e0.shape[-1]), e0), 1), inv, torch.ones_like(scale))
        raw_delta = self.out(d0)[:, 0]

        y_energy = original.abs().square().sum(-1)
        j_energy = correction.abs().square().sum(-1)
        physical1 = torch.log(j_energy / (y_energy + 1e-12 * scale[:, 0].square()) + 1e-6)
        mf = torch.fft.ifft(torch.fft.fft(residual, n=FFT_LENGTH) *
                            matched_spectrum.to(device=residual.device).conj(),
                            n=FFT_LENGTH)[..., :T].abs()
        mf_mean = mf.square().mean(-1)
        physical2 = torch.log(mf.square().amax(-1) / (mf_mean + 1e-12 * scale[:, 0].square()) + 1e-6)
        alpha_in = torch.cat((self._stats(inv), physical1[:, None], physical2[:, None]), 1)
        alpha = torch.sigmoid(self.alpha(alpha_in.float()))
        delta = alpha * raw_delta * scale
        return {"delta": delta, "alpha": alpha, "invariant_features": inv}


class ReCoVPHybrid(nn.Module):
    """Full ReCoVP-Hybrid model; ``forward`` returns restored RAW_IQ [B,2,T]."""
    def __init__(self, enforce_parameter_budget: bool = True):
        super().__init__()
        self.stage1 = ReCoVPStage1()
        self.stage2 = EquivComplexUNetRefiner()
        self.enforce_parameter_budget = enforce_parameter_budget
        if enforce_parameter_budget:
            count = effective_real_parameters(self)
            if count > MAX_EFFECTIVE_REAL_PARAMETERS:
                raise ValueError(f"Hybrid has {count} effective real parameters; limit is "
                                 f"{MAX_EFFECTIVE_REAL_PARAMETERS}")

    def factor(self, iq: torch.Tensor, anchors_override: torch.Tensor | None = None) -> dict[str, Any]:
        y = _complex_input(iq)
        one = self.stage1(iq, anchors_override=anchors_override)
        refined = self.stage2(one["residual"], one["correction"], y,
                              one["scale"], self.stage1.matched_spectrum)
        final = one["residual"] - refined["delta"]
        return {**one, **refined, "restored": final, "stage1_restored": one["residual"],
                "stage2_delta": refined["delta"]}

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        restored = self.factor(iq)["restored"].to(torch.complex64)
        return torch.stack((restored.real, restored.imag), 1)

    def parameter_report(self) -> dict[str, int]:
        return {"stage1_effective_real_parameters": effective_real_parameters(self.stage1),
                "stage2_effective_real_parameters": effective_real_parameters(self.stage2),
                "total_effective_real_parameters": effective_real_parameters(self),
                "limit": MAX_EFFECTIVE_REAL_PARAMETERS}


def parameter_report() -> dict[str, int]:
    """Instantiate the model and return its auditable effective-parameter totals."""
    return ReCoVPHybrid().parameter_report()


__all__ = ["ANCHOR_COUNT", "T", "PULSE_LENGTH", "ReCoVPStage1",
           "BiasFreeComplexConv1d", "EquivComplexGatedBlock",
           "EquivComplexUNetRefiner", "ReCoVPHybrid", "effective_real_parameters",
           "parameter_report"]
