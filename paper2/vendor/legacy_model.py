"""Modular reliability-guided selective complex-waveform reconstruction.

The deployable graph has four explicit parts::

    observed IQ -> fixed dechirp -> reliability estimator -> restoration U-Net
                -> selective write-back -> inverse dechirp

Reliability means *absence of ISRJ contamination*, not target-existence
probability.  The restoration network predicts a complex correction rather
than regenerating the entire waveform.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PositionMode = Literal["none", "sinusoidal", "rope", "relative_bias"]
GateMode = Literal["none", "multiplicative", "residual_correction"]
PropagationMode = Literal["static", "learnable"]


@dataclass(frozen=True)
class RGSSRConfig:
    signal_length: int = 1800
    sample_rate_hz: float = 60.0e6
    pulse_width_s: float = 10.0e-6
    bandwidth_hz: float = 30.0e6
    reliability_channels: tuple[int, int, int] = (32, 64, 128)
    restoration_channels: tuple[int, int, int] = (32, 64, 128)
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ffn_ratio: float = 4.0
    dropout: float = 0.1
    position_mode: PositionMode = "rope"
    max_relative_distance: int = 256
    use_physics_features: bool = True
    use_reliability_input: bool = True
    use_selective_writeback: bool = True
    gate_mode: GateMode = "residual_correction"
    use_reliability_features: bool = True
    propagation_mode: PropagationMode = "static"
    detach_reliability_for_restoration: bool = False
    reliability_floor: float = 0.0

    def validate(self) -> None:
        if self.signal_length < 16:
            raise ValueError("signal_length must be at least 16")
        if min(self.sample_rate_hz, self.pulse_width_s, self.bandwidth_hz) <= 0:
            raise ValueError("radar parameters must be positive")
        if len(self.reliability_channels) != 3 or len(self.restoration_channels) != 3:
            raise ValueError("exactly three feature scales are required")
        if self.reliability_channels[-1] % self.transformer_heads:
            raise ValueError("Transformer dimension must be divisible by heads")
        head_dim = self.reliability_channels[-1] // self.transformer_heads
        if self.position_mode == "rope" and head_dim % 2:
            raise ValueError("RoPE requires an even per-head dimension")
        if self.position_mode not in ("none", "sinusoidal", "rope", "relative_bias"):
            raise ValueError(f"unknown position mode {self.position_mode}")
        if self.gate_mode not in ("none", "multiplicative", "residual_correction"):
            raise ValueError(f"unknown gate mode {self.gate_mode}")
        if self.propagation_mode not in ("static", "learnable"):
            raise ValueError(f"unknown propagation mode {self.propagation_mode}")
        if not 0.0 <= self.reliability_floor < 1.0:
            raise ValueError("reliability_floor must lie in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_iq(x: Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("expected complex IQ tensor [batch,2,samples]")
    if not torch.isfinite(x).all():
        raise ValueError("non-finite waveform input")


class FixedDechirp(nn.Module):
    """Known full-receive-window LFM phase rotation and its exact inverse."""

    def __init__(self, config: RGSSRConfig) -> None:
        super().__init__()
        time = torch.arange(config.signal_length, dtype=torch.float64) / config.sample_rate_hz
        chirp_rate = config.bandwidth_hz / config.pulse_width_s
        phase = math.pi * chirp_rate * time.square()
        reference = torch.stack((phase.cos(), phase.sin())).float()
        self.register_buffer("reference", reference, persistent=True)

    @staticmethod
    def multiply(a: Tensor, b: Tensor, *, conjugate_b: bool = False) -> Tensor:
        ar, ai = a[:, 0], a[:, 1]
        br, bi = b[:, 0], b[:, 1]
        if conjugate_b:
            return torch.stack((ar * br + ai * bi, ai * br - ar * bi), dim=1)
        return torch.stack((ar * br - ai * bi, ar * bi + ai * br), dim=1)

    def _reference(self, x: Tensor) -> Tensor:
        if x.shape[-1] > self.reference.shape[-1]:
            raise ValueError("waveform exceeds frozen receive-window reference")
        return self.reference[:, : x.shape[-1]].to(x).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        _validate_iq(x)
        return self.multiply(x, self._reference(x), conjugate_b=True)

    def inverse(self, z: Tensor) -> Tensor:
        _validate_iq(z)
        return self.multiply(z, self._reference(z))


def _same_avg(x: Tensor, kernel: int) -> Tensor:
    left = (kernel - 1) // 2
    right = kernel // 2
    return F.avg_pool1d(F.pad(x, (left, right), mode="replicate"), kernel, stride=1)


class PhysicsFeatureExtractor(nn.Module):
    """Observed-only, distance-robust local consistency features.

    Features are normalized residual frequency, phase coherence, phase
    instability, local amplitude mean, amplitude coefficient of variation,
    and neighboring residual-frequency change.  No oracle delay is used.
    """

    names = (
        "normalized_residual_frequency",
        "phase_coherence",
        "phase_instability",
        "amplitude_mean",
        "amplitude_cv",
        "neighbor_frequency_change",
    )

    def __init__(self, window: int = 9) -> None:
        super().__init__()
        if window < 3 or window % 2 == 0:
            raise ValueError("physics window must be odd and >=3")
        self.window = window

    def forward(self, z: Tensor) -> Tensor:
        _validate_iq(z)
        real, imag = z[:, :1], z[:, 1:2]
        prev_real = F.pad(real[..., :-1], (1, 0), mode="replicate")
        prev_imag = F.pad(imag[..., :-1], (1, 0), mode="replicate")
        cross_real = real * prev_real + imag * prev_imag
        cross_imag = imag * prev_real - real * prev_imag
        phase_step = torch.atan2(cross_imag, cross_real + 1e-8)
        local_frequency = _same_avg(phase_step, self.window)
        center = local_frequency.mean(-1, keepdim=True)
        scale = local_frequency.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized_frequency = torch.tanh((local_frequency - center) / scale)
        coherence_real = _same_avg(torch.cos(phase_step), self.window)
        coherence_imag = _same_avg(torch.sin(phase_step), self.window)
        coherence = torch.sqrt(coherence_real.square() + coherence_imag.square() + 1e-8)
        instability = _same_avg(1.0 - torch.cos(phase_step - local_frequency), self.window)
        amplitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        amplitude_mean = _same_avg(amplitude, self.window)
        amplitude_second = _same_avg(amplitude.square(), self.window)
        amplitude_std = (amplitude_second - amplitude_mean.square()).clamp_min(0.0).sqrt()
        amplitude_cv = amplitude_std / amplitude_mean.clamp_min(1e-4)
        amplitude_mean_norm = amplitude_mean / amplitude_mean.mean(-1, keepdim=True).clamp_min(1e-4)
        frequency_change = (local_frequency - F.pad(local_frequency[..., :-1], (1, 0), mode="replicate")).abs()
        frequency_change = torch.tanh(frequency_change / scale)
        return torch.cat(
            (
                normalized_frequency,
                coherence,
                instability,
                torch.tanh(amplitude_mean_norm),
                torch.tanh(amplitude_cv),
                frequency_change,
            ),
            dim=1,
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.projection = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.body(x) + self.projection(x))


def sinusoidal_encoding(length: int, dimension: int, reference: Tensor) -> Tensor:
    position = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
    div = torch.exp(
        torch.arange(0, dimension, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    value = torch.zeros(length, dimension, device=reference.device, dtype=torch.float32)
    value[:, 0::2] = torch.sin(position * div)
    value[:, 1::2] = torch.cos(position * div)
    return value.to(dtype=reference.dtype).unsqueeze(0)


class PositionAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, mode: PositionMode, dropout: float, max_distance: int) -> None:
        super().__init__()
        self.dimension, self.heads, self.mode = dimension, heads, mode
        self.head_dim = dimension // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.out = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.max_distance = max_distance
        if mode == "rope":
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("rope_inv", inv, persistent=True)
        else:
            self.register_buffer("rope_inv", torch.empty(0), persistent=False)
        if mode == "relative_bias":
            self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_distance + 1))
            nn.init.normal_(self.relative_bias, std=0.01)
        else:
            self.register_parameter("relative_bias", None)

    def _rope(self, x: Tensor) -> Tensor:
        position = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angle = torch.outer(position, self.rope_inv.float())
        cos, sin = angle.cos().to(x)[None, None], angle.sin().to(x)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = (item.transpose(1, 2) for item in qkv.unbind(2))
        if self.mode == "rope":
            q, k = self._rope(q), self._rope(k)
        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.mode == "relative_bias":
            index = torch.arange(length, device=x.device)
            relative = (index[:, None] - index[None, :]).clamp(-self.max_distance, self.max_distance)
            relative = relative + self.max_distance
            score = score + self.relative_bias[:, relative].to(score).unsqueeze(0)
        probability = torch.softmax(score.float(), -1).to(score)
        probability = self.dropout(probability)
        value = torch.matmul(probability, v).transpose(1, 2).contiguous().view(batch, length, self.dimension)
        return self.out(value)


class TransformerBlock(nn.Module):
    def __init__(self, config: RGSSRConfig) -> None:
        super().__init__()
        dimension = config.reliability_channels[-1]
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = PositionAttention(
            dimension, config.transformer_heads, config.position_mode,
            config.dropout, config.max_relative_distance,
        )
        self.norm2 = nn.LayerNorm(dimension)
        hidden = int(dimension * config.transformer_ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(hidden, dimension), nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ReliabilityEstimator(nn.Module):
    """Local CNN + global position-aware Transformer + dense decoder."""

    def __init__(self, config: RGSSRConfig) -> None:
        super().__init__()
        c0, c1, c2 = config.reliability_channels
        self.config = config
        self.physics = PhysicsFeatureExtractor()
        self.stem = ConvBlock(2 + (6 if config.use_physics_features else 0), c0)
        self.down1 = ConvBlock(c0, c1, stride=2)
        self.down2 = ConvBlock(c1, c2, stride=2)
        self.transformer = nn.ModuleList(TransformerBlock(config) for _ in range(config.transformer_layers))
        self.transformer_norm = nn.LayerNorm(c2)
        self.up1 = ConvBlock(c2 + c1, c1)
        self.up0 = ConvBlock(c1 + c0, c0)
        self.reliability_head = nn.Conv1d(c0, 1, 1)
        self.boundary_head = nn.Conv1d(c0, 1, 1)

    def forward(self, z: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], Tensor, Tensor]:
        physical = self.physics(z)
        source = torch.cat((z, physical), dim=1) if self.config.use_physics_features else z
        f0 = self.stem(source)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        tokens = f2.transpose(1, 2)
        if self.config.position_mode == "sinusoidal":
            tokens = tokens + sinusoidal_encoding(tokens.shape[1], tokens.shape[2], tokens)
        for block in self.transformer:
            tokens = block(tokens)
        f2 = self.transformer_norm(tokens).transpose(1, 2)
        up1 = F.interpolate(f2, size=f1.shape[-1], mode="linear", align_corners=False)
        f1_out = self.up1(torch.cat((up1, f1), dim=1))
        up0 = F.interpolate(f1_out, size=f0.shape[-1], mode="linear", align_corners=False)
        f0_out = self.up0(torch.cat((up0, f0), dim=1))
        reliability = torch.sigmoid(self.reliability_head(f0_out))
        if self.config.reliability_floor:
            reliability = self.config.reliability_floor + (1.0 - self.config.reliability_floor) * reliability
        boundary = torch.sigmoid(self.boundary_head(f0_out))
        return reliability, (f0_out, f1_out, f2), boundary, physical


class ConfidenceGate(nn.Module):
    """Learned feature gate; this G is deliberately distinct from reliability R."""

    def __init__(self, channels: int, rel_feature_channels: int, mode: GateMode, use_rel_features: bool) -> None:
        super().__init__()
        self.mode = mode
        self.use_rel_features = use_rel_features
        extra = rel_feature_channels if use_rel_features else 0
        self.gate = nn.Conv1d(channels + 1 + extra, channels, 1)
        self.correction = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.GELU())
        self.rel_adapter = nn.Conv1d(rel_feature_channels, rel_feature_channels, 1) if use_rel_features else nn.Identity()

    def forward(self, feature: Tensor, reliability: Tensor, rel_feature: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.mode == "none":
            return feature, torch.ones_like(feature)
        parts = [feature, reliability]
        if self.use_rel_features:
            if rel_feature is None:
                raise ValueError("reliability features required by configured gate")
            parts.append(self.rel_adapter(rel_feature))
        gate = torch.sigmoid(self.gate(torch.cat(parts, dim=1)))
        if self.mode == "multiplicative":
            return gate * feature, gate
        correction = self.correction(feature)
        return feature + (1.0 - reliability) * gate * correction, gate


class ReliabilityPropagator(nn.Module):
    def __init__(self, feature_channels: int, rel_feature_channels: int) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv1d(1 + feature_channels + rel_feature_channels, feature_channels, 3, padding=1),
            nn.GELU(), nn.Conv1d(feature_channels, 1, 1),
        )

    def forward(self, reliability: Tensor, feature: Tensor, rel_feature: Tensor) -> Tensor:
        resized = F.interpolate(reliability, size=feature.shape[-1], mode="linear", align_corners=False)
        return torch.sigmoid(self.update(torch.cat((resized, feature, rel_feature), dim=1)))


class ReliabilityGuidedUNet1D(nn.Module):
    """Multi-scale complex correction network with reliability side-stream."""

    def __init__(self, config: RGSSRConfig) -> None:
        super().__init__()
        self.config = config
        c0, c1, c2 = config.restoration_channels
        r0, r1, r2 = config.reliability_channels
        in_channels = 2 + int(config.use_reliability_input)
        self.enc0 = ConvBlock(in_channels, c0)
        self.enc1 = ConvBlock(c0, c1, stride=2)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.gate0 = ConfidenceGate(c0, r0, config.gate_mode, config.use_reliability_features)
        self.gate1 = ConfidenceGate(c1, r1, config.gate_mode, config.use_reliability_features)
        self.gate2 = ConfidenceGate(c2, r2, config.gate_mode, config.use_reliability_features)
        self.decode1 = ConvBlock(c2 + c1, c1)
        self.decode0 = ConvBlock(c1 + c0, c0)
        self.out = nn.Conv1d(c0, 2, 1)
        self.prop1 = ReliabilityPropagator(c1, r1)
        self.prop2 = ReliabilityPropagator(c2, r2)

    def _reliabilities(self, r: Tensor, f1: Tensor, f2: Tensor, fr: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        r0 = r
        if self.config.propagation_mode == "static":
            r1 = F.interpolate(r0, size=f1.shape[-1], mode="linear", align_corners=False)
            r2 = F.interpolate(r0, size=f2.shape[-1], mode="linear", align_corners=False)
        else:
            r1 = self.prop1(r0, f1, fr[1])
            r2 = self.prop2(r1, f2, fr[2])
        return r0, r1, r2

    def forward(self, z: Tensor, reliability: Tensor, rel_features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        source = torch.cat((z, reliability), dim=1) if self.config.use_reliability_input else z
        f0 = self.enc0(source)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        r0, r1, r2 = self._reliabilities(reliability, f1, f2, rel_features)
        g0, gate0 = self.gate0(f0, r0, rel_features[0] if self.config.use_reliability_features else None)
        g1, gate1 = self.gate1(f1, r1, rel_features[1] if self.config.use_reliability_features else None)
        g2, gate2 = self.gate2(f2, r2, rel_features[2] if self.config.use_reliability_features else None)
        d1 = self.decode1(torch.cat((F.interpolate(g2, size=g1.shape[-1], mode="linear", align_corners=False), g1), dim=1))
        d0 = self.decode0(torch.cat((F.interpolate(d1, size=g0.shape[-1], mode="linear", align_corners=False), g0), dim=1))
        return self.out(d0), {
            "gate_full": gate0, "gate_half": gate1, "gate_quarter": gate2,
            "reliability_half": r1, "reliability_quarter": r2,
        }


class SelectiveWriteback(nn.Module):
    """Structural modification-permission contract: z+(1-R)*delta."""

    def forward(self, z: Tensor, delta: Tensor, reliability: Tensor) -> Tensor:
        if z.shape != delta.shape or reliability.shape != z[:, :1].shape:
            raise ValueError("selective write-back shape mismatch")
        return z + (1.0 - reliability) * delta


class ReliabilityGuidedSelectiveReconstructor(nn.Module):
    """Deployment model accepting observed I/Q and known waveform constants only."""

    def __init__(self, config: RGSSRConfig | None = None) -> None:
        super().__init__()
        self.config = config or RGSSRConfig()
        self.config.validate()
        self.dechirp = FixedDechirp(self.config)
        self.reliability_estimator = ReliabilityEstimator(self.config)
        self.restorer = ReliabilityGuidedUNet1D(self.config)
        self.selective_writeback = SelectiveWriteback()

    def forward(self, y: Tensor) -> dict[str, Tensor]:
        _validate_iq(y)
        z = self.dechirp(y)
        reliability, rel_features, boundary, physics = self.reliability_estimator(z)
        if self.config.detach_reliability_for_restoration:
            r_restore = reliability.detach()
            fr_restore = tuple(item.detach() for item in rel_features)
        else:
            r_restore, fr_restore = reliability, rel_features
        delta, diagnostics = self.restorer(z, r_restore, fr_restore)
        z_out = (
            self.selective_writeback(z, delta, reliability)
            if self.config.use_selective_writeback
            else z + delta
        )
        u_hat = self.dechirp.inverse(z_out)
        result = {
            "u_hat": u_hat,
            "j_hat": y - u_hat,
            "z_observed": z,
            "z_out": z_out,
            "delta_z": delta,
            "reliability": reliability,
            "boundary": boundary,
            "physics_features": physics,
            "reliability_feature_full": rel_features[0],
            "reliability_feature_half": rel_features[1],
            "reliability_feature_quarter": rel_features[2],
        }
        result.update(diagnostics)
        return result


__all__ = [
    "FixedDechirp", "PhysicsFeatureExtractor", "RGSSRConfig",
    "ReliabilityEstimator", "ReliabilityGuidedSelectiveReconstructor",
    "ReliabilityGuidedUNet1D", "SelectiveWriteback",
]
