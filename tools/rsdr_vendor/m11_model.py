"""M11 internal confidence, attention bias, and uncertainty-action variants."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .d1_model import SemanticActionConfig
from .m9_model import ConfigurableActionGate
from .model import ConvBlock, PhysicsFeatureExtractor, ReliabilityEstimator, ReliabilityGuidedUNet1D


Mechanism = Literal[
    "output_gate_control", "internal_gated_conv", "confidence_biased_attention", "uncertainty_to_action"
]


@dataclass(frozen=True)
class M11Config:
    name: str
    mechanism: Mechanism
    phase_augment: bool = True
    gate_input: str = "ydr"
    gate_init: str = "default"
    gate_bias: float | None = None
    zero_delta: bool = False
    signal_length: int = 1800
    sample_rate_hz: float = 60e6
    pulse_width_s: float = 10e-6
    bandwidth_hz: float = 30e6
    reliability_channels: tuple[int, int, int] = (32, 64, 128)
    restoration_channels: tuple[int, int, int] = (32, 64, 128)
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ffn_ratio: float = 4.0
    dropout: float = 0.1

    def base(self):
        return SemanticActionConfig(
            signal_length=self.signal_length, sample_rate_hz=self.sample_rate_hz,
            pulse_width_s=self.pulse_width_s, bandwidth_hz=self.bandwidth_hz,
            reliability_channels=self.reliability_channels, restoration_channels=self.restoration_channels,
            transformer_layers=self.transformer_layers, transformer_heads=self.transformer_heads,
            transformer_ffn_ratio=self.transformer_ffn_ratio, dropout=self.dropout,
        ).base()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reliability_channels"] = list(self.reliability_channels)
        value["restoration_channels"] = list(self.restoration_channels)
        return value


class ConfidenceBiasedAttention(nn.Module):
    """RoPE attention with beta*log(local-clean-confidence) added to key logits."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dimension // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.out = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.tensor(1.0))
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv", inv, persistent=True)

    def _rope(self, x: Tensor) -> Tensor:
        position = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angle = torch.outer(position, self.rope_inv.float())
        cosine, sine = angle.cos().to(x)[None, None], angle.sin().to(x)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)

    def forward(self, tokens: Tensor, clean_prior: Tensor) -> Tensor:
        batch, length, dimension = tokens.shape
        qkv = self.qkv(tokens).view(batch, length, 3, self.heads, self.head_dim)
        q, k, value = (part.transpose(1, 2) for part in qkv.unbind(2))
        q, k = self._rope(q), self._rope(k)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_bias = self.beta.clamp(0.0, 4.0) * clean_prior.clamp_min(1e-5).log()
        logits = logits + key_bias[:, None, None, :]
        probability = torch.softmax(logits.float(), -1).to(logits)
        probability = self.dropout(probability)
        mixed = torch.matmul(probability, value).transpose(1, 2).contiguous().view(batch, length, dimension)
        return self.out(mixed)


class ConfidenceBiasedBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = ConfidenceBiasedAttention(dimension, heads, dropout)
        self.norm2 = nn.LayerNorm(dimension)
        hidden = int(dimension * ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dimension), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, clean_prior: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x), clean_prior)
        return x + self.ffn(self.norm2(x))


class ConfidenceBiasedReliabilityEstimator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        c0, c1, c2 = config.reliability_channels
        self.physics = PhysicsFeatureExtractor()
        self.stem = ConvBlock(2, c0)
        self.down1 = ConvBlock(c0, c1, stride=2)
        self.down2 = ConvBlock(c1, c2, stride=2)
        self.coarse_clean = nn.Conv1d(c2, 1, 1)
        self.transformer = nn.ModuleList(
            ConfidenceBiasedBlock(c2, config.transformer_heads, config.transformer_ffn_ratio, config.dropout)
            for _ in range(config.transformer_layers)
        )
        self.transformer_norm = nn.LayerNorm(c2)
        self.up1 = ConvBlock(c2 + c1, c1)
        self.up0 = ConvBlock(c1 + c0, c0)
        self.reliability_head = nn.Conv1d(c0, 1, 1)
        self.boundary_head = nn.Conv1d(c0, 1, 1)

    def forward(self, z: Tensor):
        physical = self.physics(z)
        f0 = self.stem(z)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        clean_prior = torch.sigmoid(self.coarse_clean(f2)).squeeze(1)
        tokens = f2.transpose(1, 2)
        for block in self.transformer:
            tokens = block(tokens, clean_prior)
        f2 = self.transformer_norm(tokens).transpose(1, 2)
        up1 = F.interpolate(f2, size=f1.shape[-1], mode="linear", align_corners=False)
        f1_out = self.up1(torch.cat((up1, f1), 1))
        up0 = F.interpolate(f1_out, size=f0.shape[-1], mode="linear", align_corners=False)
        f0_out = self.up0(torch.cat((up0, f0), 1))
        reliability = torch.sigmoid(self.reliability_head(f0_out))
        boundary = torch.sigmoid(self.boundary_head(f0_out))
        return reliability, (f0_out, f1_out, f2), boundary, physical


class UncertaintyActionGate(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.uncertainty = nn.Sequential(
            nn.Conv1d(5, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(6, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, y: Tensor, delta: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor]:
        source = torch.cat((y, delta, reliability), 1)
        log_variance = self.uncertainty(source).clamp(-8.0, 6.0)
        sigma = torch.exp(0.5 * log_variance).clamp_max(20.0) / 20.0
        gate = torch.sigmoid(self.gate(torch.cat((source, sigma), 1)))
        return gate, log_variance


class M11RGSSR(nn.Module):
    def __init__(self, config: M11Config) -> None:
        super().__init__()
        self.config = config
        base = config.base()
        internal = config.mechanism == "internal_gated_conv"
        base = type(base)(**{
            **base.to_dict(),
            "reliability_channels": tuple(base.reliability_channels),
            "restoration_channels": tuple(base.restoration_channels),
            "use_reliability_input": internal,
            "gate_mode": "residual_correction" if internal else "none",
            "use_reliability_features": internal,
        })
        self.reliability_estimator = (
            ConfidenceBiasedReliabilityEstimator(base)
            if config.mechanism == "confidence_biased_attention" else ReliabilityEstimator(base)
        )
        self.restorer = ReliabilityGuidedUNet1D(base)
        self.uncertainty_gate = UncertaintyActionGate() if config.mechanism == "uncertainty_to_action" else None
        self.action_gate = ConfigurableActionGate("ydr", "default", None)

    def forward(self, y: Tensor) -> dict[str, Tensor]:
        reliability, features, boundary, physics = self.reliability_estimator(y)
        action_r = reliability.detach()
        delta, diagnostics = self.restorer(y, action_r, tuple(feature.detach() for feature in features))
        if self.uncertainty_gate is None:
            gate = self.action_gate(y, delta, action_r)
            log_variance = torch.zeros_like(action_r)
        else:
            gate, log_variance = self.uncertainty_gate(y, delta, action_r)
        action = (1.0 - action_r) * gate
        u_hat = y + action * delta
        result = {
            "u_hat": u_hat, "j_hat": y - u_hat, "z_observed": y, "z_out": u_hat,
            "delta_z": delta, "reliability": reliability, "boundary": boundary,
            "physics_features": physics, "action_gate": gate, "action": action,
            "log_variance": log_variance,
        }
        result.update(diagnostics)
        return result


__all__ = ["M11Config", "M11RGSSR"]
