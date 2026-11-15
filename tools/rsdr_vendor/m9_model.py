"""M9 phase-safe and configurable action-gate models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .d1_model import SemanticActionConfig
from .model import ReliabilityEstimator, ReliabilityGuidedUNet1D


GateInput = Literal["r", "yr", "dr", "ydr", "ydrm"]
GateInit = Literal["default", "zero_weight_bias"]


@dataclass(frozen=True)
class M9Config:
    name: str
    phase_augment: bool
    gate_input: GateInput
    gate_init: GateInit
    gate_bias: float | None
    zero_delta: bool
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
            gradient_mode="semantic_stopgrad",
            signal_length=self.signal_length,
            sample_rate_hz=self.sample_rate_hz,
            pulse_width_s=self.pulse_width_s,
            bandwidth_hz=self.bandwidth_hz,
            reliability_channels=self.reliability_channels,
            restoration_channels=self.restoration_channels,
            transformer_layers=self.transformer_layers,
            transformer_heads=self.transformer_heads,
            transformer_ffn_ratio=self.transformer_ffn_ratio,
            dropout=self.dropout,
        ).base()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["reliability_channels"] = list(self.reliability_channels)
        value["restoration_channels"] = list(self.restoration_channels)
        return value


def gate_channels(mode: GateInput) -> int:
    return {"r": 1, "yr": 3, "dr": 3, "ydr": 5, "ydrm": 6}[mode]


class ConfigurableActionGate(nn.Module):
    def __init__(self, mode: GateInput, initialization: GateInit, bias: float | None, hidden: int = 32) -> None:
        super().__init__()
        self.mode = mode
        self.network = nn.Sequential(
            nn.Conv1d(gate_channels(mode), hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )
        if initialization == "zero_weight_bias":
            if bias is None:
                raise ValueError("zero_weight_bias requires a numeric bias")
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(self.network[-1].bias, float(bias))
        elif initialization != "default":
            raise ValueError(f"unknown gate initialization {initialization}")

    def inputs(self, y: Tensor, delta: Tensor, reliability: Tensor) -> Tensor:
        magnitude = delta.square().sum(1, keepdim=True).add(1e-12).sqrt()
        parts = {
            "r": (reliability,),
            "yr": (y, reliability),
            "dr": (delta, reliability),
            "ydr": (y, delta, reliability),
            "ydrm": (y, delta, reliability, magnitude),
        }[self.mode]
        return torch.cat(parts, dim=1)

    def forward(self, y: Tensor, delta: Tensor, reliability: Tensor) -> Tensor:
        return torch.sigmoid(self.network(self.inputs(y, delta, reliability)))


class M9RGSSR(nn.Module):
    """Raw-domain RoPE reliability + plain U-Net + configurable action gate."""

    def __init__(self, config: M9Config) -> None:
        super().__init__()
        self.config = config
        base = config.base()
        base = type(base)(**{
            **base.to_dict(),
            "reliability_channels": tuple(base.reliability_channels),
            "restoration_channels": tuple(base.restoration_channels),
            "use_reliability_input": False,
        })
        self.reliability_estimator = ReliabilityEstimator(base)
        self.restorer = ReliabilityGuidedUNet1D(base)
        self.action_gate = ConfigurableActionGate(config.gate_input, config.gate_init, config.gate_bias)
        if config.zero_delta:
            nn.init.zeros_(self.restorer.out.weight)
            if self.restorer.out.bias is not None:
                nn.init.zeros_(self.restorer.out.bias)

    def forward(self, y: Tensor) -> dict[str, Tensor]:
        reliability, features, boundary, physics = self.reliability_estimator(y)
        action_r = reliability.detach()
        detached_features = tuple(value.detach() for value in features)
        delta, diagnostics = self.restorer(y, action_r, detached_features)
        gate = self.action_gate(y, delta, action_r)
        action = (1.0 - action_r) * gate
        u_hat = y + action * delta
        result = {
            "u_hat": u_hat,
            "j_hat": y - u_hat,
            "z_observed": y,
            "z_out": u_hat,
            "delta_z": delta,
            "reliability": reliability,
            "boundary": boundary,
            "physics_features": physics,
            "action_gate": gate,
            "action": action,
        }
        result.update(diagnostics)
        return result


def rotate_iq(value: Tensor, phase: Tensor) -> Tensor:
    """Apply one global complex phase per batch item."""
    if value.ndim != 3 or value.shape[1] != 2:
        raise ValueError("expected [batch,2,length] I/Q tensor")
    phase = phase.to(value).view(-1, 1)
    cosine, sine = phase.cos(), phase.sin()
    real, imag = value[:, 0], value[:, 1]
    return torch.stack((cosine * real - sine * imag, sine * real + cosine * imag), dim=1)


__all__ = ["ConfigurableActionGate", "M9Config", "M9RGSSR", "rotate_iq"]
