"""Minimal RoPE RGSSR with explicit semantic reliability and action gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .model import (
    FixedDechirp,
    RGSSRConfig,
    ReliabilityEstimator,
    ReliabilityGuidedUNet1D,
)


GradientMode = Literal["shared_gradient", "semantic_stopgrad"]


@dataclass(frozen=True)
class SemanticActionConfig:
    gradient_mode: GradientMode = "semantic_stopgrad"
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

    def base(self) -> RGSSRConfig:
        if self.gradient_mode not in ("shared_gradient", "semantic_stopgrad"):
            raise ValueError(f"unknown gradient mode {self.gradient_mode}")
        return RGSSRConfig(
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
            position_mode="rope",
            use_physics_features=False,
            use_reliability_input=True,
            use_selective_writeback=False,
            gate_mode="none",
            use_reliability_features=False,
            propagation_mode="static",
            detach_reliability_for_restoration=False,
        )

    def to_dict(self) -> dict:
        value = dict(self.__dict__)
        value["reliability_channels"] = list(self.reliability_channels)
        value["restoration_channels"] = list(self.restoration_channels)
        return value


class ActionGate(nn.Module):
    """Learn how much of delta-z to apply inside semantic permission."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(5, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, z: Tensor, delta: Tensor, reliability: Tensor) -> Tensor:
        return torch.sigmoid(self.network(torch.cat((z, delta, reliability), dim=1)))


class SemanticActionRGSSR(nn.Module):
    """Same graph and parameters; only endpoint-gradient routing differs.

    shared_gradient lets restoration losses alter semantic R.  In
    semantic_stopgrad, every restoration-path use of R is detached, so only
    semantic reliability supervision can update the estimator.
    """

    def __init__(self, config: SemanticActionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticActionConfig()
        base = self.config.base()
        base.validate()
        self.dechirp = FixedDechirp(base)
        self.reliability_estimator = ReliabilityEstimator(base)
        self.restorer = ReliabilityGuidedUNet1D(base)
        self.action_gate = ActionGate()

    def forward(self, y: Tensor) -> dict[str, Tensor]:
        z = self.dechirp(y)
        reliability, features, boundary, physics = self.reliability_estimator(z)
        if self.config.gradient_mode == "semantic_stopgrad":
            action_r = reliability.detach()
            action_features = tuple(item.detach() for item in features)
        else:
            action_r, action_features = reliability, features
        delta, diagnostics = self.restorer(z, action_r, action_features)
        gate = self.action_gate(z, delta, action_r)
        action = (1.0 - action_r) * gate
        z_out = z + action * delta
        u_hat = self.dechirp.inverse(z_out)
        result = {
            "u_hat": u_hat, "j_hat": y - u_hat,
            "z_observed": z, "z_out": z_out, "delta_z": delta,
            "reliability": reliability, "boundary": boundary,
            "physics_features": physics, "action_gate": gate, "action": action,
        }
        result.update(diagnostics)
        return result

    def restoration_only_loss(self, y: Tensor) -> Tensor:
        """Diagnostic loss used to verify gradient isolation, not training."""
        output = self(y)
        return output["z_out"].square().mean()


__all__ = ["ActionGate", "SemanticActionConfig", "SemanticActionRGSSR"]
