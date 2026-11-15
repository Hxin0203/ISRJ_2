"""All models return correction C(x); the restored waveform is x-C(x)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from paper2.network import CorrectionNet, training_loss

VARIANTS = ("unet", "no_dechirp", "no_rms", "no_pc", "no_damage")


class DoubleConv(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        groups = min(8, outgoing)
        while outgoing % groups:
            groups -= 1
        super().__init__(
            nn.Conv1d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing), nn.ReLU())


class PlainUNet(nn.Module):
    """Three-scale real-valued I/Q U-Net, no dechirp or local residual blocks.

    Widths are selected by parameter count only, with no validation scores.
    RMS normalization and correction output are retained as shared controls.
    All 59,714 parameters participate, exactly matching the original active core.
    """
    def __init__(self):
        super().__init__()
        a, b, c = 18, 42, 74
        self.enc0 = DoubleConv(2, a)
        self.enc1 = DoubleConv(a, b)
        self.enc2 = DoubleConv(b, c)
        self.dec1 = DoubleConv(c + b, b)
        self.dec0 = DoubleConv(b + a, a)
        self.out = nn.Conv1d(a, 2, 1)

    def forward(self, y):
        scale = y.square().sum(1, keepdim=True).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        f0 = self.enc0(y / scale)
        f1 = self.enc1(F.max_pool1d(f0, 2))
        f2 = self.enc2(F.max_pool1d(f1, 2))
        d1 = self.dec1(torch.cat((F.interpolate(f2, size=f1.shape[-1], mode="linear", align_corners=False), f1), 1))
        d0 = self.dec0(torch.cat((F.interpolate(d1, size=f0.shape[-1], mode="linear", align_corners=False), f0), 1))
        return self.out(d0) * scale


class StructuralAblation(CorrectionNet):
    """Keep the frozen original core, initialization and parameter registry."""
    def __init__(self, variant):
        super().__init__()
        self.variant = variant

    def forward(self, y):
        scale = (torch.ones_like(y[:, :1, :1]) if self.variant == "no_rms" else
                 y.square().sum(1, keepdim=True).mean(-1, keepdim=True).sqrt().clamp_min(1e-8))
        z = y / scale
        if self.variant != "no_dechirp":
            z = self.rotate(z)
        r = torch.ones_like(z[:, :1])
        delta, _ = self.core(z, r, (r, r, r))
        if self.variant != "no_dechirp":
            delta = self.rotate.inverse(delta)
        return delta * scale


def make_model(variant):
    if variant == "unet":
        return PlainUNet()
    if variant in ("no_dechirp", "no_rms"):
        return StructuralAblation(variant)
    if variant in ("full", "no_pc", "no_damage"):
        return CorrectionNet()
    raise ValueError(f"unregistered variant: {variant}")


def objective(c, s, j, e, variant):
    if variant not in (*VARIANTS, "full"):
        raise ValueError(variant)
    # Always retain all three diagnostics, including omitted terms.
    values = training_loss(c, s, j, e, "B4")
    pc_weight = 0.0 if variant == "no_pc" else 0.1
    damage_weight = 0.0 if variant == "no_damage" else 1.0
    return {**values, "total": values["endpoint"] + pc_weight * values["pc"] + damage_weight * values["damage"]}


def architecture(variant):
    with torch.random.fork_rng(devices=[]):
        model = make_model(variant)
        model(torch.ones(1, 2, 1800)).square().mean().backward()
        active = sum(p.numel() for p in model.parameters() if p.grad is not None)
        return {"variant": variant, "class": type(model).__name__,
                "parameters_total": sum(p.numel() for p in model.parameters()),
                "parameters_in_forward_graph": active, "module_repr": str(model),
                "state_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
                "output_contract": "correction C(x); restored = x-C(x)"}
