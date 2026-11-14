"""Minimal I/Q conversion helpers for the vendored GMM-OMP reproduction."""
import torch


def to_complex(x):
    if torch.is_complex(x):
        return x
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("expected [batch, 2, samples]")
    return torch.complex(x[:, 0], x[:, 1])


def to_2channel(x, channel_dim=1):
    if not torch.is_complex(x):
        raise TypeError("expected complex tensor")
    return torch.stack((x.real, x.imag), dim=channel_dim)
