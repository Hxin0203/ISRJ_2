"""Differentiable matched-filter tail objective for the fixed U-Net screen."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

FS_HZ = 60e6
BW_HZ = 30e6
PULSE_WIDTH_S = 10e-6
PULSE_SAMPLES = 600
N_RANGE = 1201
N_FFT = 4096
CFAR_VALID_RANGE_START = 36
CFAR_VALID_RANGE_STOP = 1165
PFA = 1e-6
N_CFAR_REFERENCE = 436
TOP_K = 8
SMOOTH_TAU_DB = 3.0
TAIL_WEIGHT = 0.05


def _matched_filter_rd(x: torch.Tensor) -> torch.Tensor:
    """Pulse-compress [record, 8 pulses, fast-time], then slow-time FFT."""
    if x.ndim != 3 or x.shape[1] != 8 or x.shape[-1] != 1800:
        raise ValueError(f"expected [records,8,1800], got {tuple(x.shape)}")
    t = torch.arange(PULSE_SAMPLES, dtype=x.real.dtype, device=x.device) / FS_HZ - PULSE_WIDTH_S / 2
    ref = torch.exp(1j * math.pi * (BW_HZ / PULSE_WIDTH_S) * t.square())
    xf = torch.fft.fft(x, n=N_FFT, dim=-1)
    matched = torch.fft.ifft(xf * torch.conj(torch.fft.fft(ref, n=N_FFT)), dim=-1)[..., :N_RANGE]
    return torch.fft.fft(matched, dim=1)


def matched_filter_tail_loss(correction: torch.Tensor, jammer: torch.Tensor,
                             noise: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize a smooth top-eight RD tail in H0 jammer-only residuals.

    ``correction`` is the predicted C(x), flattened in parent/pulse/state order
    as [records*8,4,2,1800]. The target-absent state-2 residual is ``j-C(j+e)``.
    Thermal noise e supplies a fixed per-record reference power. The tail is a
    loss proxy for the idealized 436-cell CA-CFAR threshold, not CFAR itself.
    """
    if correction.ndim != 5 or correction.shape[1:4] != (8, 4, 2) or correction.shape[-1] != 1800:
        raise ValueError(f"expected [records,8,4,2,1800], got {tuple(correction.shape)}")
    records = correction.shape[0]
    if jammer.shape != (records * 8, 2, 1800) or noise.shape != jammer.shape:
        raise ValueError("jammer/noise IQ arrays must match the eight-pulse records")
    j = torch.complex(jammer[:, 0], jammer[:, 1]).reshape(records, 8, 1800)
    e = torch.complex(noise[:, 0], noise[:, 1]).reshape(records, 8, 1800)
    c_j = torch.complex(correction[:, :, 2, 0], correction[:, :, 2, 1])
    residual = j - c_j
    residual_rd = _matched_filter_rd(residual)
    noise_rd = _matched_filter_rd(e)
    residual_power = residual_rd.abs().square()
    noise_power = noise_rd.abs().square().mean(dim=(1, 2)).clamp_min(1e-20)
    ratio_db = 10.0 * torch.log10(residual_power.clamp_min(1e-20) / noise_power[:, None, None])

    alpha = N_CFAR_REFERENCE * (PFA ** (-1.0 / N_CFAR_REFERENCE) - 1.0)
    threshold_db = 10.0 * math.log10(alpha)
    # Match radar_eval.ca_cfar: range cells whose reference window crosses a
    # range boundary are excluded from detection scoring (36:1165).
    valid_ratio_db = ratio_db[:, :, CFAR_VALID_RANGE_START:CFAR_VALID_RANGE_STOP]
    tail_db = torch.topk(valid_ratio_db.flatten(1), k=TOP_K, dim=1).values
    z = (tail_db - threshold_db) / SMOOTH_TAU_DB
    smooth_excess = (SMOOTH_TAU_DB * F.softplus(z) - SMOOTH_TAU_DB * math.log(2.0)).clamp_min(0.0)
    # Normalize the linear hinge by the fixed CFAR threshold in dB. This keeps
    # the added term on a controlled scale during early training and avoids a
    # quadratic penalty that could swamp target-preservation terms.
    loss = smooth_excess.mean() / threshold_db
    diagnostics = {
        "tail_loss": loss.detach(),
        "tail_peak_db_mean": tail_db[:, 0].mean().detach(),
        "tail_top8_db_mean": tail_db.mean().detach(),
        "idealized_threshold_db": correction.new_tensor(threshold_db).detach(),
    }
    return loss, diagnostics
