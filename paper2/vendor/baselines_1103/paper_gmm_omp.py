"""Reproduction baseline for Liu et al., DSP 161 (2025) 105132.

Pipeline: two-component amplitude GMM fitted by EM, midpoint threshold for
jamming-free samples, full-record de-chirping, Fourier OMP reconstruction,
and inverse de-chirping.  The implementation uses only the received signal
and transmitter-known waveform parameters.  It does not consume oracle masks,
target delay, SNR, JSR, or clean components.

The paper's printed variance update omits the responsibility multiplier in
the numerator.  This implementation follows the standard weighted EM update,
which is consistent with its stated maximum-likelihood derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .physical import to_2channel, to_complex


@dataclass(frozen=True)
class PaperGMMOMPConfig:
    sample_rate_hz: float = 60.0e6
    chirp_rate_hz_s: float = 3.0e12
    gmm_components: int = 2
    gmm_max_iterations: int = 100
    gmm_tolerance: float = 1.0e-6
    gmm_variance_floor: float = 1.0e-6
    minimum_clean_samples: int = 8
    omp_sparsity: int = 8
    omp_relative_tolerance: float = 1.0e-3
    omp_ridge: float = 1.0e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if not math.isfinite(self.chirp_rate_hz_s) or self.chirp_rate_hz_s <= 0.0:
            raise ValueError("chirp_rate_hz_s must be finite and positive")
        if self.gmm_components != 2:
            raise ValueError("the paper baseline currently requires K=2")
        if self.gmm_max_iterations < 1 or self.omp_sparsity < 1:
            raise ValueError("iteration counts and sparsity must be positive")
        if self.minimum_clean_samples < 1:
            raise ValueError("minimum_clean_samples must be positive")
        if self.gmm_tolerance <= 0.0 or self.gmm_variance_floor <= 0.0:
            raise ValueError("GMM numerical floors must be positive")
        if self.omp_relative_tolerance < 0.0 or self.omp_ridge < 0.0:
            raise ValueError("OMP tolerance and ridge must be non-negative")


def fit_two_component_gmm(
    amplitude: Tensor,
    *,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6,
    variance_floor: float = 1.0e-6,
) -> dict[str, Tensor]:
    """Fit independent 1-D two-component GMMs for a batch ``[B,L]``."""

    if amplitude.ndim != 2 or not torch.is_floating_point(amplitude):
        raise ValueError("amplitude must be floating [B, L]")
    if not torch.isfinite(amplitude).all() or (amplitude < 0.0).any():
        raise ValueError("amplitude must be finite and non-negative")
    quantiles = torch.quantile(
        amplitude, torch.tensor([0.25, 0.75], device=amplitude.device), dim=1
    ).movedim(0, 1)
    means = quantiles
    global_variance = amplitude.var(dim=1, unbiased=False).clamp_min(variance_floor)
    variances = global_variance[:, None].expand(-1, 2).clone()
    mixing = amplitude.new_full((amplitude.shape[0], 2), 0.5)
    log_two_pi = math.log(2.0 * math.pi)
    iterations = 0
    for iteration in range(int(max_iterations)):
        centered = amplitude[:, :, None] - means[:, None, :]
        log_component = (
            mixing.clamp_min(1.0e-12).log()[:, None, :]
            - 0.5
            * (
                log_two_pi
                + variances.clamp_min(variance_floor).log()[:, None, :]
                + centered.square() / variances[:, None, :].clamp_min(variance_floor)
            )
        )
        responsibility = torch.softmax(log_component, dim=-1)
        effective_count = responsibility.sum(dim=1).clamp_min(1.0e-8)
        new_mixing = effective_count / float(amplitude.shape[1])
        new_means = (responsibility * amplitude[:, :, None]).sum(dim=1) / effective_count
        centered_new = amplitude[:, :, None] - new_means[:, None, :]
        new_variances = (
            responsibility * centered_new.square()
        ).sum(dim=1) / effective_count
        new_variances = new_variances.clamp_min(variance_floor)
        change = (new_means - means).abs().amax()
        means, variances, mixing = new_means, new_variances, new_mixing
        iterations = iteration + 1
        if float(change) <= float(tolerance):
            break
    order = means.argsort(dim=1)
    means = means.gather(1, order)
    variances = variances.gather(1, order)
    mixing = mixing.gather(1, order)
    threshold = means.mean(dim=1)
    return {
        "means": means,
        "variances": variances,
        "mixing": mixing,
        "threshold": threshold,
        "iterations": amplitude.new_full((amplitude.shape[0],), float(iterations)),
    }


def paper_jamming_free_mask(amplitude: Tensor, threshold: Tensor) -> Tensor:
    """Implement paper equations (30)-(32) for batched amplitudes."""

    if amplitude.ndim != 2 or threshold.shape != (amplitude.shape[0],):
        raise ValueError("amplitude must be [B,L] and threshold [B]")
    dirty = amplitude >= threshold[:, None]
    positions = torch.arange(amplitude.shape[1], device=amplitude.device)[None, :]
    any_dirty = dirty.any(dim=1)
    first = torch.where(dirty, positions, amplitude.shape[1]).amin(dim=1)
    last = torch.where(dirty, positions, -1).amax(dim=1)
    span = (positions >= first[:, None]) & (positions <= last[:, None])
    span = torch.where(any_dirty[:, None], span, torch.ones_like(span))
    return (~dirty) & span


def fourier_omp_reconstruct(
    dechirped: Tensor,
    observed_mask: Tensor,
    *,
    sparsity: int,
    relative_tolerance: float,
    ridge: float,
    minimum_samples: int,
) -> tuple[Tensor, Tensor]:
    """Recover a Fourier-sparse complex signal from observed time samples."""

    if dechirped.ndim != 2 or observed_mask.shape != dechirped.shape:
        raise ValueError("dechirped and mask must have shape [B,L]")
    if not torch.is_complex(dechirped) or observed_mask.dtype != torch.bool:
        raise TypeError("dechirped must be complex and mask must be bool")
    batch, length = dechirped.shape
    reconstructed = torch.zeros_like(dechirped)
    selected_count = torch.zeros(batch, device=dechirped.device, dtype=torch.long)
    all_positions = torch.arange(length, device=dechirped.device, dtype=dechirped.real.dtype)
    normalization = math.sqrt(float(length))
    for sample in range(batch):
        indices = observed_mask[sample].nonzero(as_tuple=False).flatten()
        if indices.numel() < minimum_samples:
            continue
        values = dechirped[sample, indices]
        initial_norm = values.abs().square().sum().sqrt().clamp_min(1.0e-12)
        residual = values.clone()
        selected: list[int] = []
        coefficients = values.new_zeros(0)
        for _ in range(min(int(sparsity), int(indices.numel()), length)):
            residual_full = torch.zeros(length, device=values.device, dtype=values.dtype)
            residual_full[indices] = residual
            correlations = torch.fft.fft(residual_full, norm="ortho").abs()
            if selected:
                correlations[torch.tensor(selected, device=values.device)] = -1.0
            atom = int(correlations.argmax())
            selected.append(atom)
            frequency = torch.tensor(selected, device=values.device, dtype=indices.dtype)
            phase = (
                2.0
                * torch.pi
                * indices[:, None].to(dechirped.real.dtype)
                * frequency[None, :].to(dechirped.real.dtype)
                / float(length)
            )
            design = torch.polar(torch.ones_like(phase), phase) / normalization
            if ridge > 0.0:
                gram = design.conj().transpose(0, 1) @ design
                rhs = design.conj().transpose(0, 1) @ values
                gram = gram + float(ridge) * torch.eye(
                    len(selected), device=values.device, dtype=values.dtype
                )
                coefficients = torch.linalg.solve(gram, rhs)
            else:
                coefficients = torch.linalg.lstsq(design, values).solution
            residual = values - design @ coefficients
            if float(residual.abs().square().sum().sqrt() / initial_norm) <= relative_tolerance:
                break
        spectrum = torch.zeros(length, device=values.device, dtype=values.dtype)
        spectrum[torch.tensor(selected, device=values.device)] = coefficients
        reconstructed[sample] = torch.fft.ifft(spectrum, norm="ortho")
        selected_count[sample] = len(selected)
    return reconstructed, selected_count


class PaperGMMOMPBaseline(nn.Module):
    """Deployable GMM + de-chirp + OMP paper reproduction baseline."""

    def __init__(self, config: PaperGMMOMPConfig | None = None) -> None:
        super().__init__()
        self.config = config or PaperGMMOMPConfig()

    def forward(self, y: Tensor, reference_waveform: Tensor | None = None) -> dict[str, Tensor]:
        del reference_waveform  # waveform parameters are explicit in the frozen config.
        if y.ndim != 3 or y.shape[1] != 2 or not torch.is_floating_point(y):
            raise ValueError("y must have shape [B,2,L] and floating dtype")
        z = to_complex(y)
        amplitude = z.abs()
        gmm = fit_two_component_gmm(
            amplitude,
            max_iterations=self.config.gmm_max_iterations,
            tolerance=self.config.gmm_tolerance,
            variance_floor=self.config.gmm_variance_floor,
        )
        clean_mask = paper_jamming_free_mask(amplitude, gmm["threshold"])
        length = y.shape[-1]
        time = torch.arange(length, device=y.device, dtype=y.dtype) / float(
            self.config.sample_rate_hz
        )
        phase = torch.pi * float(self.config.chirp_rate_hz_s) * time.square()
        local_oscillator = torch.polar(torch.ones_like(phase), phase)
        dechirped = z * local_oscillator.conj()[None, :]
        reconstructed_dechirped, selected_count = fourier_omp_reconstruct(
            dechirped,
            clean_mask,
            sparsity=self.config.omp_sparsity,
            relative_tolerance=self.config.omp_relative_tolerance,
            ridge=self.config.omp_ridge,
            minimum_samples=self.config.minimum_clean_samples,
        )
        reconstructed = reconstructed_dechirped * local_oscillator[None, :]
        # Equation (43) reconstructs only the first-to-last-jamming receive
        # window.  Avoid inventing a full-PRI target outside that window.
        span = clean_mask | (amplitude >= gmm["threshold"][:, None])
        reconstructed = reconstructed * span
        s_hat = to_2channel(reconstructed, channel_dim=1).to(dtype=y.dtype)
        reliability = clean_mask[:, None, :].to(dtype=y.dtype)
        return {
            "u_hat": s_hat,
            "s_hat": s_hat,
            "reliability": reliability,
            "estimated_reliability": reliability,
            "gmm_threshold": gmm["threshold"],
            "gmm_means": gmm["means"],
            "gmm_variances": gmm["variances"],
            "gmm_mixing": gmm["mixing"],
            "gmm_iterations": gmm["iterations"],
            "selected_frequency_count": selected_count,
            "jamming_free_fraction": clean_mask.float().mean(dim=1),
        }


__all__ = [
    "PaperGMMOMPBaseline",
    "PaperGMMOMPConfig",
    "fit_two_component_gmm",
    "fourier_omp_reconstruct",
    "paper_jamming_free_mask",
]
