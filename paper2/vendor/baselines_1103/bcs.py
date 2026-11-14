"""Original-domain BCS reconstruction for ideal jammer-rejected samples.

The implementation follows Eqs. (14), (17)--(23) and Algorithm 1 of
Sha et al., Sensors 19(15):3279 (2019).  It solves the smooth-L1 MAP
problem by an outer iteratively reweighted fixed point and an inner
complex conjugate-gradient solve.  No OMP/LASSO/ISTA surrogate is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class BCSConfig:
    lambda_value: float
    tau: float = 1e-8
    outer_tolerance: float = 1e-2
    inner_tolerance: float = 1e-3
    outer_max_iterations: int = 40
    inner_max_iterations: int = 200


@dataclass
class BCSResult:
    theta: np.ndarray
    reconstructed_time: np.ndarray
    outer_iterations: int
    inner_iterations: int
    converged: bool
    objective: float
    runtime_seconds: float


def _validate(samples: np.ndarray, clean_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
    clean_mask = np.asarray(clean_mask, dtype=bool).reshape(-1)
    if samples.shape != clean_mask.shape:
        raise ValueError("samples and clean_mask must have identical one-dimensional shape")
    if not np.any(clean_mask):
        raise ValueError("clean_mask must contain at least one retained sample")
    if not np.isfinite(samples.real).all() or not np.isfinite(samples.imag).all():
        raise ValueError("samples must be finite")
    return samples, clean_mask


def _a(theta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.fft.ifft(theta, norm="ortho")[mask]


def _ah(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = np.zeros(mask.size, dtype=np.complex128)
    filled[mask] = values
    return np.fft.fft(filled, norm="ortho")


def _complex_cg(operator, rhs: np.ndarray, initial: np.ndarray, tolerance: float, max_iterations: int) -> tuple[np.ndarray, int]:
    x = np.asarray(initial, dtype=np.complex128).copy()
    residual = rhs - operator(x)
    direction = residual.copy()
    rr = float(np.vdot(residual, residual).real)
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-15)
    if np.sqrt(rr) <= tolerance * rhs_norm:
        return x, 0
    for iteration in range(1, max_iterations + 1):
        applied = operator(direction)
        denominator = float(np.vdot(direction, applied).real)
        if denominator <= 1e-24 or not np.isfinite(denominator):
            break
        alpha = rr / denominator
        x += alpha * direction
        residual -= alpha * applied
        rr_new = float(np.vdot(residual, residual).real)
        if np.sqrt(max(rr_new, 0.0)) <= tolerance * rhs_norm:
            return x, iteration
        beta = rr_new / max(rr, 1e-30)
        direction = residual + beta * direction
        rr = rr_new
    return x, iteration


def bcs_reconstruct(samples: np.ndarray, clean_mask: np.ndarray, config: BCSConfig) -> BCSResult:
    """Recover sparse range coefficients from ideal unjammed time samples."""
    samples, mask = _validate(samples, clean_mask)
    if config.lambda_value <= 0 or config.tau <= 0:
        raise ValueError("lambda_value and tau must be positive")
    started = time.perf_counter()
    retained = samples[mask]
    rhs = 2.0 * _ah(retained, mask)
    theta = _ah(retained, mask)  # Algorithm-1 zero-filled Fourier initialization.
    total_inner = 0
    converged = False
    outer_iteration = 0
    for outer_iteration in range(1, config.outer_max_iterations + 1):
        weights = 1.0 / np.sqrt(np.abs(theta) ** 2 + config.tau)

        def hessian(vector: np.ndarray) -> np.ndarray:
            return 2.0 * _ah(_a(vector, mask), mask) + config.lambda_value * weights * vector

        updated, inner = _complex_cg(
            hessian, rhs, theta, config.inner_tolerance, config.inner_max_iterations
        )
        total_inner += inner
        relative = float(np.linalg.norm(updated - theta) / max(np.linalg.norm(theta), 1e-15))
        theta = updated
        if relative <= config.outer_tolerance:
            converged = True
            break
    residual = retained - _a(theta, mask)
    objective = float(
        np.vdot(residual, residual).real
        + config.lambda_value * np.sum(np.sqrt(np.abs(theta) ** 2 + config.tau))
    )
    return BCSResult(
        theta=theta,
        reconstructed_time=np.fft.ifft(theta, norm="ortho"),
        outer_iterations=outer_iteration,
        inner_iterations=total_inner,
        converged=converged,
        objective=objective,
        runtime_seconds=time.perf_counter() - started,
    )


def select_lambda_gcv(
    scenes: list[tuple[np.ndarray, np.ndarray]],
    lambda_grid: list[float],
    *,
    tau: float = 1e-8,
) -> tuple[float, list[dict[str, float]]]:
    """Freeze lambda on DEV using the paper's influence-matrix GCV.

    The trace of K=A H^-1 2A^H is estimated with four deterministic
    Hutchinson probes.  The paper does not publish the lambda search interval;
    the finite grid remains an explicit DEV-only reproduction choice.
    """
    diagnostics: list[dict[str, float]] = []
    for value in lambda_grid:
        scores = []
        convergence = []
        for scene_index, (samples, mask) in enumerate(scenes):
            configuration = BCSConfig(lambda_value=float(value), tau=tau)
            result = bcs_reconstruct(samples, mask, configuration)
            observed = np.asarray(samples)[mask]
            residual = observed - _a(result.theta, np.asarray(mask, dtype=bool))
            mask_array = np.asarray(mask, dtype=bool)
            weights = 1.0 / np.sqrt(np.abs(result.theta) ** 2 + tau)

            def hessian(vector: np.ndarray) -> np.ndarray:
                return 2.0 * _ah(_a(vector, mask_array), mask_array) + float(value) * weights * vector

            rng = np.random.default_rng(91000 + 37 * scene_index + int(round(-np.log10(value) * 100)))
            trace_samples = []
            for _ in range(4):
                probe = rng.choice(np.asarray([-1.0, 1.0]), size=observed.size).astype(np.complex128)
                solution, _ = _complex_cg(
                    hessian,
                    2.0 * _ah(probe, mask_array),
                    np.zeros(mask_array.size, dtype=np.complex128),
                    3e-4,
                    300,
                )
                trace_samples.append(float(np.vdot(probe, _a(solution, mask_array)).real))
            effective_degrees = float(np.clip(np.mean(trace_samples), 0.0, observed.size - 1.0))
            denominator = max(observed.size - effective_degrees, 1.0)
            scores.append(float(np.vdot(residual, residual).real / denominator ** 2))
            convergence.append(float(result.converged))
        diagnostics.append({
            "lambda": float(value),
            "mean_gcv": float(np.mean(scores)),
            "convergence_rate": float(np.mean(convergence)),
        })
    diagnostics.sort(key=lambda item: item["mean_gcv"])
    return float(diagnostics[0]["lambda"]), diagnostics
