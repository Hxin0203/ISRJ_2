"""Reliability-label construction for the V2 ISRJ data model.

The V2 observation model is

    y = u + j,  u = s + epsilon,

where ``s`` is the noiseless target, ``epsilon`` is receiver noise, ``u`` is
the jammer-free received echo, and ``j`` is the jammer contribution.  The
main recovery target is ``u``.  Reliability therefore describes whether the
observation is useful with respect to *jamming*, rather than whether a target
is present.

No target delay or oracle noise gate is used to build these labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ReliabilityLabels:
    """Training-only precision surrogate and binary clean-state labels.

    ``z`` is the Bernoulli target for the predicted clean probability
    ``r=P(z=1|y)``.  ``oracle_precision_surrogate`` is a deterministic
    function of the simulated jammer and receiver noise; it is never a
    probability target or an inference input.
    """

    oracle_precision_surrogate: np.ndarray
    z: np.ndarray
    noise_power: np.ndarray

    @property
    def reliability(self) -> np.ndarray:
        """Backward-compatible archive-writer alias for legacy V2 files."""

        return self.oracle_precision_surrogate


def estimate_noise_power(epsilon: np.ndarray, axis: int = -1) -> np.ndarray:
    """Estimate complex-noise power ``E[|epsilon|^2]``.

    Parameters
    ----------
    epsilon:
        Complex receiver noise, either ``[L]`` or a batched array ending in
        the fast-time dimension.
    axis:
        Dimension over which the mean power is computed.
    """

    epsilon = np.asarray(epsilon)
    if not np.iscomplexobj(epsilon):
        raise TypeError("epsilon must be a complex-valued array")
    power = np.mean(np.abs(epsilon) ** 2, axis=axis)
    return np.asarray(power, dtype=np.float64)


def _broadcast_power(noise_power: np.ndarray | float, target: np.ndarray) -> np.ndarray:
    power = np.asarray(noise_power, dtype=np.float64)
    if np.any(~np.isfinite(power)) or np.any(power <= 0):
        raise ValueError("noise_power must contain finite positive values")

    # Per-example power [B] should broadcast against a jammer array [B, L].
    while power.ndim < target.ndim:
        power = np.expand_dims(power, axis=-1)
    return power


def oracle_precision_surrogate_from_components(
    j: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    noise_power: Optional[np.ndarray | float] = None,
    floor: float = 0.0,
    exponent: float = 1.0,
) -> np.ndarray:
    """Build the training-only observation-precision surrogate ``q``.

    The surrogate is

    ``q = sigma_e^2 / (sigma_e^2 + |j|^2)``.

    It equals one when no jammer is present and approaches zero as jammer
    power dominates receiver noise.  Because it uses the simulated component
    ``j``, it may weight supervised losses but must not supervise the predicted
    Bernoulli probability directly or enter model inference.
    """

    j = np.asarray(j)
    if not np.iscomplexobj(j):
        raise TypeError("j must be a complex-valued array")
    if noise_power is None:
        if epsilon is None:
            raise ValueError("either epsilon or noise_power must be provided")
        noise_power = estimate_noise_power(epsilon)
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor must be in [0, 1)")
    if exponent <= 0:
        raise ValueError("exponent must be positive")

    jammer_power = np.abs(j).astype(np.float64) ** 2
    sigma2 = _broadcast_power(noise_power, jammer_power)
    precision = sigma2 / (sigma2 + jammer_power)
    if exponent != 1.0:
        precision = precision**exponent
    if floor > 0.0:
        precision = floor + (1.0 - floor) * precision
    return np.clip(precision, 0.0, 1.0).astype(np.float32)


def soft_reliability_from_components(
    j: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    noise_power: Optional[np.ndarray | float] = None,
    floor: float = 0.0,
    exponent: float = 1.0,
) -> np.ndarray:
    """Compatibility alias for :func:`oracle_precision_surrogate_from_components`.

    New code should use the explicit name.  The saved NPZ field remains named
    ``reliability`` so existing governed assets retain their exact byte-level
    identity.
    """

    return oracle_precision_surrogate_from_components(
        j,
        epsilon=epsilon,
        noise_power=noise_power,
        floor=floor,
        exponent=exponent,
    )


def clean_state_from_components(
    j: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    noise_power: Optional[np.ndarray | float] = None,
    threshold_db: float = 0.0,
) -> np.ndarray:
    """Build binary clean-state labels ``z``.

    ``z[n] = 1`` when instantaneous jammer power is no greater than
    ``threshold_db`` relative to receiver-noise power.  With the default
    0 dB threshold, jammer samples below the noise floor are treated as not
    harmful.  This definition is more useful than an exact nonzero-support
    test at floating-point segment boundaries.
    """

    j = np.asarray(j)
    if not np.iscomplexobj(j):
        raise TypeError("j must be a complex-valued array")
    if noise_power is None:
        if epsilon is None:
            raise ValueError("either epsilon or noise_power must be provided")
        noise_power = estimate_noise_power(epsilon)

    jammer_power = np.abs(j).astype(np.float64) ** 2
    sigma2 = _broadcast_power(noise_power, jammer_power)
    threshold = sigma2 * (10.0 ** (float(threshold_db) / 10.0))
    return (jammer_power <= threshold).astype(np.float32)


def build_reliability_labels(
    j: np.ndarray,
    *,
    epsilon: Optional[np.ndarray] = None,
    noise_power: Optional[np.ndarray | float] = None,
    clean_threshold_db: float = 0.0,
    reliability_floor: float = 0.0,
    reliability_exponent: float = 1.0,
) -> ReliabilityLabels:
    """Build both V2 reliability targets from physical components."""

    if noise_power is None:
        if epsilon is None:
            raise ValueError("either epsilon or noise_power must be provided")
        noise_power = estimate_noise_power(epsilon)
    noise_power_array = np.asarray(noise_power, dtype=np.float64)

    precision_surrogate = oracle_precision_surrogate_from_components(
        j,
        noise_power=noise_power_array,
        floor=reliability_floor,
        exponent=reliability_exponent,
    )
    z = clean_state_from_components(
        j,
        noise_power=noise_power_array,
        threshold_db=clean_threshold_db,
    )
    return ReliabilityLabels(
        oracle_precision_surrogate=precision_surrogate,
        z=z,
        noise_power=noise_power_array,
    )


__all__ = [
    "ReliabilityLabels",
    "build_reliability_labels",
    "clean_state_from_components",
    "estimate_noise_power",
    "oracle_precision_surrogate_from_components",
    "soft_reliability_from_components",
]
