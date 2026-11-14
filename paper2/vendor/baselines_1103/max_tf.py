"""Max-TF ISRJ suppression following Chen et al. (IET RSN, 2019).

The source-fixed chain is Eq. (17)--(26): squared STFT, maximum over
frequency, global and per-segment quarter-peak thresholds, inversion of the
jammer mask, FFT-derived normalized band-pass response, smoothing, filtering.
Discrete tie handling and the practical smoothing detector are explicitly
reproduction choices documented in ``SOURCE_AND_FORMULA_AUDIT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complexfloating]
RealArray = NDArray[np.floating]


@dataclass(frozen=True)
class MaxTFConfig:
    sample_rate_hz: float = 50e6
    window_length: int = 65
    hop_length: int = 16
    n_fft: int = 128
    min_clean_frames: int = 2
    smoothing_bins: int = 5
    peak_quantile: float = 0.70

    def __post_init__(self) -> None:
        if self.window_length < 3 or self.window_length > self.n_fft:
            raise ValueError("require 3 <= window_length <= n_fft")
        if self.hop_length < 1 or self.hop_length > self.window_length:
            raise ValueError("invalid hop_length")
        if self.min_clean_frames < 1 or self.smoothing_bins < 1:
            raise ValueError("run and smoothing lengths must be positive")
        if not 0.0 < self.peak_quantile < 1.0:
            raise ValueError("peak_quantile must be in (0, 1)")


@dataclass
class MaxTFResult:
    dechirped: ComplexArray
    stft: ComplexArray
    tf_energy: RealArray
    max_tf: RealArray
    jammer_frame_mask: NDArray[np.bool_]
    clean_sample_mask: NDArray[np.bool_]
    raw_filter: RealArray
    smoothed_filter: RealArray
    pc_before: ComplexArray
    pc_after: ComplexArray
    recovered_waveform: ComplexArray


def _as_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim != 2:
        raise ValueError("expected [samples] or [batch, samples]")
    return x, False


def dechirp(received: ComplexArray, reference: ComplexArray) -> ComplexArray:
    """Paper Eq. (5): conjugated received waveform times LFM reference."""

    y = np.asarray(received)
    ref = np.asarray(reference)
    if y.shape[-1] != ref.shape[-1]:
        raise ValueError("received/reference lengths differ")
    return np.conj(y) * ref


def compute_stft(signal: ComplexArray, config: MaxTFConfig) -> tuple[ComplexArray, RealArray]:
    """Vectorized complex STFT and Eq. (17) squared TF energy."""

    x, squeezed = _as_batch(np.asarray(signal, dtype=np.complex64))
    half = config.window_length // 2
    padded = np.pad(x, ((0, 0), (half, half)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, config.window_length, axis=-1
    )[:, :: config.hop_length, :]
    window = np.hamming(config.window_length).astype(np.float32)
    spectrum = np.fft.fft(frames * window, n=config.n_fft, axis=-1).astype(np.complex64)
    spectrum = np.transpose(spectrum, (0, 2, 1))
    energy = np.abs(spectrum).astype(np.float32) ** 2
    if squeezed:
        return spectrum[0], energy[0]
    return spectrum, energy


def compute_max_tf(tf_energy: RealArray) -> RealArray:
    """Eq. (18): maximum TF energy over frequency for each time frame."""

    p = np.asarray(tf_energy)
    if p.ndim not in (2, 3):
        raise ValueError("TF energy must be [frequency,time] or batched")
    return np.max(p, axis=-2)


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _valley(values: RealArray, index: int) -> bool:
    return 0 < index < values.size - 1 and values[index] <= values[index - 1] and values[index] <= values[index + 1]


def _extract_one(values: RealArray, min_clean_frames: int) -> NDArray[np.bool_]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("max-TF must be finite and non-empty")
    peak = float(np.max(values))
    if peak <= np.finfo(np.float64).tiny:
        return np.zeros(values.size, dtype=bool)
    coarse = values >= peak / 4.0  # Eq. (22)--(23)
    extended = np.zeros_like(coarse)
    for start, stop in _runs(coarse):
        local_threshold = float(np.max(values[start:stop])) / 4.0  # Eq. (24)
        left, right = start, stop - 1
        while left > 0 and values[left] > local_threshold and not _valley(values, left):
            left -= 1
        while right < values.size - 1 and values[right] > local_threshold and not _valley(values, right):
            right += 1
        extended[left : right + 1] = True
    clean = ~extended  # Eq. (26)
    for start, stop in _runs(clean):
        if stop - start < min_clean_frames:
            clean[start:stop] = False
    return ~clean


def extract_jamming_free_segments(
    max_tf: RealArray, *, num_samples: int, config: MaxTFConfig
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Estimate frame jammer masks and sample-domain jamming-free masks."""

    p, squeezed = _as_batch(np.asarray(max_tf, dtype=np.float32))
    jammer_frames = np.stack(
        [_extract_one(row, config.min_clean_frames) for row in p], axis=0
    )
    centers = np.minimum(np.arange(p.shape[1]) * config.hop_length, num_samples - 1)
    sample_idx = np.arange(num_samples)
    nearest = np.clip(
        np.rint(sample_idx / config.hop_length).astype(np.int64), 0, p.shape[1] - 1
    )
    clean_samples = ~jammer_frames[:, nearest]
    if squeezed:
        return jammer_frames[0], clean_samples[0]
    return jammer_frames, clean_samples


def _smooth_one(response: RealArray, config: MaxTFConfig) -> RealArray:
    """High-PFA peak preservation plus interval averaging (paper Step 6)."""

    h = np.asarray(response, dtype=np.float64)
    power = h * h
    threshold = float(np.quantile(power, config.peak_quantile))
    local = np.zeros(h.size, dtype=bool)
    if h.size >= 3:
        local[1:-1] = (power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:])
    peaks = np.flatnonzero(local & (power >= threshold))
    if peaks.size < 2:
        kernel = np.ones(config.smoothing_bins, dtype=np.float64) / config.smoothing_bins
        out = np.convolve(h, kernel, mode="same")
    else:
        out = h.copy()
        for left, right in zip(peaks[:-1], peaks[1:]):
            if right - left > 1:
                out[left + 1 : right] = np.mean(h[left + 1 : right])
        kernel = np.ones(config.smoothing_bins, dtype=np.float64) / config.smoothing_bins
        out = np.convolve(out, kernel, mode="same")
        out[peaks] = np.maximum(out[peaks], h[peaks])
    maximum = float(np.max(out))
    return (out / maximum if maximum > 0 else np.ones_like(out)).astype(np.float32)


def construct_bandpass_filter(
    dechirped_signal: ComplexArray,
    clean_sample_mask: NDArray[np.bool_],
    config: MaxTFConfig,
) -> tuple[RealArray, RealArray]:
    """Eq. (19)--(20) and Step 6 smoothing."""

    x, squeezed = _as_batch(np.asarray(dechirped_signal, dtype=np.complex64))
    mask, _ = _as_batch(np.asarray(clean_sample_mask, dtype=bool))
    if x.shape != mask.shape:
        raise ValueError("signal/mask shapes differ")
    raw, smooth = [], []
    for row, keep in zip(x, mask):
        if not np.any(keep):
            h = np.ones(row.size, dtype=np.float32)
        else:
            magnitude = np.abs(np.fft.fft(row * keep))
            maximum = float(np.max(magnitude))
            h = (magnitude / maximum if maximum > 0 else np.ones_like(magnitude)).astype(np.float32)
        raw.append(h)
        smooth.append(_smooth_one(h, config))
    raw_a, smooth_a = np.stack(raw), np.stack(smooth)
    if squeezed:
        return raw_a[0], smooth_a[0]
    return raw_a, smooth_a


def apply_filter(dechirped_signal: ComplexArray, response: RealArray) -> ComplexArray:
    """Eq. (21): pulse-compression spectrum times smoothed response."""

    x = np.asarray(dechirped_signal)
    h = np.asarray(response)
    if x.shape != h.shape:
        raise ValueError("dechirped signal/filter shapes differ")
    return np.fft.fft(x, axis=-1) * h


def inverse_dechirp(pc_result: ComplexArray, reference: ComplexArray) -> ComplexArray:
    """Return a pulse-compression-compatible waveform-domain result."""

    x_de = np.fft.ifft(pc_result, axis=-1)
    return x_de * np.asarray(reference)


def max_tf_suppress(
    received: ComplexArray, reference: ComplexArray, config: MaxTFConfig
) -> MaxTFResult:
    """Run the complete source-ordered Max-TF suppression chain."""

    x_de = dechirp(received, reference)
    stft, energy = compute_stft(x_de, config)
    pmax = compute_max_tf(energy)
    jam_frames, clean_samples = extract_jamming_free_segments(
        pmax, num_samples=x_de.shape[-1], config=config
    )
    raw_h, smooth_h = construct_bandpass_filter(x_de, clean_samples, config)
    before = np.fft.fft(x_de, axis=-1)
    after = apply_filter(x_de, smooth_h)
    recovered = inverse_dechirp(after, reference)
    return MaxTFResult(
        dechirped=x_de,
        stft=stft,
        tf_energy=energy,
        max_tf=pmax,
        jammer_frame_mask=jam_frames,
        clean_sample_mask=clean_samples,
        raw_filter=raw_h,
        smoothed_filter=smooth_h,
        pc_before=before,
        pc_after=after,
        recovered_waveform=recovered,
    )
