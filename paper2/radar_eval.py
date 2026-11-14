"""Physical-coordinate pulse compression, Doppler processing, and CA-CFAR."""
from __future__ import annotations

import numpy as np


FS_HZ = 60e6
BANDWIDTH_HZ = 30e6
PULSE_WIDTH_S = 10e-6
PULSE_SAMPLES = 600
PRI_S = 1800 / FS_HZ
PRF_HZ = 1.0 / PRI_S


def lfm_reference() -> np.ndarray:
    """Return the 600-sample complex baseband template used by the simulator."""
    t = np.arange(PULSE_SAMPLES, dtype=np.float64) / FS_HZ - PULSE_WIDTH_S / 2
    return np.exp(1j * np.pi * (BANDWIDTH_HZ / PULSE_WIDTH_S) * t**2)


def linear_matched_filter(x: np.ndarray, reference: np.ndarray | None = None):
    """Full linear correlation with explicit lag coordinates.

    The returned lag is the input pulse's start index when ``x`` contains a
    delayed copy of ``reference``. This avoids confusing the circular FFT
    indexing used in the training loss with physical range coordinates.
    """
    x = np.asarray(x, dtype=np.complex128)
    ref = lfm_reference() if reference is None else np.asarray(reference, dtype=np.complex128)
    if x.ndim < 1 or ref.ndim != 1 or ref.size == 0:
        raise ValueError("x must have a sample axis and reference must be a nonempty vector")
    n = x.shape[-1] + ref.size - 1
    nfft = 1 << (n - 1).bit_length()
    kernel = np.conj(ref[::-1])
    out = np.fft.ifft(np.fft.fft(x, nfft, axis=-1) * np.fft.fft(kernel, nfft), axis=-1)[..., :n]
    lags = np.arange(-(ref.size - 1), x.shape[-1], dtype=np.int64)
    return out, lags


def pulse_compress(x: np.ndarray, reference: np.ndarray | None = None):
    """Return complete-waveform nonnegative delays, matching the 1103 grid.

    With 1800 receive samples and a 600-sample template this returns lags
    0..1200 (1201 range cells), excluding negative delays and partial-template
    correlations at the end of the receive window.
    """
    x = np.asarray(x, dtype=np.complex128)
    ref = lfm_reference() if reference is None else np.asarray(reference, dtype=np.complex128)
    if x.shape[-1] < ref.size:
        raise ValueError("receive window must contain at least one full template")
    full, all_lags = linear_matched_filter(x, ref)
    selection = (all_lags >= 0) & (all_lags <= x.shape[-1] - ref.size)
    return full[..., selection], all_lags[selection]


def target_peak_range(target_echoes: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Return each pulse group's no-jammer range peak in physical lag coordinates.

    The simulator's target start delay and matched-filter peak differ when
    fast-time Doppler couples into LFM range. Scoring must use the target-only
    compressed peak rather than treating the waveform start as the peak.
    """
    target = np.asarray(target_echoes)
    if target.ndim < 2:
        raise ValueError("target_echoes must have [pulse, fast-time] axes")
    compressed, lags = pulse_compress(target, reference)
    power = np.mean(np.abs(compressed) ** 2, axis=-2)
    return lags[np.argmax(power, axis=-1)]


def doppler_process(pulse_compressed: np.ndarray, pri_s: float = PRI_S):
    """Return slow-time FFT (zero Doppler centered) and Doppler frequencies."""
    z = np.asarray(pulse_compressed)
    if z.ndim < 2 or pri_s <= 0:
        raise ValueError("input must end in [pulse, range] dimensions and PRI must be positive")
    n_pulses = z.shape[-2]
    rd = np.fft.fftshift(np.fft.fft(z, axis=-2), axes=-2)
    fd = np.fft.fftshift(np.fft.fftfreq(n_pulses, d=pri_s))
    return rd, fd


def alias_doppler_frequency(frequency_hz: float, pri_s: float = PRI_S) -> float:
    """Fold a physical Doppler frequency into the sampled slow-time Nyquist interval."""
    if not np.isfinite(frequency_hz) or not np.isfinite(pri_s) or pri_s <= 0:
        raise ValueError("frequency must be finite and PRI must be finite and positive")
    prf = 1.0 / pri_s
    return float((frequency_hz + prf / 2) % prf - prf / 2)


def ca_cfar(power: np.ndarray, *, p_fa: float = 1e-6,
            range_guard: int = 12, range_reference: int = 24,
            doppler_guard: int = 1, doppler_reference: int = 2):
    """2-D CA-CFAR threshold with periodic Doppler and clipped range edges.

    ``power`` has [doppler, range] axes. Range cells without the complete
    reference window are marked invalid. The returned reference count follows
    the stated rectangular-window subtraction and is 436 at 12/24 and 1/2.
    """
    p = np.asarray(power, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("power must be a finite, nonnegative [doppler, range] array")
    if not 0 < p_fa < 1:
        raise ValueError("p_fa must lie strictly between zero and one")
    if min(range_guard, range_reference, doppler_guard, doppler_reference) < 0:
        raise ValueError("guard and reference widths must be nonnegative")
    nr = ((2 * (range_guard + range_reference) + 1)
          * (2 * (doppler_guard + doppler_reference) + 1)
          - (2 * range_guard + 1) * (2 * doppler_guard + 1))
    if nr <= 0:
        raise ValueError("CFAR reference window is empty")
    alpha = nr * (p_fa ** (-1.0 / nr) - 1.0)
    nd, nrange = p.shape
    r_outer, d_outer = range_guard + range_reference, doppler_guard + doppler_reference

    # Range cumulative sums make each rectangular Doppler-window sum O(1).
    padded = np.pad(p, ((0, 0), (r_outer, r_outer)))
    cs = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    outer_range = cs[:, 2*r_outer+1:2*r_outer+1+nrange] - cs[:, :nrange]
    inner_left = r_outer - range_guard
    inner_right = r_outer + range_guard + 1
    inner_range = cs[:, inner_right:inner_right+nrange] - cs[:, inner_left:inner_left+nrange]

    # Sum the outer and inner range strips over periodic Doppler neighborhoods.
    d_outer_sum = np.zeros_like(p)
    d_inner_sum = np.zeros_like(p)
    for shift in range(-d_outer, d_outer + 1):
        d_outer_sum += np.roll(outer_range, shift, axis=0)
    for shift in range(-doppler_guard, doppler_guard + 1):
        d_inner_sum += np.roll(inner_range, shift, axis=0)
    ref_sum = d_outer_sum - d_inner_sum
    valid = np.zeros_like(p, dtype=bool)
    if nrange > 2 * r_outer:
        valid[:, r_outer:nrange-r_outer] = True
    threshold = np.full_like(p, np.inf)
    threshold[valid] = alpha * ref_sum[valid] / nr
    return p > threshold, threshold, valid, nr, alpha


def sjrif(before: np.ndarray, after: np.ndarray, target_range_index: int,
          *, mainlobe_guard: int) -> dict[str, float]:
    """Compute target-peak to strongest non-target-range power improvement."""
    b = np.asarray(before, dtype=np.complex128)
    a = np.asarray(after, dtype=np.complex128)
    if b.shape != a.shape or b.ndim != 2:
        raise ValueError("before and after must share [pulse, range] shape")
    if not 0 <= target_range_index < b.shape[-1] or mainlobe_guard < 0:
        raise ValueError("invalid target range index or mainlobe guard")
    pb = np.mean(np.abs(b)**2, axis=0)
    pa = np.mean(np.abs(a)**2, axis=0)
    non_target = np.ones(pb.size, dtype=bool)
    non_target[target_range_index] = False
    outside_mainlobe = np.ones(pb.size, dtype=bool)
    outside_mainlobe[max(0, target_range_index-mainlobe_guard):
                     min(pb.size, target_range_index+mainlobe_guard+1)] = False
    if not np.any(non_target) or not np.any(outside_mainlobe):
        raise ValueError("no non-target range bins remain")
    floor = np.finfo(np.float64).tiny
    def sjr(power):
        return 10*np.log10(max(power[target_range_index], floor)/max(power[non_target].max(), floor))
    false_peak = 10*np.log10(max(pb[outside_mainlobe].max(), floor)/max(pa[outside_mainlobe].max(), floor))
    return {"before_db": float(sjr(pb)), "after_db": float(sjr(pa)),
            "sjrif_db": float(sjr(pa)-sjr(pb)),
            "false_peak_suppression_db": float(false_peak)}


def mainlobe_guard(reference: np.ndarray | None = None) -> int:
    """First right-side null of the template autocorrelation, in range bins."""
    ref = lfm_reference() if reference is None else np.asarray(reference, dtype=np.complex128)
    ac, lags = linear_matched_filter(ref, ref)
    center = int(np.flatnonzero(lags == 0)[0])
    mag = np.abs(ac[center:])
    peaks = np.flatnonzero((mag[1:-1] <= mag[:-2]) & (mag[1:-1] < mag[2:])) + 1
    peaks = peaks[peaks > 0]
    if not len(peaks):
        raise RuntimeError("could not locate the template mainlobe null")
    return int(peaks[0])
