"""Group-energy metrics and periodic Doppler scoring for 1103 experiments.

All arrays below use [parent, state, pulse, fast_time]; states are E, S+E,
J+E, S+J+E. No training, legacy Gate, or TEST data is read here.
"""
from __future__ import annotations

import numpy as np
import torch

from .radar_eval import (PRF_HZ, ca_cfar, doppler_process, mainlobe_guard,
                         pulse_compress, sjrif, target_peak_range)


def energy(x):
    return np.sum(np.abs(x) ** 2, axis=(-2, -1))


def nearest_doppler_bin(frequency_hz, bins):
    """Nearest slow-time bin on the PRF circle, including Nyquist wrap."""
    distance = (np.asarray(bins) - frequency_hz + PRF_HZ / 2) % PRF_HZ - PRF_HZ / 2
    return int(np.argmin(np.abs(distance)))


def states_from_components(s, j, e):
    return np.stack((e, s + e, j + e, s + j + e), axis=1)


def group_metrics(s, j, e, restored):
    s, j, e, u = (np.asarray(a, np.complex128) for a in (s, j, e, restored))
    if u.shape != (len(s), 4, *s.shape[1:]):
        raise ValueError("expected [parent,4,pulse,time] restored waveforms")
    floor = 1e-24
    response = u[:, 3] - u[:, 2]
    es, eu, ee = [energy(x).clip(floor) for x in (s, s + e, e)]
    gain = (s.conj() * response).sum(axis=(-2, -1)) / es
    return dict(
        u_nmse=energy(u[:, 3] - (s + e)) / eu,
        damage=energy(response - s) / es,
        background_nmse=energy(u[:, 2] - e) / ee,
        h0_nmse=energy(u[:, 1] - (s + e)) / eu,
        noise_nmse=energy(u[:, 0] - e) / ee,
        no_jammer_pooled_nmse=(energy(u[:, 0] - e) + energy(u[:, 1] - (s + e))) / (ee + eu),
        amplitude=np.abs(gain), phase_deg=np.angle(gain, deg=True))


def restore_model(model, states, device, batch_pulses=192):
    """Model predicts correction C; restored waveform is observation minus C."""
    x = np.asarray(states)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty_like(flat, dtype=np.complex64)
    with torch.inference_mode():
        for lo in range(0, len(flat), batch_pulses):
            hi = min(lo + batch_pulses, len(flat))
            iq = np.stack((flat[lo:hi].real, flat[lo:hi].imag), axis=1).astype(np.float32)
            c = model(torch.from_numpy(iq).to(device)).cpu().numpy()
            out[lo:hi] = flat[lo:hi] - (c[:, 0] + 1j * c[:, 1])
    return out.reshape(x.shape)


def radar_metrics(s, observed, restored, doppler):
    """Per-parent radar values, including BOTH target-absent states' Pfa."""
    before, _ = pulse_compress(observed[:, 3])
    compressed, _ = pulse_compress(restored)
    peaks = target_peak_range(s)
    guard = mainlobe_guard()
    values = {k: [] for k in (
        "sjrif_db", "false_peak_suppression_db", "pd", "pd_no_jammer",
        "cell_pfa_noise", "frame_pfa_noise", "cell_pfa_jammer", "frame_pfa_jammer")}
    for i, tr in enumerate(peaks):
        measure = sjrif(before[i], compressed[i, 3], int(tr), mainlobe_guard=guard)
        values["sjrif_db"].append(measure["sjrif_db"])
        values["false_peak_suppression_db"].append(measure["false_peak_suppression_db"])
        for state in range(4):
            rd, bins = doppler_process(compressed[i, state])
            detections, _, valid, nr, _ = ca_cfar(np.abs(rd) ** 2)
            assert nr == 436 and valid.sum() == 9032
            if state in (0, 2):
                label = "noise" if state == 0 else "jammer"
                false = detections & valid
                values[f"cell_pfa_{label}"].append(float(false.sum() / valid.sum()))
                values[f"frame_pfa_{label}"].append(float(false.any()))
            else:
                db = nearest_doppler_bin(float(doppler[i]), bins)
                if not valid[db, tr]:
                    raise ValueError(f"target outside valid CFAR region: {tr}")
                hit = detections[db, max(0, tr-1):tr+2].any()
                values["pd" if state == 3 else "pd_no_jammer"].append(float(hit))
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}


def coherent_metrics(s, output, fft_points=4096):
    """1103 eq. (4-3), relative to target-only pulse compression.

    Peak error uses the SAME zero-padded slow-time FFT for target and output;
    padding interpolates the spectrum and does not increase physical resolution.
    """
    zs, _ = pulse_compress(s)
    zy, _ = pulse_compress(output)
    peaks = target_peak_range(s)
    phase, power, frequency = [], [], []
    bins = np.fft.fftshift(np.fft.fftfreq(fft_points, d=1/PRF_HZ))
    for i, tr in enumerate(peaks):
        target, y = zs[i, :, tr], zy[i, :, tr]
        chi = y * target.conj() / np.maximum(np.abs(target)**2, 1e-24)
        phi0 = np.angle(chi.sum())
        phase.append(float(np.rad2deg(np.sqrt(np.mean(np.angle(chi*np.exp(-1j*phi0))**2)))))
        ratio = np.abs(np.sum(y*target.conj()/np.maximum(np.abs(target), 1e-24)))**2 / np.sum(np.abs(target))**2
        power.append(float(10*np.log10(max(ratio, 1e-300))))
        ft = bins[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(target, n=fft_points))))]
        fy = bins[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(y, n=fft_points))))]
        frequency.append(float(abs((fy-ft+PRF_HZ/2) % PRF_HZ-PRF_HZ/2)))
    return dict(phase_rmse_deg=np.asarray(phase), coherent_power_db=np.asarray(power),
                peak_error_hz=np.asarray(frequency))


def summarize(values, seed=20261114, reps=4000):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite one-dimensional parent values required")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(x), size=(reps, len(x)))
    ci = np.quantile(x[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.quantile(x, .95)),
                parent_bootstrap_ci95=ci.tolist(), parents=len(x))
