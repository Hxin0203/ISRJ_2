"""1103 comparison adapters returning waveforms in the common receive grid.

Max-TF and GMM-OMP use only the observation and public LFM parameters.
BCS receives the exact non-jammer support and is therefore an oracle reference.
"""
from __future__ import annotations

import numpy as np
import torch

from .vendor.baselines_1103.bcs import BCSConfig, bcs_reconstruct
from .vendor.baselines_1103.max_tf import MaxTFConfig, max_tf_suppress
from .vendor.baselines_1103.paper_gmm_omp import PaperGMMOMPBaseline, PaperGMMOMPConfig

FS = 60e6
BANDWIDTH = 30e6
DURATION = 10e-6
CHIRP_RATE = BANDWIDTH / DURATION
N = 1800


def tx_reference(length: int = N) -> np.ndarray:
    t = (np.arange(length, dtype=np.float64) - (round(DURATION * FS) - 1) / 2) / FS
    return np.exp(1j * np.pi * CHIRP_RATE * t**2)


def run_max_tf(observed: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=np.complex128)
    if y.ndim != 2 or y.shape[-1] != N:
        raise ValueError("expected [pulse, 1800] waveform")
    ref = tx_reference(y.shape[-1])
    cfg = MaxTFConfig(sample_rate_hz=FS, window_length=51, hop_length=12,
                      n_fft=128, min_clean_frames=2, smoothing_bins=3,
                      peak_quantile=.7)
    result = max_tf_suppress(y, ref, cfg)
    # Dechirp convention is x_de=conj(y)*ref, so invert with conjugation.
    out = np.conj(np.fft.ifft(result.pc_after, axis=-1)) * ref
    return np.asarray(out, dtype=np.complex64)


def run_gmm_omp(observed: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=np.complex64)
    if y.ndim != 2 or y.shape[-1] != N:
        raise ValueError("expected [pulse, 1800] waveform")
    tensor = torch.from_numpy(np.stack((y.real, y.imag), axis=1))
    cfg = PaperGMMOMPConfig(sample_rate_hz=FS, chirp_rate_hz_s=CHIRP_RATE,
                            gmm_max_iterations=100, omp_sparsity=8)
    model = PaperGMMOMPBaseline(cfg).eval()
    with torch.inference_mode():
        result = model(tensor)
    iq = result["u_hat"].cpu().numpy()
    return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)


def run_bcs_oracle(observed: np.ndarray, jammer: np.ndarray) -> tuple[np.ndarray, dict]:
    y = np.asarray(observed, dtype=np.complex128)
    j = np.asarray(jammer)
    if y.ndim != 2 or y.shape[-1] != N or j.shape != y.shape:
        raise ValueError("expected matching [pulse, 1800] observed and jammer arrays")
    ref = tx_reference(y.shape[-1])
    mask = np.abs(j) == 0
    cfg = BCSConfig(lambda_value=.01, tau=1e-8, outer_tolerance=.01,
                    inner_tolerance=.001, outer_max_iterations=40,
                    inner_max_iterations=200)
    output, outer, converged = [], [], []
    for pulse in range(len(y)):
        if not mask[pulse].any():
            output.append(np.zeros(N, dtype=np.complex64))
            outer.append(0)
            converged.append(False)
            continue
        dechirped = np.conj(y[pulse]) * ref
        result = bcs_reconstruct(dechirped, mask[pulse], cfg)
        output.append(np.conj(result.reconstructed_time) * ref)
        outer.append(result.outer_iterations)
        converged.append(result.converged)
    return np.asarray(output, dtype=np.complex64), {
        "oracle_support": True,
        "converged_pulses": int(sum(converged)),
        "pulse_count": int(len(y)),
        "outer_iterations": outer,
    }
