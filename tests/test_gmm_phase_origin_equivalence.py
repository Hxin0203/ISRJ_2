"""Regression test for the GMM-OMP dechirp phase-origin hypothesis.

For the project waveform, centered and zero-origin dechirp references differ
by an exact integer-bin complex modulation. Fourier OMP should therefore be
equivariant to the phase-origin choice; changing the reference must not be
presented as a sparsity fix.
"""

import torch

from paper2.vendor.baselines_1103.paper_gmm_omp import fourier_omp_reconstruct


def test_project_phase_origin_is_integer_bin_shift_and_omp_equivariant() -> None:
    length = 1800
    sample_rate_hz = 60.0e6
    pulse_width_s = 10.0e-6
    chirp_rate_hz_s = 3.0e12
    shift_bins = chirp_rate_hz_s * pulse_width_s / 2.0 / (sample_rate_hz / length)
    assert shift_bins == 450

    # Several contiguous clean intervals mimic the observation pattern passed
    # to OMP, without loading or exposing any project dataset.
    index = torch.arange(length, dtype=torch.float64)
    time = index / sample_rate_hz
    centered_time = time - pulse_width_s / 2.0
    tx = torch.exp(1j * torch.pi * chirp_rate_hz_s * centered_time.square())
    mask = torch.zeros(length, dtype=torch.bool)
    mask[:180] = True
    mask[520:730] = True
    mask[1190:1510] = True

    lo_zero = torch.exp(1j * torch.pi * chirp_rate_hz_s * time.square())
    lo_centered = torch.exp(
        1j * torch.pi * chirp_rate_hz_s * centered_time.square()
    )
    dechirped_zero = tx * lo_zero.conj()
    dechirped_centered = tx * lo_centered.conj()
    expected_modulation = torch.exp(
        1j * 2.0 * torch.pi * shift_bins * index / length
        - 1j * torch.pi * chirp_rate_hz_s * pulse_width_s**2 / 4.0
    )

    torch.testing.assert_close(
        dechirped_centered,
        dechirped_zero * expected_modulation,
        rtol=1e-10,
        atol=1e-10,
    )
    zero_rec, zero_count = fourier_omp_reconstruct(
        dechirped_zero[None, :],
        mask[None, :],
        sparsity=8,
        relative_tolerance=1e-3,
        ridge=1e-6,
        minimum_samples=8,
    )
    centered_rec, centered_count = fourier_omp_reconstruct(
        dechirped_centered[None, :],
        mask[None, :],
        sparsity=8,
        relative_tolerance=1e-3,
        ridge=1e-6,
        minimum_samples=8,
    )

    torch.testing.assert_close(zero_count, centered_count)
    torch.testing.assert_close(
        centered_rec,
        zero_rec * expected_modulation[None, :],
        rtol=2e-5,
        atol=2e-5,
    )
    zero_energy = torch.fft.fft(dechirped_zero).abs().square()
    centered_energy = torch.fft.fft(dechirped_centered).abs().square()
    torch.testing.assert_close(
        torch.sort(zero_energy).values,
        torch.sort(centered_energy).values,
        rtol=1e-10,
        atol=1e-10,
    )
