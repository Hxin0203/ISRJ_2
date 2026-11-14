import numpy as np

from paper2.evaluation_1103_v4 import (coherent_metrics, group_metrics,
    nearest_doppler_bin, states_from_components)
from paper2.radar_eval import PRF_HZ, doppler_process, lfm_reference


def test_nyquist_bin_scoring_wraps_to_first_bin():
    _, bins = doppler_process(np.zeros((8, 1)))
    assert nearest_doppler_bin(PRF_HZ/2-100, bins) == 0
    assert nearest_doppler_bin(3*PRF_HZ+PRF_HZ/2-100, bins) == 0


def test_group_metric_is_ratio_of_group_energies_not_mean_of_ratios():
    s = np.ones((1, 8, 4), complex)
    e, j = np.zeros_like(s), np.zeros_like(s)
    s[:, 0] *= 10
    u = states_from_components(s, j, e)
    u[:, 3, 0] += 1
    m = group_metrics(s, j, e, u)
    assert np.isclose(m['u_nmse'][0], 4/(400+7*4))
    assert np.isclose(m['damage'][0], 4/(400+7*4))
    assert m['no_jammer_pooled_nmse'][0] == 0


def test_coherent_gain_preserves_phase_and_peak_with_known_power_loss():
    s = np.zeros((1, 8, 1800), complex)
    s[:, :, 200:800] = lfm_reference()[None, None] * np.exp(2j*np.pi*2*np.arange(8)/8)[None, :, None]
    m = coherent_metrics(s, .5*np.exp(.7j)*s)
    assert m['phase_rmse_deg'][0] < 1e-10
    assert np.isclose(m['coherent_power_db'][0], 20*np.log10(.5))
    assert m['peak_error_hz'][0] == 0
