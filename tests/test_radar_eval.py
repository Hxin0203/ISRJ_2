import numpy as np

from paper2.radar_eval import (
    alias_doppler_frequency, ca_cfar, doppler_process, linear_matched_filter, lfm_reference,
    mainlobe_guard, pulse_compress, sjrif,
    target_peak_range,
)


def test_linear_matched_filter_reports_physical_delay_lag():
    ref = lfm_reference()
    x = np.zeros(1800, dtype=np.complex128)
    x[137:137+len(ref)] = ref
    corr, lags = linear_matched_filter(x, ref)
    assert len(corr) == len(x) + len(ref) - 1
    assert lags[np.argmax(np.abs(corr))] == 137


def test_slow_time_fft_places_coherent_doppler_at_expected_bin():
    n, pri, bin_index = 8, 30e-6, 2
    signal = np.exp(2j*np.pi*(bin_index/(n*pri))*np.arange(n)*pri)[:, None]
    rd, fd = doppler_process(signal, pri)
    assert rd.shape == signal.shape
    assert fd[np.argmax(np.abs(rd[:, 0]))] == bin_index/(n*pri)


def test_physical_doppler_is_folded_before_slow_time_detection():
    pri = 30e-6
    prf = 1 / pri
    assert np.isclose(alias_doppler_frequency(200e3, pri), 200e3 - 6 * prf)
    assert np.isclose(alias_doppler_frequency(-200e3, pri), -200e3 + 6 * prf)
    assert np.isclose(alias_doppler_frequency(prf / 2, pri), -prf / 2)


def test_cfar_reference_count_threshold_and_edge_mask():
    power = np.arange(8*100, dtype=np.float64).reshape(8, 100) + 1
    detections, threshold, valid, nr, alpha = ca_cfar(power)
    assert nr == 436
    assert alpha > 0
    assert valid[:, 36:-36].all() and not valid[:, :36].any()
    assert np.isfinite(threshold[valid]).all()
    assert not detections.any()
    d, r = 4, 50
    outer = sum(power[(d+dd) % 8, r+dr]
                for dd in range(-3, 4) for dr in range(-36, 37))
    inner = sum(power[(d+dd) % 8, r+dr]
                for dd in range(-1, 2) for dr in range(-12, 13))
    assert threshold[d, r] * nr / alpha == outer-inner


def test_sjrif_and_false_peak_suppression_have_power_db_meaning():
    before = np.zeros((8, 30), dtype=np.complex128)
    after = np.zeros_like(before)
    before[:, 10] = 1
    before[:, 20] = 2
    after[:, 10] = 1
    after[:, 20] = 1
    # SJRIF excludes only the target bin; the companion false-peak measure
    # excludes the configured mainlobe neighborhood.
    m = sjrif(before, after, 10, mainlobe_guard=1)
    assert m["sjrif_db"] == 6.020599913279624
    assert m["false_peak_suppression_db"] == 6.020599913279624
    assert mainlobe_guard() > 0


def test_pulse_compression_accepts_pulse_groups():
    group = np.zeros((8, 1800), dtype=np.complex128)
    group[:, 200:800] = lfm_reference()
    compressed, lags = pulse_compress(group)
    assert compressed.shape == (8, 1201)
    np.testing.assert_array_equal(lags, np.arange(1201))
    assert np.all(lags[np.argmax(np.abs(compressed), axis=-1)] == 200)


def test_target_peak_range_uses_doppler_coupled_compressed_peak():
    fs, duration, slope = 60e6, 10e-6, 30e6 / 10e-6
    delay, doppler = 820.25, 120e3
    sample = np.arange(1800)
    local_t = (sample - delay) / fs
    active = (local_t >= 0) & (local_t < duration)
    centered = local_t - duration / 2
    target = np.zeros((8, 1800), dtype=np.complex128)
    pulse = active * np.exp(1j * (np.pi * slope * centered**2 + 2*np.pi*doppler*centered))
    target[:] = pulse
    peak = target_peak_range(target)
    compressed, lags = pulse_compress(target)
    expected = lags[np.argmax(np.mean(np.abs(compressed)**2, axis=0))]
    assert peak == expected
    assert peak != round(delay)
