import unittest

import numpy as np

from tools.audit_idea_structure_controls_20261118 import local_lag_values, normalize_rows


class StructureControlsTests(unittest.TestCase):
    def test_local_lag_matches_direct_formula(self):
        rng = np.random.default_rng(17)
        signal = rng.normal(size=100) + 1j * rng.normal(size=100)
        actual = local_lag_values(signal, 12, lags=(7,))
        expected = []
        for start in range(100 - 7 - 12 + 1):
            left, right = signal[start:start + 12], signal[start + 7:start + 7 + 12]
            expected.append(abs(np.vdot(right, left)) / (np.linalg.norm(left) * np.linalg.norm(right)))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_tone_local_similarity_and_scale_invariance(self):
        signal = np.exp(2j * np.pi * .13 * np.arange(400))
        np.testing.assert_allclose(local_lag_values(signal, 32), 1., atol=1e-12)
        np.testing.assert_allclose(local_lag_values(signal * (3 - 4j), 32), 1., atol=1e-12)

    def test_empty_signal_does_not_generate_lag_evidence(self):
        self.assertEqual(len(local_lag_values(np.zeros(400), 32)), 0)

    def test_row_normalization_preserves_occupancy(self):
        stack = np.zeros((8, 40), complex)
        stack[2] = 3.
        stack[7] = 8j
        normalized, active = normalize_rows(stack)
        self.assertEqual(active, 2)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.)
        empty, active = normalize_rows(np.zeros_like(stack))
        self.assertEqual(active, 0)
        self.assertEqual(empty.shape, (0, 40))


if __name__ == '__main__':
    unittest.main()
