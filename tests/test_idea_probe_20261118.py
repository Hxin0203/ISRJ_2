import inspect
import unittest

import numpy as np

from tools.audit_idea_probe_20261118 import check_rows, finite_tree
from tools.idea_probe_20261118.run import parent_mean
from tools.idea_probe_20261118.mechanics import (
    DECHIRP, FS, LENGTH, RATE, complement, energy, fit_operator, hankel_metrics, infer,
    operation_for, orth, spectrum_metrics, target_atom,
)


class IdeaProbeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(71)

    def complex_array(self, shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def test_fixed_target_preservation(self):
        targets = self.complex_array((40, 2))
        dictionary = self.complex_array((40, 5))
        operator = fit_operator(targets, dictionary)
        self.assertLess(np.linalg.norm(operator.apply(targets)), 1e-12)

    def test_exact_subspace_intersection_is_not_inverted(self):
        targets = self.complex_array((40, 2))
        for ridge in (0., .01):
            operator = fit_operator(targets, targets, ridge)
            self.assertLess(np.linalg.norm(operator.factor), 1e-12)

    def test_no_protection_matches_ridge(self):
        dictionary = self.complex_array((40, 5))
        signal = self.complex_array(40)
        operator = fit_operator(np.empty((40, 0)), dictionary)
        normalized = operator.dictionary
        direct = normalized @ np.linalg.solve(normalized.conj().T @ normalized + .01 * np.eye(5),
                                              normalized.conj().T @ signal)
        np.testing.assert_allclose(operator.apply(signal), direct, atol=1e-12)

    def test_adaptive_damage_decomposition(self):
        background, target = self.complex_array((2, 40))
        operators = [fit_operator(self.complex_array((40, 2)), self.complex_array((40, 5)))
                     for _ in range(2)]
        before, after = operators
        damage = before.apply(background) - after.apply(background + target)
        leakage = -after.apply(target)
        drift = before.apply(background) - after.apply(background)
        np.testing.assert_allclose(damage, leakage + drift, atol=1e-12)

    def test_projected_only_estimation_invariant_for_fixed_targets(self):
        targets = target_atom(323., 100e3)[:, None]
        observation = self.complex_array(LENGTH)
        before = operation_for(observation, targets)
        after = operation_for(observation + targets[:, 0] * (2 + 3j), targets)
        np.testing.assert_allclose(before.apply(observation), after.apply(observation), atol=1e-10)

    def test_observation_only_signature(self):
        self.assertEqual(list(inspect.signature(infer).parameters), ['observation'])

    def test_noiseless_interior_dechirped_chirp_rank_one(self):
        indices = np.arange(300)
        tone = np.exp(2j * np.pi * .13 * indices)
        result = hankel_metrics(tone)
        self.assertEqual(result['rank90'], 1)
        self.assertAlmostEqual(result['top1'], 1., places=12)

    def test_full_scene_boundary_not_assumed_rank_one(self):
        tone = np.exp(2j * np.pi * .13 * np.arange(300))
        signal = np.pad(tone, (300, 300))
        self.assertLess(hankel_metrics(signal)['top1'], .99)

    def test_slow_time_coherence_and_empty_signal(self):
        pulse = self.complex_array(100)
        stack = np.exp(1j * np.arange(8))[:, None] * pulse
        self.assertAlmostEqual(spectrum_metrics(stack)['top1'], 1.)
        self.assertEqual(spectrum_metrics(np.zeros((8, 100)))['rank90'], 0)

    def test_projection_residual_and_atom_support(self):
        atom = target_atom(323.4, 190e3)
        self.assertEqual(np.count_nonzero(atom), 600)
        self.assertLess(energy(complement(orth(atom[:, None]), atom)), 1e-20)

    def test_dechirp_delay_requires_frequency_coupling_without_wrap(self):
        signal = self.complex_array(LENGTH)
        delay = 39
        shifted = np.pad(signal[:-delay], (delay, 0))
        dechirped_then_shifted = np.pad((signal * DECHIRP)[:-delay], (delay, 0))
        times = np.arange(LENGTH) / FS
        seconds = delay / FS
        coupling = np.exp(-1j * np.pi * RATE * (2 * times * seconds - seconds**2))
        np.testing.assert_allclose(DECHIRP * shifted, coupling * dechirped_then_shifted, atol=1e-10)

    def test_parent_mean_does_not_expand_correlated_rows(self):
        rows = [dict(seed=1, parent=1000, value=value) for value in (1., 3., 5.)]
        rows.append(dict(seed=1, parent=1001, value=8.))
        self.assertEqual(parent_mean(rows, 'value'), {(1, 1000): 3., (1, 1001): 8.})

    def test_audit_rejects_nonfinite_values(self):
        self.assertTrue(finite_tree({'rows': [1., 0., 'label']}))
        self.assertFalse(finite_tree({'rows': [float('nan')]}))
        self.assertFalse(finite_tree({'value': float('inf')}))

    def test_audit_requires_complete_unique_cells(self):
        manifest = dict(seeds=[1], parents_per_seed=1)
        inference = [dict(seed=1, parent=1000, kind=kind, method=method)
                     for kind in ('direct', 'repeat', 'cyclic')
                     for method in ('unprotected', 'hard', 'mixture')]
        representations = ('slow_time_8x1800', 'raw_full', 'dechirped_full',
                           'dechirped_oracle_target_crop', 'raw_long_lag_64_600')
        structure = [dict(seed=1, parent=1000, kind=kind, component=component,
                          representation=representation)
                     for kind in ('direct', 'repeat', 'cyclic')
                     for component in ('s', 'j', 'e', 'y') for representation in representations]
        self.assertEqual(check_rows(inference, structure, manifest),
                         dict(parents=1, inference_rows=9, structure_rows=60))
        with self.assertRaises(ValueError):
            check_rows(inference + inference[:1], structure, manifest)
        with self.assertRaises(ValueError):
            check_rows(inference, structure[:-1], manifest)


if __name__ == '__main__':
    unittest.main()
