"""Observation-only candidate inference and separate truth-based scoring helpers."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

LENGTH = 1800
FS = 60e6
RATE = 3e12
WIDTH = 600
TIME = np.arange(LENGTH) / FS
DECHIRP = np.exp(-1j * np.pi * RATE * TIME**2)
RIDGE = .01


def energy(value):
    return float(np.vdot(value, value).real)


def target_atom(delay, doppler=0.):
    centered = (np.arange(LENGTH) - delay) / FS - 5e-6
    active = (centered >= -5e-6) & (centered < 5e-6)
    return active * np.exp(1j * (np.pi * RATE * centered**2 + 2 * np.pi * doppler * centered))


def orth(columns):
    if columns.shape[1] == 0:
        return np.empty((len(columns), 0), complex)
    vectors, singular, _ = np.linalg.svd(columns, full_matrices=False)
    return vectors[:, singular > max(1e-12, singular[0] * 1e-10)]


def complement(basis, value):
    return value - basis @ (basis.conj().T @ value)


@dataclass
class Operator:
    dictionary: np.ndarray
    factor: np.ndarray
    protected: np.ndarray

    def apply(self, value):
        return self.dictionary @ (self.factor @ value)


def fit_operator(targets, dictionary, ridge=RIDGE):
    if ridge < 0:
        raise ValueError('ridge must be nonnegative')
    basis = orth(targets)
    norms = np.linalg.norm(dictionary, axis=0)
    normalized = dictionary[:, norms > 1e-12] / norms[norms > 1e-12]
    projected = complement(basis, normalized)
    left, singular, right = np.linalg.svd(projected, full_matrices=False)
    keep = singular > 1e-10 * max(1., np.linalg.norm(normalized))
    gain = np.zeros_like(singular)
    gain[keep] = singular[keep] / (singular[keep]**2 + ridge)
    factor = (right.conj().T * gain) @ left.conj().T
    factor = complement(basis, factor.conj().T).conj().T
    return Operator(normalized, factor, basis)


def target_candidates(observation, count=3):
    """Global public-waveform bank, without target delay or jammer metadata."""
    proposals = []
    for doppler in (-200e3, -100e3, 0., 100e3, 200e3):
        reference = target_atom(0., doppler)[:WIDTH]
        scores = np.abs(fftconvolve(observation, reference[::-1].conj(), mode='valid'))
        for _ in range(count):
            delay = int(scores.argmax())
            proposals.append((float(scores[delay]), delay, doppler))
            scores[max(0, delay - 12):delay + 13] = -1
    selected = []
    for _, delay, doppler in sorted(proposals, reverse=True):
        if any(abs(delay - old_delay) < 12 and abs(doppler - old_doppler) < 150e3
               for old_delay, old_doppler in selected):
            continue
        selected.append((delay, doppler))
        if len(selected) == count:
            break
    return [target_atom(delay, doppler)[:, None] for delay, doppler in selected]


def chirplet_dictionary(projected_observation, count=6):
    """Relaxed short-chirp atoms; not a legal shared periodic replay estimator."""
    residual = projected_observation.copy()
    atoms = []
    for _ in range(count):
        dechirped = residual * DECHIRP
        best_power = -1.
        best_atom = None
        for width in (12, 36, 72, 144):
            starts = np.arange(0, LENGTH - width + 1, max(4, width // 2))
            indices = starts[:, None] + np.arange(width)
            powers = np.abs(np.fft.fft(dechirped[indices], n=256, axis=1))**2 / width
            row, frequency = np.unravel_index(powers.argmax(), powers.shape)
            if powers[row, frequency] > best_power:
                best_power = powers[row, frequency]
                best_atom = np.zeros(LENGTH, complex)
                best_atom[indices[row]] = (np.exp(2j * np.pi * frequency * np.arange(width) / 256)
                                          / DECHIRP[indices[row]] / np.sqrt(width))
        atoms.append(best_atom)
        dictionary = np.stack(atoms, axis=1)
        coefficients = np.linalg.lstsq(dictionary, projected_observation, rcond=1e-10)[0]
        residual = projected_observation - dictionary @ coefficients
    return np.stack(atoms, axis=1)


def operation_for(observation, targets):
    projected = complement(orth(targets), observation)
    dictionary = chirplet_dictionary(projected)
    return fit_operator(targets, dictionary)


@dataclass
class Inference:
    operators: list
    scores: np.ndarray
    corrections: list
    weights: np.ndarray

    def apply(self, value, method):
        if method == 'unprotected':
            return self.operators[0].apply(value)
        if method == 'hard':
            return self.operators[int(self.scores.argmin())].apply(value)
        if method == 'mixture':
            return sum(weight * operator.apply(value)
                       for weight, operator in zip(self.weights, self.operators))
        raise ValueError(method)


def infer(observation):
    """No truth input. Weights are a heuristic, not a calibrated posterior."""
    targets = [np.empty((LENGTH, 0), complex), *target_candidates(observation)]
    operators = []
    corrections = []
    scores = []
    noise_power = max(float(np.median(np.abs(np.diff(observation))**2) / (2 * np.log(2))), 1e-20)
    for candidate in targets:
        operator = operation_for(observation, candidate)
        correction = operator.apply(observation)
        residual = complement(operator.protected, observation - correction)
        if any(energy(correction - previous) <= 1e-12 * max(energy(correction), energy(previous), 1e-20)
               for previous in corrections):
            continue
        operators.append(operator)
        corrections.append(correction)
        scores.append(energy(residual) / noise_power)
    scores = np.asarray(scores)
    weights = np.exp(-(scores - scores.min()))
    weights /= weights.sum()
    return Inference(operators, scores, corrections, weights)


def spectrum_metrics(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    power = singular**2
    if power.sum() < 1e-24:
        return {'top1': 0., 'rank90': 0., 'effective_rank': 0.}
    probability = power / power.sum()
    positive = probability[probability > 0]
    return {'top1': float(probability[0]),
            'rank90': int(np.searchsorted(np.cumsum(probability), .9) + 1),
            'effective_rank': float(np.exp(-np.sum(positive * np.log(positive))))}


def hankel_metrics(signal, rows=64):
    return spectrum_metrics(np.lib.stride_tricks.sliding_window_view(signal, rows).T)


def long_lag_peak(signal):
    total = energy(signal)
    if total < 1e-24:
        return 0.
    correlation = fftconvolve(signal, signal[::-1].conj(), mode='full')[len(signal) - 1:]
    return float(np.abs(correlation[64:601]).max() / total)
