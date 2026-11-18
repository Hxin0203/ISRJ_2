"""Exact checks of two conditional squared-error identities on scalar examples."""

from fractions import Fraction
from itertools import product


def compare(probability, outputs):
    # All probabilities refer to one fixed observation.
    posterior = list(zip((1 - probability, probability), outputs))
    mean = sum(weight * value for weight, value in posterior)
    mean_risk = sum(weight * (value - mean) ** 2 for weight, value in posterior)
    draw_risk = sum(
        p * q * (value - draw) ** 2
        for (p, value), (q, draw) in product(posterior, repeat=2)
    )
    assert draw_risk == 2 * mean_risk
    return mean, mean_risk, draw_risk


if __name__ == "__main__":
    for p in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
        for outputs in ((1, 1), (-1, 1)):
            mean, mean_risk, draw_risk = compare(p, outputs)
            assert mean_risk == p * (1 - p) * (outputs[1] - outputs[0]) ** 2
            print(f"p={p} outputs={outputs} mean={mean} risks={mean_risk},{draw_risk}")
