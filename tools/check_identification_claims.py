"""Exact finite counterexamples for two algebraic claims. No project data required."""
from fractions import Fraction
from itertools import product
import json


def four_cell_check():
    counts = {"both": 0, "interaction_only": 0, "reference_only": 0, "neither": 0}
    for baseline, treated, second, both in product((-1, 0, 1), repeat=4):
        change_alone = treated - baseline
        change_together = both - second
        interaction = change_together == change_alone
        reference = change_together == 1
        if interaction and reference:
            counts["both"] += 1
        elif interaction:
            counts["interaction_only"] += 1
        elif reference:
            counts["reference_only"] += 1
        else:
            counts["neither"] += 1
    if sum(counts.values()) != 81 or min(counts["interaction_only"], counts["reference_only"]) == 0:
        raise AssertionError("Expected a counterexample to each proposed implication")
    return counts


def near_collision_check():
    # Use one injective map g(x) = 1/x on [1, infinity) across every pair.
    examples = []
    for first in (1, 10, 100, 1000):
        second = first + 1
        left, right = Fraction(1, first), Fraction(1, second)
        gap = left - right
        if gap != Fraction(1, first * second) or 1 / right - 1 / left != 1:
            raise AssertionError("Inverse failed")
        examples.append({"states": [first, second], "state_distance": 1,
                         "observation_distance": str(gap)})
    return examples


if __name__ == "__main__":
    print(json.dumps({"four_cell_tables": four_cell_check(),
                      "injective_examples": near_collision_check()}, indent=2))
