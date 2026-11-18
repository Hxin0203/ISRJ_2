"""Recalculate descriptive comparisons from two archived aggregate tables."""
import argparse
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def disagreement_bounds(rows, suite, expected_conditions, parents):
    subset = [r for r in rows if r["suite"] == suite and r["metric"] == "frame_pfa_jammer"]
    require(len(subset) == len({r["index"] for r in subset}) == expected_conditions,
            f"{suite}: incomplete or duplicated conditions")
    lower = upper = net = 0
    for row in subset:
        counts = []
        for model in ("paper2", "rsdr"):
            n = int(row[model + "_parents"])
            k = int(row[model + "_events"])
            require(n == int(row["count"]) == parents and 0 <= k <= n and
                    isclose(k / n, float(row[model + "_mean"]), abs_tol=1e-12),
                    f"{suite}: inconsistent event count")
            counts.append(k)
        a, b = counts
        lower += max(0, a - b)
        upper += min(a, parents - b)
        net += a - b
    total = len(subset) * parents
    return {"conditions": len(subset), "parents_per_condition": parents,
            "correction_only_pct": [100 * lower / total, 100 * upper / total],
            "rsdr_only_pct": [100 * (lower - net) / total, 100 * (upper - net) / total]}


def loss_interactions(rows):
    seeds = sorted(r["seed"] for r in rows)
    require(seeds == [17, 29, 41], "unexpected seed identities")
    result = {}
    for metric in ("frame_pfa_jammer", "pd", "damage"):
        by_seed = {}
        for row in rows:
            effects = []
            for model in ("correction", "unet"):
                before = row[model + "_original"][metric]
                after = row[model + "_tail"][metric]
                require(before["parents"] == after["parents"] == 256, "unexpected DEV count")
                effects.append(after["mean"] - before["mean"])
            diff = effects[0] - effects[1]
            archived = row["loss_effect_correction_minus_unet"][metric]["mean"]
            require(isclose(diff, archived, abs_tol=1e-12), "interaction disagrees with paired archive")
            by_seed[row["seed"]] = diff
        leave_one_out = [mean(by_seed[s] for s in seeds if s != held) for held in seeds]
        result[metric] = {"by_seed": by_seed, "mean": mean(by_seed.values()),
                          "leave_one_seed_out_range": [min(leave_one_out), max(leave_one_out)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path, help="archived evidence/outputs directory")
    root = parser.parse_args().outputs
    table = root / "RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv"
    dev = root / "correction_mf_tail_20261117_v1/FOUR_CELL_DEV_SUMMARY.json"
    with table.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    results = {
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (table, dev)},
        "main": disagreement_bounds(rows, "main", 168, 256),
        "dense": disagreement_bounds(rows, "pd", 310, 512),
        "tail_minus_original_correction_minus_unet": loss_interactions(json.loads(dev.read_bytes())["rows"]),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
