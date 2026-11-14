"""Verify completed TEST blocks and summarize common-grid paired comparisons.

Equal weighting across completed cells is descriptive. Parent means, not cells
or pulses, are resampled. Partial common-grid results are explicitly labelled.
Raw arrays remain under outputs/; only aggregate JSON/CSV enters evidence/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from paper2.evaluation_1103_v4 import summarize
from paper2.io import ROOT, now, read, sha, source_bindings, write

RUNS = ("paper2_1103_test_20261114", "paper2_1103_test_classical_20261114")
METRICS = ("u_nmse", "damage", "sjrif_db", "false_peak_suppression_db", "pd", "pd_no_jammer",
           "no_jammer_pooled_nmse", "frame_pfa_noise", "frame_pfa_jammer", "cell_pfa_noise", "cell_pfa_jammer")
LIGHT = ROOT / "evidence/outputs"


def load_verified():
    frozen_path = ROOT / "configs/paper2_1103_scan_freeze.json"
    frozen = read(frozen_path)
    assert frozen["sources"] == source_bindings(), "frozen source identity changed"
    assert sha(ROOT / frozen["selected_checkpoint"]) == frozen["checkpoint_sha256"]
    records = {}
    failures = []
    blocks_checked = 0
    for run in RUNS:
        root = ROOT / "outputs" / run
        failures.extend(str(p.relative_to(ROOT)) for p in root.glob("FAILURE_*.json"))
        for p in sorted(root.glob("*/*/result.json")):
            r = read(p); identity = r["identity"]
            suite, index, method = identity["suite"], identity["index"], identity["method"]
            assert r["status"] == "COMPLETE" and r["scope"] == "INDEPENDENT_TEST"
            assert identity["binding"]["freeze_sha256"] == sha(frozen_path)
            assert identity["condition"] == frozen["grids"][suite][index]
            parts, ids = [], []
            for block in r["blocks"]:
                path = root / block["file"]
                assert sha(path) == block["sha256"], path
                receipt = read(path.with_suffix(".json"))
                assert receipt["sha256"] == block["sha256"]
                ids.extend(m["parent_id"] for m in receipt["metadata"])
                with np.load(path, allow_pickle=False) as z:
                    parts.append({k: z[k].copy() for k in METRICS})
                blocks_checked += 1
            values = {k: np.concatenate([b[k] for b in parts]) for k in METRICS}
            n = identity["condition"]["count"]
            assert len(ids) == len(set(ids)) == n
            for k, v in values.items():
                assert len(v) == n and np.isfinite(v).all()
                assert np.isclose(v.mean(), r["metrics"][k]["mean"], rtol=1e-12, atol=1e-14)
            key = (suite, index, method)
            assert key not in records, key
            records[key] = dict(result=r, values=values, ids=ids)
    return frozen, records, blocks_checked, failures


def pair_grid(records, suite, methods, indexes):
    if not indexes:
        return None
    n = len(records[suite, indexes[0], methods[0]]["ids"])
    base_ids = records[suite, indexes[0], methods[0]]["ids"]
    per_method = {}
    for method in methods:
        for index in indexes:
            assert records[suite, index, method]["ids"] == base_ids
        per_method[method] = {
            k: np.mean([records[suite, i, method]["values"][k] for i in indexes], axis=0)
            for k in METRICS}
    metrics = {method: {k: summarize(v) for k, v in vals.items()} for method, vals in per_method.items()}
    differences = {}
    if "model" in methods:
        for method in methods:
            if method == "model": continue
            differences[f"model_minus_{method}"] = {
                k: summarize(per_method["model"][k] - per_method[method][k]) for k in METRICS}
    return dict(condition_indexes=indexes, independent_parents=n,
                weighting="equal cell weighting, followed by 4000 bootstrap resamples of physical-parent means",
                metrics=metrics, paired_differences=differences)


def main():
    frozen, records, blocks_checked, failures = load_verified()
    groups = {}
    for competitor in ("keep", "reference", "max_tf", "gmm_omp", "bcs_oracle"):
        for jam in ("all", "direct", "repeat", "cyclic"):
            eligible = [i for i,c in enumerate(frozen["grids"]["main"]) if jam == "all" or c["jam_type"] == jam]
            indexes = [i for i in eligible if ("main",i,"model") in records and ("main",i,competitor) in records]
            paired = pair_grid(records,"main",("model",competitor),indexes)
            if paired is not None:
                paired.update(expected_conditions=len(eligible), complete=len(indexes)==len(eligible),
                              analysis_intent="descriptive common-grid summary; selection does not alter frozen model")
                groups[f"main_{jam}_model_vs_{competitor}"] = paired
    coverage = {s:{m:sum((s,i,m) in records for i in range(len(grid))) for m in frozen["methods"][s]}
                for s,grid in frozen["grids"].items()}
    complete = all(n==len(frozen["grids"][s]) for s,methods in coverage.items() for n in methods.values())
    result = dict(checked_utc=now(), all_expected_complete=complete, coverage=coverage,
                  complete_method_conditions=len(records), blocks_verified=blocks_checked,
                  failure_records=failures, groups=groups,
                  note="Grid cells and pulses are repeated measurements, not extra independent parents. Probability means across grid are descriptive; no pooled-cell binomial CI.")
    write(LIGHT / "TEST_PAIRED_AGGREGATES_1103.json",result)
    rows=[]
    for group,g in groups.items():
        for comparison,vals in g["paired_differences"].items():
            for metric,v in vals.items():
                rows.append(dict(group=group,comparison=comparison,metric=metric,conditions=len(g["condition_indexes"]),
                    expected_conditions=g["expected_conditions"],complete=g["complete"],parents=g["independent_parents"],
                    mean_difference=v["mean"],ci95_lower=v["parent_bootstrap_ci95"][0],ci95_upper=v["parent_bootstrap_ci95"][1]))
    if rows:
        with (LIGHT / "TEST_PAIRED_DIFFERENCES_1103.csv").open("w",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print({k:result[k] for k in ("checked_utc","all_expected_complete","complete_method_conditions","blocks_verified","failure_records")})


if __name__ == "__main__":
    main()
