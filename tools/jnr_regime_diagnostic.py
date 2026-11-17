"""Re-read frozen TEST summaries by jammer-to-noise ratio (JNR = SNR + JSR).

Offline post-analysis only: no model, weights, raw arrays or training are touched.
Inputs are the two tracked per-condition summary tables; outputs are a small CSV
and a JSON receipt with input SHA-256 values.

In the target-absent jammed state (J+E) the input depends on SNR and JSR only
through their sum, because the target is absent and the scenes use common random
numbers.  Grouping by JNR therefore separates the regimes in which the residual
false alarms arise.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITION_TABLE = ROOT / "evidence/outputs/TEST_CONDITION_TABLE_1103.csv"
RSDR_TABLE = ROOT / "evidence/outputs/RSDR_UNIFIED_20261115/CONDITION_COMPARISONS.csv"
OUT_DIR = ROOT / "evidence/outputs/JNR_REGIME_DIAGNOSTIC_20261117"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text: str) -> float:
    return float(text) if text not in ("", "nan", "NaN") else math.nan


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def load_methods(suite: str) -> dict:
    """{(jam_type, snr, jsr): {method: row}} from the frozen condition table."""
    table: dict = collections.defaultdict(dict)
    with CONDITION_TABLE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["suite"] == suite:
                key = (row["jam_type"], int(float(row["snr_db"])), int(float(row["jsr_db"])))
                table[key][row["method"]] = row
    return table


def load_rsdr(suite: str) -> dict:
    """{(jam_type, snr, jsr): {metric: (paper2_mean, rsdr_mean)}} from the unified comparison."""
    table: dict = collections.defaultdict(dict)
    with RSDR_TABLE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["suite"] == suite:
                key = (row["jam_type"], int(float(row["snr_db"])), int(float(row["jsr_db"])))
                table[key][row["metric"]] = (number(row["paper2_mean"]), number(row["rsdr_mean"]))
    return table


def by_jnr(suite: str, jam_types: tuple[str, ...]) -> list[dict]:
    methods, rsdr = load_methods(suite), load_rsdr(suite)
    groups: dict = collections.defaultdict(list)
    for (jam_type, snr, jsr) in methods:
        if jam_type in jam_types:
            groups[snr + jsr].append((jam_type, snr, jsr))
    rows = []
    for jnr in sorted(groups):
        keys = groups[jnr]

        def method_mean(method: str, column: str) -> float:
            return mean([number(methods[k][method][column]) for k in keys if method in methods[k]])

        def rsdr_mean(metric: str) -> float:
            return mean([rsdr[k][metric][1] for k in keys if metric in rsdr.get(k, {})])

        model_fa = [number(methods[k]["model"]["frame_pfa_jammer"]) for k in keys]
        # Same J+E shape, different absolute input scale (scale = 10^(-SNR/20)).
        rsdr_fa = [rsdr[k]["frame_pfa_jammer"][1] for k in keys if "frame_pfa_jammer" in rsdr.get(k, {})]
        single = len(jam_types) == 1 and bool(rsdr_fa)
        keep_bg, model_bg = method_mean("keep", "background_nmse"), method_mean("model", "background_nmse")
        rsdr_bg = rsdr_mean("background_nmse")
        rows.append(dict(
            suite=suite, jam_types="+".join(jam_types), jnr_db=jnr, conditions=len(keys),
            frame_fa_keep=method_mean("keep", "frame_pfa_jammer"),
            frame_fa_correctionnet=method_mean("model", "frame_pfa_jammer"),
            frame_fa_rsdr=rsdr_mean("frame_pfa_jammer"),
            frame_fa_max_tf=method_mean("max_tf", "frame_pfa_jammer"),
            frame_fa_bcs_oracle=method_mean("bcs_oracle", "frame_pfa_jammer"),
            frame_fa_reference=method_mean("reference", "frame_pfa_jammer"),
            # ||T(J+E)-E||^2 / ||E||^2 ; for KEEP this is the full-window J/E energy ratio.
            background_nmse_keep=keep_bg,
            background_nmse_correctionnet=model_bg,
            background_nmse_rsdr=rsdr_bg,
            residual_over_jammer_db_correctionnet=10 * math.log10(model_bg / keep_bg) if keep_bg > 0 and model_bg > 0 else math.nan,
            residual_over_jammer_db_rsdr=10 * math.log10(rsdr_bg / keep_bg) if keep_bg > 0 and rsdr_bg > 0 else math.nan,
            # Zero spread confirms that the J+E state is a function of JNR alone.
            correctionnet_fa_spread_same_jnr=max(model_fa) - min(model_fa) if len(jam_types) == 1 else math.nan,
            # RSDR-Net is not scale-normalised; its J+E alarms change with absolute scale.
            rsdr_fa_min_same_jnr=min(rsdr_fa) if single else math.nan,
            rsdr_fa_max_same_jnr=max(rsdr_fa) if single else math.nan,
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = by_jnr("pd", ("repeat",)) + by_jnr("main", ("direct", "repeat", "cyclic"))
    for jam_type in ("direct", "repeat", "cyclic"):
        rows += by_jnr("main", (jam_type,))
    with (args.out / "FA_BY_JNR.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})

    dense = [r for r in rows if r["suite"] == "pd"]
    spread = max(r["correctionnet_fa_spread_same_jnr"] for r in dense)
    picks = {r["jnr_db"]: r for r in dense if r["jnr_db"] in (-10, -5, 0, 5, 10, 15, 20, 25, 30, 35)}
    summary = dict(
        created="2026-11-17",
        scope="offline re-read of frozen TEST summaries; CorrectionNet seed 29 epoch 128; RSDR-Net F-R7",
        definition="JNR_dB = SNR_dB + JSR_dB (active-support power convention of the main generator)",
        inputs={
            str(CONDITION_TABLE.relative_to(ROOT)).replace("\\", "/"): sha256(CONDITION_TABLE),
            str(RSDR_TABLE.relative_to(ROOT)).replace("\\", "/"): sha256(RSDR_TABLE),
        },
        dense_grid_max_fa_spread_at_equal_jnr=spread,
        dense_grid_repeat_512_parents={
            str(jnr): dict(
                frame_fa_percent=dict(
                    keep=round(100 * r["frame_fa_keep"], 2),
                    correctionnet=round(100 * r["frame_fa_correctionnet"], 2),
                    rsdr=round(100 * r["frame_fa_rsdr"], 2),
                    rsdr_range_over_input_scale=[round(100 * r["rsdr_fa_min_same_jnr"], 2),
                                                 round(100 * r["rsdr_fa_max_same_jnr"], 2)],
                    reference=round(100 * r["frame_fa_reference"], 2),
                ),
                residual_over_jammer_db=dict(
                    correctionnet=round(r["residual_over_jammer_db_correctionnet"], 2),
                    rsdr=round(r["residual_over_jammer_db_rsdr"], 2),
                ),
                background_nmse=dict(
                    keep=float(f"{r['background_nmse_keep']:.4g}"),
                    correctionnet=float(f"{r['background_nmse_correctionnet']:.4g}"),
                    rsdr=float(f"{r['background_nmse_rsdr']:.4g}"),
                ),
            )
            for jnr, r in sorted(picks.items())
        },
        limits=[
            "single frozen seed and checkpoint per model; simulated single-target scenes only",
            "TEST was already analysed; this re-read is diagnostic, not a new confirmation",
            "equal-parameter U-Net and ablations have no per-JNR table yet",
            "jammer parameters are drawn independently per pulse in the generator",
            "RSDR-Net values at one JNR average conditions that differ only in absolute input scale",
        ],
    )
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["dense_grid_repeat_512_parents"], indent=1))
    print("max FA spread at equal JNR (dense grid):", spread)


if __name__ == "__main__":
    main()
