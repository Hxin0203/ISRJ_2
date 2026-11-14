"""Validate and summarize the completed 8-seed, 256-epoch trajectory."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from paper2.dev_256_8seed_v4 import EPOCHS as DEV_EPOCHS, SEEDS, evaluator_sources
from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json

RUN_ID = "paper2_256_epoch_8seed_20261114"
RUN = ROOT / "outputs" / RUN_ID
OLD_RUN = ROOT / "outputs" / "paper2_1103_retry_20261114"
DEV_DATA = ROOT / "data" / "paper2_1103" / "dev.npz"
OUT = ROOT / "evidence" / "outputs" / "CONVERGENCE_256_8SEED_20261114"
TRAIN_METRICS = ("total", "endpoint", "pc", "damage")
DEV_METRICS = (
    ("u_nmse", "Restored-signal NMSE", True),
    ("damage", "Target-increment error", True),
    ("no_jammer_pooled_nmse", "No-jammer pooled NMSE", True),
    ("sjrif_db", "SJRIF (dB)", False),
    ("pd", "Pd: target with jammer", False),
    ("frame_pfa_jammer", "Frame Pfa: jammer + noise", False),
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#E69F00", "#56B4E9", "#000000", "#999999")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_training_rows(complete: dict) -> list[dict]:
    old_complete = read(OLD_RUN / "TRAINING_COMPLETE.json")
    cfg = read(RUN / "CONFIG.json")
    old_history_hashes = cfg["parent_training_history_sha256"]
    rows: list[dict] = []

    for seed in SEEDS:
        if seed in (17, 29, 41):
            history_path = OLD_RUN / f"seed_{seed}" / "history.json"
            if sha(history_path) != old_history_hashes[str(seed)]:
                raise RuntimeError(f"ancestral training history hash mismatch: {history_path}")
            history = read(history_path)
            for epoch in range(1, 129):
                updates = [item for item in history if int(item["epoch"]) == epoch]
                if len(updates) != 48:
                    raise RuntimeError(f"seed {seed} epoch {epoch} has {len(updates)} updates, expected 48")
                rows.append({"seed": seed, "epoch": epoch, "segment": 0,
                             "updates_in_epoch": len(updates),
                             **{metric: statistics.fmean(float(item[metric]) for item in updates)
                                for metric in TRAIN_METRICS},
                             "epoch_elapsed_seconds": None,
                             "record_source": str(history_path.relative_to(ROOT)).replace("\\", "/")})

        folder = RUN / f"seed_{seed}"
        expected_hashes = complete["runs_by_seed"][str(seed)]["training_epoch_record_sha256"]
        candidates: dict[tuple[int, int], Path] = {}
        for path in folder.glob("training_epoch_*_segment_*.json"):
            match = re.fullmatch(r"training_epoch_(\d{3})_segment_(\d{3})", path.stem)
            if not match:
                raise RuntimeError(f"unexpected per-epoch receipt name: {path}")
            epoch, segment = map(int, match.groups())
            key = f"{epoch:03d}_segment_{segment:03d}"
            if key not in expected_hashes or sha(path) != expected_hashes[key]:
                raise RuntimeError(f"per-epoch receipt missing from completion manifest or hash mismatch: {path}")
            receipt = read(path)
            if int(receipt["seed"]) != seed or int(receipt["epoch"]) != epoch:
                raise RuntimeError(f"per-epoch receipt identity mismatch: {path}")
            candidates[(epoch, segment)] = path

        for epoch in range(1, 257):
            if seed in (17, 29, 41) and epoch <= 128:
                continue
            matching = [(segment, path) for (row_epoch, segment), path in candidates.items()
                        if row_epoch == epoch and all(metric in read(path) for metric in TRAIN_METRICS)]
            if not matching:
                raise RuntimeError(f"missing training metrics for seed {seed}, epoch {epoch}")
            segment, path = max(matching, key=lambda item: item[0])
            receipt = read(path)
            if int(receipt["updates_this_epoch"]) != 48:
                raise RuntimeError(f"seed {seed} epoch {epoch} contains an incomplete update count")
            rows.append({"seed": seed, "epoch": epoch, "segment": segment,
                         "updates_in_epoch": int(receipt["updates_this_epoch"]),
                         **{metric: float(receipt[metric]) for metric in TRAIN_METRICS},
                         "epoch_elapsed_seconds": float(receipt["epoch_elapsed_seconds"]),
                         "record_source": str(path.relative_to(ROOT)).replace("\\", "/")})

    keys = {(int(row["seed"]), int(row["epoch"])) for row in rows}
    expected = {(seed, epoch) for seed in SEEDS for epoch in range(1, 257)}
    if keys != expected or len(rows) != len(expected):
        raise RuntimeError(f"training trajectory coverage mismatch: got {len(keys)}, expected {len(expected)}")
    return sorted(rows, key=lambda row: (row["seed"], row["epoch"]))


def _validate_training_complete() -> dict:
    receipt_path = RUN / "TRAINING_COMPLETE.json"
    receipt = read(receipt_path)
    cfg_sha = sha(RUN / "CONFIG.json")
    data_sha = sha(ROOT / "data" / "paper2_1103" / "train.npz")
    if receipt.get("run_id") != RUN_ID or receipt.get("config_sha256") != cfg_sha:
        raise RuntimeError("overall completion receipt does not match the frozen run config")
    if receipt.get("dataset_sha256") != data_sha or receipt.get("seeds") != list(SEEDS):
        raise RuntimeError("overall completion receipt does not match dataset or seed roster")
    runs = receipt.get("runs", [])
    if {int(run["seed"]) for run in runs} != set(SEEDS) or len(runs) != len(SEEDS):
        raise RuntimeError("overall completion receipt does not cover eight unique seeds")
    by_seed = {str(run["seed"]): run for run in runs}
    for seed in SEEDS:
        run = by_seed[str(seed)]
        folder = RUN / f"seed_{seed}"
        if read(folder / "TRAINING_COMPLETE.json") != run:
            raise RuntimeError(f"seed-level and overall completion receipts disagree for seed {seed}")
        events = folder / "events.jsonl"
        if not events.is_file() or sha(events) != run.get("training_events_sha256"):
            raise RuntimeError(f"training event log missing or hash mismatch for seed {seed}")
        for name, digest in run.get("recovery_archives_sha256", {}).items():
            if sha(folder / name) != digest:
                raise RuntimeError(f"per-epoch recovery archive hash mismatch: {folder / name}")
        for epoch in DEV_EPOCHS:
            cp = folder / f"epoch_{epoch:03d}.pt"
            milestone = run["milestones"].get(str(epoch))
            if not cp.is_file() or sha(cp) != milestone:
                raise RuntimeError(f"milestone checkpoint verification failed: {cp}")
            cp_receipt = read(cp.with_suffix(".json"))
            if cp_receipt.get("checkpoint_sha256") != milestone:
                raise RuntimeError(f"milestone receipt mismatch: {cp}")
    receipt["runs_by_seed"] = by_seed
    return receipt


def _load_dev_rows(complete: dict) -> list[dict]:
    expected_data_sha = sha(DEV_DATA)
    expected_sources = evaluator_sources()
    rows = []
    for seed in SEEDS:
        folder = RUN / f"seed_{seed}"
        for epoch in DEV_EPOCHS:
            cp = folder / f"epoch_{epoch:03d}.pt"
            path = folder / f"dev_epoch_{epoch:03d}_v4.json"
            result = read(path)
            if (result.get("seed") != seed or result.get("epoch") != epoch
                    or result.get("checkpoint_sha256") != sha(cp)
                    or result.get("data_sha256") != expected_data_sha
                    or result.get("evaluator_source_sha256") != expected_sources):
                raise RuntimeError(f"DEV checkpoint/evaluator identity mismatch: {path}")
            raw = folder / f"dev_epoch_{epoch:03d}_v4.npz"
            if not raw.is_file() or sha(raw) != result.get("local_parent_values_sha256"):
                raise RuntimeError(f"per-parent DEV evidence missing or damaged: {raw}")
            row = {"seed": seed, "epoch": epoch,
                   "checkpoint_sha256": result["checkpoint_sha256"],
                   "result_sha256": sha(path), "parents": int(result["parents"])}
            for name, metric in result["metrics"]["model"].items():
                row[f"model__{name}__mean"] = metric["mean"]
                row[f"model__{name}__median"] = metric["median"]
                row[f"model__{name}__p95"] = metric["p95"]
                row[f"model__{name}__parent_bootstrap_ci95_low"] = metric["parent_bootstrap_ci95"][0]
                row[f"model__{name}__parent_bootstrap_ci95_high"] = metric["parent_bootstrap_ci95"][1]
            for name, metric in result["model_minus_keep"].items():
                row[f"paired_model_minus_keep__{name}__mean"] = metric["mean"]
                row[f"paired_model_minus_keep__{name}__parent_bootstrap_ci95_low"] = metric["parent_bootstrap_ci95"][0]
                row[f"paired_model_minus_keep__{name}__parent_bootstrap_ci95_high"] = metric["parent_bootstrap_ci95"][1]
            rows.append(row)
    if len(rows) != len(SEEDS) * len(DEV_EPOCHS):
        raise RuntimeError("DEV trajectory does not include exactly 64 seed-checkpoint evaluations")
    return rows


def plot_training(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    epochs = np.arange(1, 257)
    for ax, metric in zip(axes.flat, TRAIN_METRICS):
        seed_values = []
        for index, seed in enumerate(SEEDS):
            selected = [row for row in rows if row["seed"] == seed]
            x = np.array([row["epoch"] for row in selected], dtype=int)
            y = np.array([row[metric] for row in selected], dtype=float)
            seed_values.append(y)
            ax.plot(x, y, color=COLORS[index], linewidth=0.9, alpha=0.72,
                    linestyle="-" if index < 3 else "--",
                    label=f"Seed {seed}" if metric == TRAIN_METRICS[0] else None)
        matrix = np.stack(seed_values)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        ax.plot(epochs, mean, color="#222222", linewidth=2.2,
                label="Eight-seed mean ± SD" if metric == TRAIN_METRICS[0] else None)
        ax.fill_between(epochs, np.maximum(mean - sd, np.finfo(float).tiny), mean + sd,
                        color="#555555", alpha=0.15)
        ax.set_yscale("log")
        ax.set_title({"total": "Total B4 objective", "endpoint": "Endpoint loss",
                      "pc": "Pulse-compression loss", "damage": "Target-increment loss"}[metric])
        ax.set(xlabel="Training epoch", ylabel="Mean loss per epoch (log scale)", xlim=(1, 256))
        ax.grid(True, which="both", alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.suptitle("CorrectionNet training trajectory: 8 seeds × 256 epochs\n"
                 "Solid lines: three continued seeds; dashed lines: five new seeds; "
                 "shading: mean ± SD across seeds")
    fig.savefig(OUT / "TRAINING_CONVERGENCE_8SEED.png", dpi=300)
    fig.savefig(OUT / "TRAINING_CONVERGENCE_8SEED.pdf")
    plt.close(fig)


def plot_dev(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, layout="constrained")
    for ax, (metric, title, log_scale) in zip(axes.flat, DEV_METRICS):
        field = f"model__{metric}__mean"
        seed_values = []
        epochs = np.asarray(DEV_EPOCHS)
        for index, seed in enumerate(SEEDS):
            selected = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["epoch"])
            y = np.asarray([row[field] for row in selected], dtype=float)
            seed_values.append(y)
            ax.plot(epochs, y, color=COLORS[index], marker=MARKERS[index], markersize=3.4,
                    linewidth=0.9, alpha=0.75, linestyle="-" if index < 3 else "--",
                    label=f"Seed {seed}" if metric == DEV_METRICS[0][0] else None)
        matrix = np.stack(seed_values)
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        ax.plot(epochs, mean, color="#222222", marker="D", markersize=4.0, linewidth=2.0,
                label="Eight-seed mean ± SD" if metric == DEV_METRICS[0][0] else None)
        lower = np.maximum(mean - sd, np.finfo(float).tiny) if log_scale else mean - sd
        ax.fill_between(epochs, lower, mean + sd, color="#555555", alpha=0.14)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(epochs)
        ax.set_xlabel("Checkpoint epoch")
        ax.grid(True, which="both", alpha=0.22)
    axes[0, 0].set_ylabel("Parent-level metric")
    axes[1, 0].set_ylabel("Parent-level metric")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.suptitle("DEV v4 convergence at 32-epoch checkpoints\n"
                 "Shading: descriptive mean ± SD across eight training seeds; "
                 "not a confidence interval")
    fig.savefig(OUT / "DEV_CONVERGENCE_8SEED.png", dpi=300)
    fig.savefig(OUT / "DEV_CONVERGENCE_8SEED.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    complete = _validate_training_complete()
    training_rows = _load_training_rows(complete)
    dev_rows = _load_dev_rows(complete)
    training_fields = ["seed", "epoch", "segment", "updates_in_epoch", *TRAIN_METRICS,
                       "epoch_elapsed_seconds", "record_source"]
    write_csv(OUT / "TRAINING_LOSS_BY_EPOCH.csv", training_rows, training_fields)
    dev_fields = list(dict.fromkeys(key for row in dev_rows for key in row))
    write_csv(OUT / "DEV_METRICS_BY_CHECKPOINT.csv", dev_rows, dev_fields)
    plot_training(training_rows)
    plot_dev(dev_rows)

    summary = {"created_utc": now(), "run_id": RUN_ID,
               "training_seeds": list(SEEDS), "training_epochs": 256,
               "dev_checkpoint_epochs": list(DEV_EPOCHS), "dev_seed_checkpoint_pairs": len(dev_rows),
               "training": {}, "dev": {}}
    for metric in TRAIN_METRICS:
        summary["training"][metric] = {}
        for epoch in range(1, 257):
            values = [float(row[metric]) for row in training_rows if row["epoch"] == epoch]
            summary["training"][metric][str(epoch)] = {
                "mean_across_seeds": statistics.fmean(values),
                "sd_across_seeds": statistics.stdev(values), "n_seeds": len(values)}
    for metric, _, _ in DEV_METRICS:
        summary["dev"][metric] = {}
        field = f"model__{metric}__mean"
        for epoch in DEV_EPOCHS:
            values = [float(row[field]) for row in dev_rows if row["epoch"] == epoch]
            summary["dev"][metric][str(epoch)] = {
                "mean_across_seeds": statistics.fmean(values),
                "sd_across_seeds": statistics.stdev(values), "n_seeds": len(values),
                "uncertainty_note": "descriptive training-seed SD; per-seed parent bootstrap CIs are in the CSV"}
    (OUT / "CONVERGENCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    input_paths = [RUN / "CONFIG.json", RUN / "TRAINING_COMPLETE.json", DEV_DATA,
                   ROOT / "tools" / "aggregate_convergence_256_8seed.py"]
    for seed in SEEDS:
        seed_folder = RUN / f"seed_{seed}"
        input_paths.extend((seed_folder / "TRAINING_COMPLETE.json", seed_folder / "events.jsonl"))
        # Include every immutable epoch receipt from every recovery segment,
        # including superseded attempts retained after a verified rollback.
        input_paths.extend(sorted(seed_folder.glob("training_epoch_*_segment_*.json")))
    input_paths.extend(RUN / f"seed_{seed}" / f"dev_epoch_{epoch:03d}_v4.json"
                       for seed in SEEDS for epoch in DEV_EPOCHS)
    manifest = {
        "created_utc": now(), "run_id": RUN_ID,
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                          for path in input_paths},
        "evaluator_source_sha256": evaluator_sources(),
        "analysis_script_sha256": sha(ROOT / "tools" / "aggregate_convergence_256_8seed.py"),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "independent_training_seed_count": 8,
        "physical_dev_parent_count_per_checkpoint": 256,
        "uncertainty": "Training trajectory band is mean ± sample SD over 8 seeds; per-seed DEV bootstrap resamples physical parents.",
        "test_data_accessed": False, "held_out_gate_arrays_accessed": False,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 8-seed、256 轮收敛结果\n\n"
        "训练图按轮展示 B4 总损失及分量；实线为从原始 128 轮检查点续训的 seed 17/29/41，"
        "虚线为五个从头训练的新 seed。灰带是八个训练 seed 间的描述性均值 ± 样本标准差。\n\n"
        "DEV 图只在 32、64、96、128、160、192、224、256 轮检查点上评估。阴影同样是 seed 间标准差，"
        "不是置信区间；逐 seed 的 parent bootstrap 区间保存在 CSV。DEV 估计以 256 个物理 parent 为单位，"
        "不把脉冲或 jammer 变体当作独立样本。\n\n"
        "文件包括两张 PNG/PDF 图、逐 seed CSV、跨 seed 汇总 JSON 和 SHA-256 manifest。训练权重、"
        "训练数组与每 parent 的完整 DEV 数组留在本地 outputs/。未读取 TEST 或旧项目 held-out Gate 数组。\n",
        encoding="utf-8")
    print(f"Validated eight seeds and wrote convergence artifacts to {OUT}")


if __name__ == "__main__":
    main()
