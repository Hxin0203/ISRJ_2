"""Plot the saved Paper-2 1103 training and DEV checkpoint trajectories.

The script reads local training histories and lightweight DEV aggregates. It
does not access TEST data, held-out Gate arrays, or model weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "paper2_1103_retry_20261114"
DEV_CSV = ROOT / "evidence" / "outputs" / "paper2_1103_retry_20261114" / "DEV_SELECTION_TABLE_v4.csv"
OUT = ROOT / "evidence" / "outputs" / "CONVERGENCE_1103_20261114"
SEEDS = (17, 29, 41)
TRAIN_METRICS = ("total", "endpoint", "pc", "damage")
DEV_METRICS = (
    ("u_nmse", "eU NMSE", True),
    ("damage", "eS target-increment error", True),
    ("no_jammer_pooled_nmse", "No-jammer pooled NMSE", True),
    ("sjrif_db", "SJRIF (dB)", False),
    ("pd", "Pd (jammer + target)", False),
    ("frame_pfa_jammer", "Frame Pfa (jammer + noise)", False),
)
COLORS = {17: "#0072B2", 29: "#D55E00", 41: "#009E73"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_rows() -> list[dict[str, float | int]]:
    rows = []
    for seed in SEEDS:
        history_path = RUN / f"seed_{seed}" / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for epoch in sorted({int(item["epoch"]) for item in history}):
            updates = [item for item in history if int(item["epoch"]) == epoch]
            row: dict[str, float | int] = {
                "seed": seed,
                "epoch": epoch,
                "updates_in_epoch": len(updates),
            }
            row.update({name: statistics.fmean(float(item[name]) for item in updates)
                        for name in TRAIN_METRICS})
            rows.append(row)
    return rows


def load_dev_rows() -> list[dict[str, float | int]]:
    with DEV_CSV.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    return [{
        "seed": int(row["seed"]),
        "epoch": int(row["epoch"]),
        "u_nmse": float(row["u_nmse"]),
        "damage": float(row["damage"]),
        "no_jammer_pooled_nmse": float(row["no_jammer_pooled_nmse"]),
        "sjrif_db": float(row["sjrif_db"]),
        "pd": float(row["pd"]),
        "pd_no_jammer": float(row["pd_no_jammer"]),
        "frame_pfa_jammer": float(row["frame_pfa_jammer"]),
    } for row in raw]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_training(rows: list[dict[str, float | int]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    for ax, metric in zip(axes.flat, TRAIN_METRICS):
        for seed in SEEDS:
            chosen = [r for r in rows if r["seed"] == seed]
            ax.plot([r["epoch"] for r in chosen], [r[metric] for r in chosen],
                    color=COLORS[seed], alpha=0.68, linewidth=1.15,
                    label=f"Seed {seed}" if metric == TRAIN_METRICS[0] else None)
        epochs = sorted({int(r["epoch"]) for r in rows})
        means = [statistics.fmean(float(r[metric]) for r in rows if r["epoch"] == epoch)
                 for epoch in epochs]
        ax.plot(epochs, means, color="#222222", linewidth=2.25,
                label="Three-seed mean" if metric == TRAIN_METRICS[0] else None)
        ax.set_yscale("log")
        ax.set_title({"total": "Total objective", "endpoint": "Endpoint loss",
                      "pc": "Pulse-compression loss", "damage": "Target-response loss"}[metric])
        ax.set_ylabel("Mean training loss / epoch (log scale)")
        ax.grid(True, which="both", alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel("Training epoch")
        ax.set_xlim(1, 128)
    fig.suptitle("Paper-2 CorrectionNet training convergence\n"
                 "48 optimizer updates per epoch; lines show epoch-mean loss", y=1.02)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "TRAINING_CONVERGENCE.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "TRAINING_CONVERGENCE.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dev(rows: list[dict[str, float | int]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, (metric, title, log_scale) in zip(axes.flat, DEV_METRICS):
        epochs = sorted({int(r["epoch"]) for r in rows})
        for seed in SEEDS:
            chosen = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["epoch"])
            ax.plot([r["epoch"] for r in chosen], [r[metric] for r in chosen],
                    marker="o", markersize=4, color=COLORS[seed], alpha=0.72,
                    linewidth=1.15, label=f"Seed {seed}" if metric == DEV_METRICS[0][0] else None)
        means = [statistics.fmean(float(r[metric]) for r in rows if r["epoch"] == epoch)
                 for epoch in epochs]
        sds = [statistics.stdev(float(r[metric]) for r in rows if r["epoch"] == epoch)
               for epoch in epochs]
        ax.plot(epochs, means, color="#222222", linewidth=2.3, marker="D", markersize=4,
                label="Mean ± SD across seeds" if metric == DEV_METRICS[0][0] else None)
        ax.fill_between(epochs, np.asarray(means) - np.asarray(sds),
                        np.asarray(means) + np.asarray(sds), color="#555555", alpha=0.13)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(epochs)
        ax.set_xlabel("Checkpoint epoch")
        ax.grid(True, which="both", alpha=0.22)
    fig.suptitle("DEV checkpoint trajectory (four evaluations only; not an epoch-wise validation curve)\n"
                 "Shading: descriptive ±1 SD across three seeds", y=1.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "DEV_CHECKPOINT_TRAJECTORY.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "DEV_CHECKPOINT_TRAJECTORY.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_rows = load_training_rows()
    dev_rows = load_dev_rows()
    train_fields = ["seed", "epoch", "updates_in_epoch", *TRAIN_METRICS]
    dev_fields = ["seed", "epoch", *(metric[0] for metric in DEV_METRICS), "pd_no_jammer"]
    write_csv(OUT / "TRAINING_LOSS_BY_EPOCH.csv", train_rows, train_fields)
    write_csv(OUT / "DEV_METRICS_BY_CHECKPOINT.csv", dev_rows, dev_fields)
    plot_training(train_rows)
    plot_dev(dev_rows)

    summary = {}
    for epoch in sorted({int(r["epoch"]) for r in dev_rows}):
        at_epoch = [r for r in dev_rows if r["epoch"] == epoch]
        summary[str(epoch)] = {
            metric[0]: {
                "mean": statistics.fmean(float(r[metric[0]]) for r in at_epoch),
                "sd_across_seeds": statistics.stdev(float(r[metric[0]]) for r in at_epoch),
            }
            for metric in DEV_METRICS
        }
    (OUT / "DEV_MEAN_AND_SD.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    inputs = [DEV_CSV, *(RUN / f"seed_{seed}" / "history.json" for seed in SEEDS)]
    manifest = {
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "plot_script_sha256": sha256(Path(__file__)),
        "training_history_records_per_seed": 6144,
        "training_epochs": 128,
        "updates_per_epoch": 48,
        "dev_checkpoint_epochs": [32, 64, 96, 128],
        "dev_independent_parent_count": 256,
        "test_data_accessed": False,
        "held_out_gate_arrays_accessed": False,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 1103 训练收敛与 DEV 检查点轨迹\n\n"
        "训练图使用三种 seed 的完整逐更新历史，并按 epoch 求均值后绘制训练目标及分量。"
        "DEV 图只使用 32、64、96、128 轮的四个冻结检查点；阴影是三个 seed 间的描述性标准差，"
        "不是统计置信区间。DEV 图不代表逐 epoch 验证。\n\n"
        "数据文件：`TRAINING_LOSS_BY_EPOCH.csv`、`DEV_METRICS_BY_CHECKPOINT.csv`、"
        "`DEV_MEAN_AND_SD.json`。图文件同时提供 PNG 与 PDF。原始训练历史和权重仍留在 outputs/；"
        "未读取独立 TEST 或旧项目 held-out Gate 数组。\n\n"
        f"输入与代码 SHA-256 记录于 `MANIFEST.json`。\n",
        encoding="utf-8")
    print(f"Wrote convergence artifacts to {OUT}")


if __name__ == "__main__":
    main()
