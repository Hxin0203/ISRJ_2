"""Incremental convergence plots and paired multi-seed summaries."""
from __future__ import annotations

import json
import numpy as np

from paper2.io import now, read, sha
from paper2.evaluation_1103_v4 import summarize
from tools.durable_io import atomic_json, atomic_write
from .models import VARIANTS
from .protocol import BASE, EVIDENCE, ORIGINAL, CONFIG, SEEDS, MILESTONES


def evaluation_files(variant, seed, scope):
    folder = BASE / variant / f"seed_{seed}"
    return folder / f"{scope.lower()}_epoch_256.json"


def load_values(path, metric, variant):
    saved = read(path)
    raw = path.with_suffix(".npz")
    if sha(raw) != saved["local_parent_values_sha256"]:
        raise RuntimeError("parent arrays changed")
    with np.load(raw, allow_pickle=False) as z:
        key = f"model__{metric}" if "stage" in saved and variant == "full" else metric
        return z["parent_id"].copy(), z[key].copy()


def comparison(scope):
    rows = []
    for variant in VARIANTS:
        paths = [evaluation_files(variant, s, scope) for s in SEEDS]
        originals = [evaluation_files("full", s, scope) for s in SEEDS]
        if not all(p.exists() for p in [*paths, *originals]):
            continue
        first = read(paths[0])
        metrics = {}
        for metric in first["metrics"]:
            per_seed, orig_seed, differences = [], [], []
            parent_ids = None
            for path, original in zip(paths, originals):
                ids, values = load_values(path, metric, variant)
                old_ids, old_values = load_values(original, metric, "full")
                if not np.array_equal(ids, old_ids) or (parent_ids is not None and not np.array_equal(ids, parent_ids)):
                    raise RuntimeError("paired physical-parent identity mismatch")
                parent_ids = ids
                per_seed.append(float(values.mean()))
                orig_seed.append(float(old_values.mean()))
                differences.append(values - old_values)
            delta = np.asarray(per_seed) - orig_seed
            metrics[metric] = {"variant_seed_means": per_seed, "full_seed_means": orig_seed,
                               "mean": float(np.mean(per_seed)), "seed_sample_sd": float(np.std(per_seed, ddof=1)),
                               "paired_seed_differences": delta.tolist(), "paired_mean_difference": float(delta.mean()),
                               "paired_difference_seed_sd": float(delta.std(ddof=1)),
                               "seed_averaged_parent_difference": summarize(np.mean(differences, axis=0))}
        rows.append({"variant": variant, "scope": scope, "seeds": list(SEEDS), "epoch": 256,
                     "metrics": metrics, "note": "Parent CI is conditional on these three trained seeds; seed SD is reported separately. No pooled seed-times-parent significance test."})
    return rows


def refresh():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    records = []
    curves = []
    for variant in VARIANTS:
        for seed in SEEDS:
            folder = BASE / variant / f"seed_{seed}"
            records.extend(read(p) for p in sorted(folder.glob("training_epoch_*.json")))
            for epoch in MILESTONES:
                path = folder / f"dev_epoch_{epoch:03d}.json"
                if path.exists():
                    r = read(path)
                    curves.append({"variant": variant, "seed": seed, "epoch": epoch,
                                   "metrics": {k: v["mean"] for k, v in r["metrics"].items()},
                                   "checkpoint_sha256": r["checkpoint_sha256"]})
    result = {"updated_utc": now(), "config_sha256": sha(CONFIG), "training_epochs": records,
              "dev_curves": curves, "dev_epoch256_comparison": comparison("DEV"),
              "confirmation_epoch256_comparison": comparison("CONFIRMATION")}
    atomic_json(BASE / "SUMMARY.json", result)
    atomic_json(EVIDENCE / "SUMMARY.json", result)
    # Keep plots separate from scientific computation; use only saved metrics.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {v: plt.get_cmap("tab10")(i) for i, v in enumerate(("full", *VARIANTS))}
    for ax, key in zip(axes.flat, ("u_nmse", "damage", "sjrif_db", "pd")):
        for variant in ("full", *VARIANTS):
            for seed in SEEDS:
                if variant == "full":
                    points = []
                    for epoch in MILESTONES:
                        path = ORIGINAL / f"seed_{seed}" / f"dev_epoch_{epoch:03d}_v4.json"
                        if path.exists():
                            points.append((epoch, read(path)["metrics"]["model"][key]["mean"]))
                else:
                    points = [(r["epoch"], r["metrics"][key]) for r in curves if r["variant"] == variant and r["seed"] == seed]
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, color=colors[variant], alpha=.7, label=variant if seed == SEEDS[0] else None,
                            linestyle="--" if variant == "full" else "-")
        ax.set(xlabel="Epoch", ylabel=key, title="DEV only")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        atomic_write(BASE / f"dev_convergence.{suffix}", lambda stream: fig.savefig(stream, format=suffix, dpi=160))
    plt.close(fig)
    if records:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, key in zip(axes.flat, ("total", "endpoint", "pc", "damage")):
            for variant in VARIANTS:
                for seed in SEEDS:
                    points = [(r["epoch"], r[key]) for r in records if r["variant"] == variant and r["seed"] == seed]
                    if points:
                        x, y = zip(*points)
                        ax.plot(x, y, color=colors[variant], alpha=.7, label=variant if seed == SEEDS[0] else None)
            ax.set(xlabel="Epoch", ylabel=key, title="Training diagnostics (different total objectives)")
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        for suffix in ("png", "pdf"):
            atomic_write(BASE / f"training_convergence.{suffix}", lambda stream: fig.savefig(stream, format=suffix, dpi=160))
        plt.close(fig)
    return result


if __name__ == "__main__":
    refresh()
