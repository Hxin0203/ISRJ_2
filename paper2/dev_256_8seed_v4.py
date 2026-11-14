"""DEV-only checkpoint evaluation for the 256-epoch, eight-seed run."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data_1103 import JAM_TYPES
from .evaluation_1103_v4 import (group_metrics, radar_metrics, restore_model,
                                 states_from_components, summarize)
from .io import ROOT, now, read, sha, write
from .network import CorrectionNet
from tools.durable_io import atomic_json as write, atomic_write

RUN_ID = "paper2_256_epoch_8seed_20261114"
BASE = ROOT / "outputs" / RUN_ID
EVIDENCE = ROOT / "evidence" / "outputs" / RUN_ID
SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
EPOCHS = tuple(range(32, 257, 32))


def evaluator_sources() -> dict[str, str]:
    names = ("dev_256_8seed_v4.py", "evaluation_1103_v4.py", "radar_eval.py",
             "network.py", "data_1103.py", "io.py")
    values = {f"paper2/{name}": sha(ROOT / "paper2" / name) for name in names}
    values["tools/durable_io.py"] = sha(ROOT / "tools/durable_io.py")
    return values


def _save_npz_atomic(path: Path, **arrays) -> None:
    atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def evaluate_checkpoint(seed: int, epoch: int) -> dict:
    if seed not in SEEDS or epoch not in EPOCHS:
        raise ValueError(f"unregistered checkpoint: seed={seed}, epoch={epoch}")
    started = time.perf_counter()
    folder = BASE / f"seed_{seed}"
    checkpoint = folder / f"epoch_{epoch:03d}.pt"
    receipt_path = checkpoint.with_suffix(".json")
    if not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"checkpoint and receipt required: {checkpoint}")
    checkpoint_sha = sha(checkpoint)
    receipt = read(receipt_path)
    if receipt.get("checkpoint_sha256") != checkpoint_sha or int(receipt.get("epoch", -1)) != epoch:
        raise RuntimeError(f"checkpoint receipt mismatch: {checkpoint}")

    data_path = ROOT / "data" / "paper2_1103" / "dev.npz"
    data_sha = sha(data_path)
    source_sha = evaluator_sources()
    binding = {"checkpoint_sha256": checkpoint_sha, "data_sha256": data_sha,
               "evaluator_source_sha256": source_sha}
    out = folder / f"dev_epoch_{epoch:03d}_v4.json"
    raw = folder / f"dev_epoch_{epoch:03d}_v4.npz"
    if out.exists():
        saved = read(out)
        if any(saved.get(key) != value for key, value in binding.items()):
            raise RuntimeError(f"existing DEV identity changed; preserving result: {out}")
        if not raw.is_file() or sha(raw) != saved.get("local_parent_values_sha256"):
            raise RuntimeError(f"existing per-parent DEV evidence is missing or damaged: {raw}")
        return saved

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorrectionNet().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("seed") != seed or payload.get("epoch") != epoch:
        raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint}")
    # Epochs 32/64/96 are immutable byte copies from the original v4 run;
    # epoch 128 and later use the crash-resilient checkpoint envelope.
    state = payload.get("state", payload.get("model"))
    if state is None:
        raise RuntimeError(f"unrecognized checkpoint format: {checkpoint}")
    model.load_state_dict(state)

    with np.load(data_path, allow_pickle=False) as archive:
        components = [archive[key] for key in ("s", "j", "epsilon")]
        parent_ids = archive["parent_id"].copy()
        doppler = archive["doppler_hz"].copy()
    count = len(parent_ids)
    values = {method: {} for method in ("model", "keep", "reference")}
    for lo in range(0, count, 8):
        hi = min(lo + 8, count)
        s, j, e = [array[lo:hi].reshape(-1, 8, 1800).astype(np.complex128)
                   for array in components]
        observed = states_from_components(s, j, e)
        restored = restore_model(model, observed, device)
        reference = states_from_components(s, np.zeros_like(j), e)
        for method, output in (("model", restored), ("keep", observed),
                               ("reference", reference)):
            metrics = group_metrics(s, j, e, output)
            metrics.update(radar_metrics(s, observed, output,
                                         np.repeat(doppler[lo:hi], len(JAM_TYPES))))
            for name, metric_values in metrics.items():
                values[method].setdefault(name, []).append(
                    metric_values.reshape(hi - lo, len(JAM_TYPES)))
        write(BASE / "DEV_STATUS.json", {
            "stage": "DEV_V4", "seed": seed, "epoch": epoch,
            "parents_done": hi, "parents": count, "updated_utc": now()})

    parent_values = {
        method: {name: np.concatenate(chunks).mean(axis=1) for name, chunks in metrics.items()}
        for method, metrics in values.items()
    }
    _save_npz_atomic(raw, parent_id=parent_ids, **{
        f"{method}__{name}": values_by_parent
        for method, metrics in parent_values.items()
        for name, values_by_parent in metrics.items()
    })
    summaries = {method: {name: summarize(values_by_parent)
                          for name, values_by_parent in metrics.items()}
                 for method, metrics in parent_values.items()}
    paired = {name: summarize(parent_values["model"][name] - parent_values["keep"][name])
              for name in parent_values["model"]}
    result = {
        "stage": "DEV_ONLY_V4", "seed": seed, "epoch": epoch,
        "parents": count, "created_utc": now(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device), **binding, "metrics": summaries,
        "model_minus_keep": paired,
        "local_parent_values_sha256": sha(raw),
        "metric_note": (
            "Same DEV v4 group-energy, pulse-compression, Doppler and CFAR protocol. "
            "Three jammer variants are averaged within physical parent; uncertainty "
            "resamples physical parents, not pulses or jammer variants. DEV only."
        ),
    }
    write(out, result)
    write(EVIDENCE / f"seed_{seed}" / out.name, result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--epochs", nargs="+", type=int, choices=EPOCHS, default=list(EPOCHS))
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for epoch in args.epochs:
            result = evaluate_checkpoint(seed, epoch)
            rows.append({"seed": seed, "epoch": epoch,
                         "checkpoint_sha256": result["checkpoint_sha256"],
                         "metrics": result["metrics"]["model"]})
            write(EVIDENCE / "DEV_SUMMARY_v4.json", {
                "created_utc": now(), "rows": rows,
                "status": "partial", "scope": "DEV_ONLY"})
            print(f"completed DEV v4 seed={seed} epoch={epoch}", flush=True)
    write(EVIDENCE / "DEV_SUMMARY_v4.json", {
        "created_utc": now(), "rows": rows,
        "status": "complete" if len(rows) == len(SEEDS) * len(EPOCHS) else "partial",
        "scope": "DEV_ONLY"})


if __name__ == "__main__":
    main()
