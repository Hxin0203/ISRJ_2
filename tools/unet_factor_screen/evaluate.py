"""Paired DEV-only evaluation of the representation screen against frozen U-Nets."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from paper2.evaluation_1103_v4 import (group_metrics, radar_metrics,
                                      restore_model, states_from_components,
                                      summarize)
from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json, atomic_write
from tools.unet_ablation.models import PlainUNet
from tools.unet_ablation.protocol import BASE as ABLATION_BASE, DATA
from .runtime import require_server, configure_precision
from .model import FactorUNet

from .train import BASE, EVIDENCE, RUN, SEEDS


def _load_model(path: Path, *, seed: int, epoch: int, expected_run: str | None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("seed") != seed or payload.get("epoch") != epoch:
        raise RuntimeError(f"checkpoint metadata mismatch: {path}")
    if expected_run is not None and payload.get("identity", {}).get("run") != expected_run:
        raise RuntimeError(f"screen checkpoint identity mismatch: {path}")
    with torch.random.fork_rng(devices=[]):
        model = (FactorUNet() if expected_run is not None else PlainUNet()).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    return model


def _evaluate(model, archive, device):
    components = [archive[k] for k in ("s", "j", "epsilon")]
    parent_ids = archive["parent_id"].copy()
    doppler = archive["doppler_hz"].copy()
    parent_values: dict[str, list[np.ndarray]] = {}
    for lo in range(0, len(parent_ids), 8):
        hi = min(lo + 8, len(parent_ids))
        s, j, e = [v[lo:hi].reshape(-1, 8, 1800).astype(np.complex128)
                   for v in components]
        observed = states_from_components(s, j, e)
        restored = restore_model(model, observed, device)
        metrics = group_metrics(s, j, e, restored)
        metrics.update(radar_metrics(s, observed, restored,
                                     np.repeat(doppler[lo:hi], 3)))
        for name, values in metrics.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"non-finite DEV metric: {name}")
            parent_values.setdefault(name, []).append(values.reshape(hi - lo, 3).mean(1))
    return parent_ids, {k: np.concatenate(v) for k, v in parent_values.items()}


def evaluate_all() -> dict:
    require_server()
    configure_precision()
    torch.set_num_threads(4)
    data_path = DATA / "dev.npz"
    data_sha = sha(data_path)
    from .train import validate_config
    config = validate_config()
    if data_sha != config["dev_data_sha256"]:
        raise RuntimeError("DEV identity mismatch")
    config_sha = sha(BASE / "CONFIG.json")
    source_sha = {name: sha(ROOT / name) for name in config["source_sha256"]}
    if source_sha != config["source_sha256"]:
        raise RuntimeError("frozen source changed before DEV evaluation")
    rows = []
    started = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as archive:
        for seed in SEEDS:
            seed_started = time.perf_counter()
            baseline_cp = ABLATION_BASE / "unet" / f"seed_{seed}" / "epoch_256.pt"
            baseline_complete = ABLATION_BASE / "unet" / f"seed_{seed}" / "TRAINING_COMPLETE.json"
            norm_cp = BASE / "factor" / f"seed_{seed}" / "epoch_256.pt"
            norm_complete = BASE / "factor" / f"seed_{seed}" / "TRAINING_COMPLETE.json"
            baseline_sha, norm_sha = sha(baseline_cp), sha(norm_cp)
            if baseline_sha != config["baseline"][str(seed)]["checkpoint_sha256"] or read(baseline_complete)["milestones"]["256"] != baseline_sha:
                raise RuntimeError(f"frozen baseline checkpoint changed: {baseline_cp}")
            completion = read(norm_complete)
            if completion["identity"]["config_sha256"] != config_sha or completion["identity"]["data_sha256"] != config["training_data_sha256"] or completion["milestones"]["256"] != norm_sha:
                raise RuntimeError(f"norm-screen checkpoint changed: {norm_cp}")
            binding = {"run": RUN, "seed": seed, "epoch": 256,
                       "config_sha256": config_sha, "dev_data_sha256": data_sha,
                       "baseline_checkpoint_sha256": baseline_sha,
                       "norm_checkpoint_sha256": norm_sha,
                       "evaluator_source_sha256": source_sha}
            out = BASE / "factor" / f"seed_{seed}" / "dev_paired_epoch_256.json"
            raw = out.with_suffix(".npz")
            if out.exists():
                saved = read(out)
                if any(saved.get(k) != v for k, v in binding.items()) or sha(raw) != saved["parent_values_sha256"]:
                    raise RuntimeError(f"existing paired evaluation identity mismatch: {out}")
                rows.append(saved)
                continue

            baseline = _load_model(baseline_cp, seed=seed, epoch=256, expected_run=None)
            norm = _load_model(norm_cp, seed=seed, epoch=256, expected_run=RUN)
            device = torch.device("cuda")
            ids_b, baseline_values = _evaluate(baseline, archive, device)
            ids_t, norm_values = _evaluate(norm, archive, device)
            if not np.array_equal(ids_b, ids_t) or set(baseline_values) != set(norm_values):
                raise RuntimeError("paired DEV evaluation roster/metric mismatch")
            del baseline, norm
            torch.cuda.empty_cache()
            differences = {k: norm_values[k] - baseline_values[k] for k in baseline_values}
            raw_arrays = {f"baseline__{k}": v for k, v in baseline_values.items()}
            raw_arrays.update({f"norm__{k}": v for k, v in norm_values.items()})
            raw_arrays.update({f"delta__{k}": v for k, v in differences.items()})
            atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=ids_b, **raw_arrays))
            result = {**binding, "parents": len(ids_b),
                      "baseline": {k: summarize(v) for k, v in baseline_values.items()},
                      "factor": {k: summarize(v) for k, v in norm_values.items()},
                      "factor_minus_baseline": {k: summarize(v) for k, v in differences.items()},
                      "parent_values_sha256": sha(raw),
                      "elapsed_seconds": time.perf_counter() - seed_started,
                      "scope_note": "DEV screening only; 3 jammer families averaged within each physical parent."}
            atomic_json(out, result)
            atomic_json(EVIDENCE / "factor" / f"seed_{seed}" / out.name, result)
            rows.append(result)

    per_seed = []
    for row in rows:
        delta = row["factor_minus_baseline"]
        per_seed.append({"seed": row["seed"],
                         "delta_frame_pfa_jammer": delta["frame_pfa_jammer"]["mean"],
                         "delta_pd": delta["pd"]["mean"],
                         "delta_damage": delta["damage"]["mean"],
                         "delta_sjrif_db": delta["sjrif_db"]["mean"]})
    pfa_changes = [x["delta_frame_pfa_jammer"] for x in per_seed]
    pd_changes = [x["delta_pd"] for x in per_seed]
    damage_changes = [x["delta_damage"] for x in per_seed]
    gate = {
        "rule": ("Mark this factor eligible for later combination only if median seed-level jammer-frame Pfa improves; "
                 "each seed's Pd change is >= -0.02; and mean damage change is <= +0.005. "
                 "This is a screening rule, not a confirmation claim."),
        "passes": bool(np.median(pfa_changes) < 0 and min(pd_changes) >= -0.02
                       and np.mean(damage_changes) <= 0.005),
        "seed_direction_counts": {"pfa_improved": int(sum(v < 0 for v in pfa_changes)),
                                  "pd_drop_within_margin": int(sum(v >= -0.02 for v in pd_changes))},
        "pd_noninferiority_margin_absolute": -0.02,
        "damage_worsening_margin_absolute": 0.005,
    }
    summary = {"run": RUN, "stage": "DEV_SCREEN_ONLY", "config_sha256": config_sha,
               "dev_data_sha256": data_sha, "seeds": per_seed,
               "screening_gate": gate,
               "scope_note": "DEV participated in development; results can screen factors only. "
                             "Independent confirmation remains mandatory.",
               "elapsed_seconds": time.perf_counter() - started, "created_utc": now()}
    atomic_json(BASE / "DEV_SCREEN_SUMMARY.json", summary)
    atomic_json(EVIDENCE / "DEV_SCREEN_SUMMARY.json", summary)
    return summary
