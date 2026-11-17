"""Exploratory DEV-only residual and matched-filter tail diagnostics.

This reads frozen epoch-256 checkpoints and the existing DEV role. It never
trains, writes checkpoints, or exports waveform/parent arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, restore_model, states_from_components
from paper2.io import ROOT, now, read, sha
from paper2.radar_eval import ca_cfar, doppler_process, pulse_compress
from tools.durable_io import atomic_json
from tools.unet_ablation.models import VARIANTS, make_model
from tools.unet_ablation.protocol import CONFIG, DATA, ORIGINAL, BASE
from tools.unet_ablation.runtime import configure_precision, require_server

RUN = "residual_failure_diagnostic_20261117_v2"
OUT = ROOT / "evidence" / "outputs" / RUN
SEEDS = (17, 29, 41)
MODELS = ("full", *VARIANTS)
EPOCH = 256


def checkpoint_path(model: str, seed: int) -> Path:
    folder = ORIGINAL if model == "full" else BASE / model
    return folder / f"seed_{seed}" / f"epoch_{EPOCH:03d}.pt"


def describe(x: np.ndarray, seed: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(x), size=(4000, len(x)))
    ci = np.quantile(x[draws].mean(axis=1), [0.025, 0.975])
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, 0.95)), "p99": float(np.quantile(x, 0.99)),
            "parent_bootstrap_ci95": ci.tolist(), "parents": int(len(x))}


def quantile_groups(values: np.ndarray) -> np.ndarray:
    """Four equal-sized, deterministic DEV-only parent quantile groups."""
    order = np.argsort(values, kind="stable")
    groups = np.empty(len(values), dtype=np.int8)
    groups[order] = np.minimum(np.arange(len(values)) * 4 // len(values), 3)
    return groups


def failure_associations(values: dict[str, np.ndarray], seed: int) -> dict:
    """Parent-level descriptive rank associations; bootstrap parents, not pulses."""
    predictors = ("background_residual_db", "mf_peak_to_noise_median_db",
                  "rd_max_to_cfar_db")
    outcomes = {"frame_pfa_jammer": values["frame_pfa_jammer"],
                "pd_failure": 1.0 - values["pd"]}
    rng = np.random.default_rng(2026111700 + seed)
    result = {}
    for predictor in predictors:
        x = rankdata(values[predictor], method="average").astype(np.float64)
        result[predictor] = {}
        for outcome_name, y in outcomes.items():
            y = np.asarray(y, dtype=np.float64)
            yrank = rankdata(y, method="average")
            xc, yc = x - x.mean(), yrank - yrank.mean()
            denom0 = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
            r = float(np.sum(xc * yc) / denom0) if denom0 > 0 else 0.0
            draws = rng.integers(len(x), size=(2000, len(x)))
            sampled_x, sampled_y = x[draws], yrank[draws]
            dx = sampled_x - sampled_x.mean(axis=1, keepdims=True)
            dy = sampled_y - sampled_y.mean(axis=1, keepdims=True)
            denom = np.sqrt(np.sum(dx * dx, axis=1) * np.sum(dy * dy, axis=1))
            boot = np.divide(np.sum(dx * dy, axis=1), denom,
                             out=np.zeros_like(denom), where=denom > 0)
            q1, q3 = np.quantile(values[predictor], [0.25, 0.75])
            top = values[predictor] >= q3
            bottom = values[predictor] <= q1
            result[predictor][outcome_name] = {
                "spearman_rho": r,
                "parent_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "top_quartile_event_rate": float(np.mean(y[top])),
                "bottom_quartile_event_rate": float(np.mean(y[bottom])),
                "top_minus_bottom_rate": float(np.mean(y[top]) - np.mean(y[bottom])),
                "parents": int(len(x)),
                "interpretation": "descriptive DEV association; not causal or confirmatory"}
    return result


def load_bound_model(model_name: str, seed: int, cfg: dict, config_sha: str):
    cp = checkpoint_path(model_name, seed)
    receipt = read(cp.with_suffix(".json"))
    digest = sha(cp)
    if receipt.get("checkpoint_sha256") != digest or receipt.get("epoch") != EPOCH:
        raise RuntimeError(f"checkpoint receipt mismatch: {cp}")
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    if payload.get("epoch") != EPOCH or payload.get("seed") != seed:
        raise RuntimeError(f"checkpoint identity mismatch: {cp}")
    if model_name == "full":
        expected = cfg["original_epoch256"][str(seed)]["checkpoint_sha256"]
        # The original training checkpoint stores its network under ``model``;
        # retain ``state`` as a compatibility fallback for earlier receipts.
        state = payload.get("model", payload.get("state"))
    else:
        identity = payload.get("identity", {})
        if identity.get("variant") != model_name or identity.get("seed") != seed:
            raise RuntimeError(f"variant identity mismatch: {cp}")
        if identity.get("config_sha256") != config_sha:
            raise RuntimeError(f"configuration identity mismatch: {cp}")
        expected = None
        state = payload.get("model")
    if expected is not None and digest != expected:
        raise RuntimeError(f"frozen original checkpoint changed: {cp}")
    net = make_model(model_name).to("cuda").eval()
    net.load_state_dict(state, strict=True)
    return net, digest


def residual_metrics(s: np.ndarray, j: np.ndarray, e: np.ndarray,
                     restored: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-parent/per-jammer-family H0 residual and matched-filter tails."""
    # Input components: [parent, family, pulse, fast_time].
    n, families, pulses, nt = j.shape
    u = restored.reshape(n, families, 4, pulses, nt)
    noise_error = u[:, :, 0] - e
    jammer_error = u[:, :, 2] - e
    noise_pc, _ = pulse_compress(noise_error)
    jammer_pc, _ = pulse_compress(jammer_error)
    result = {k: np.zeros((n, families), dtype=np.float64) for k in (
        "background_residual_db", "mf_peak_to_noise_median_db", "rd_max_to_cfar_db",
        "rd_max_cfar_ratio", "frame_pfa")}
    floor = np.finfo(np.float64).tiny
    for parent in range(n):
        for family in range(families):
            n_pc = noise_pc[parent, family]
            j_pc = jammer_pc[parent, family]
            n_rd, _ = doppler_process(n_pc)
            j_rd, _ = doppler_process(j_pc)
            detections, threshold, valid, _, _ = ca_cfar(np.abs(j_rd) ** 2)
            power = np.abs(j_rd) ** 2
            # Work in dB before taking the maximum. Near-zero residual noise can
            # make the linear ratio overflow even though the diagnostic is finite
            # after applying the documented numerical floor.
            ratios_db = 10 * (np.log10(np.maximum(power[valid], floor))
                              - np.log10(np.maximum(threshold[valid], floor)))
            max_ratio_db = float(np.max(ratios_db))
            max_ratio = float(10 ** (np.clip(max_ratio_db, -300, 300) / 10))
            noise_median = float(np.median(np.abs(n_rd) ** 2))
            result["background_residual_db"][parent, family] = 10 * np.log10(
                max(float(np.sum(np.abs(jammer_error[parent, family]) ** 2)), floor) /
                max(float(np.sum(np.abs(j[parent, family]) ** 2)), floor))
            result["mf_peak_to_noise_median_db"][parent, family] = 10 * (
                np.log10(max(float(np.max(np.abs(j_pc) ** 2)), floor)) -
                np.log10(max(noise_median, floor)))
            result["rd_max_to_cfar_db"][parent, family] = max_ratio_db
            result["rd_max_cfar_ratio"][parent, family] = max_ratio
            result["frame_pfa"][parent, family] = float(np.any(detections & valid))
    return result


def evaluate_one(model_name: str, seed: int, data_sha: str, config_sha: str,
                 diagnostic_sha: str,
                 parent_id: np.ndarray, s: np.ndarray, j: np.ndarray, e: np.ndarray,
                 doppler: np.ndarray, snr: np.ndarray, jsr: np.ndarray) -> dict:
    started = time.perf_counter()
    cp = checkpoint_path(model_name, seed)
    net, cp_sha = load_bound_model(model_name, seed, read(CONFIG), config_sha)
    row_path = OUT / f"{model_name}_seed_{seed}.json"
    binding = {"model": model_name, "seed": seed, "epoch": EPOCH,
               "checkpoint_sha256": cp_sha, "data_sha256": data_sha,
               "config_sha256": config_sha, "diagnostic_source_sha256": diagnostic_sha}
    if row_path.exists():
        saved = read(row_path)
        if any(saved.get(k) != v for k, v in binding.items()):
            raise RuntimeError(f"existing diagnostic identity mismatch: {row_path}")
        del net
        return saved

    n = len(parent_id)
    arrays: dict[str, list[np.ndarray]] = {}
    residual: dict[str, list[np.ndarray]] = {}
    for lo in range(0, n, 8):
        hi = min(lo + 8, n)
        sb, jb, eb = (v[lo:hi].reshape(-1, 8, 1800).astype(np.complex128)
                      for v in (s, j, e))
        observed = states_from_components(sb, jb, eb)
        restored = restore_model(net, observed, "cuda")
        metrics = group_metrics(sb, jb, eb, restored)
        metrics.update(radar_metrics(sb, observed, restored, np.repeat(doppler[lo:hi], 3)))
        tail = residual_metrics(s[lo:hi], j[lo:hi], e[lo:hi], restored)
        for key, value in metrics.items():
            arrays.setdefault(key, []).append(np.asarray(value).reshape(hi - lo, 3).mean(axis=1))
        for key, value in tail.items():
            residual.setdefault(key, []).append(value.mean(axis=1))
        atomic_json(OUT / "STATUS.json", {**binding, "stage": "EVALUATING",
                    "parents_done": hi, "parents_total": n, "updated_utc": now()})
    del net
    torch.cuda.empty_cache()

    values = {k: np.concatenate(v) for k, v in arrays.items()}
    values.update({k: np.concatenate(v) for k, v in residual.items()})
    invalid = {key: int((~np.isfinite(value)).sum()) for key, value in values.items()
               if not np.isfinite(value).all()}
    if invalid:
        raise RuntimeError(f"nonfinite per-parent diagnostic metrics: {invalid}")
    snr_parent = snr.mean(axis=1)
    # NaN encodes a jammer family/pulse without an assigned JSR level in this
    # DEV design; summarize only assigned JSR values within each parent.
    jsr_parent = np.nanmean(jsr, axis=(1, 2))
    if not np.isfinite(snr_parent).all() or not np.isfinite(jsr_parent).all():
        raise RuntimeError("nonfinite DEV SNR/JSR values")
    groups = {"all": np.zeros(n, dtype=np.int8),
              "snr_q": quantile_groups(snr_parent),
              "jsr_q": quantile_groups(jsr_parent)}
    metrics_out = {}
    for group_name, group_ids in groups.items():
        labels = [0] if group_name == "all" else range(4)
        metrics_out[group_name] = {}
        for label in labels:
            take = np.ones(n, dtype=bool) if group_name == "all" else group_ids == label
            metrics_out[group_name][str(label)] = {
                key: describe(value[take], 20261117 + seed + label)
                for key, value in values.items()}
    result = {**binding, "parents": n,
              "parent_id_sha256": hashlib.sha256(parent_id.tobytes()).hexdigest(),
              "snr_quartile_edges_db": np.quantile(snr_parent, [0, .25, .5, .75, 1]).tolist(),
              "jsr_quartile_edges_db": np.quantile(jsr_parent, [0, .25, .5, .75, 1]).tolist(),
              "metrics": metrics_out,
              "failure_associations": failure_associations(values, seed),
              "created_utc": now(), "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["DEV-only exploratory diagnostics; DEV participated in prior development.",
                "Quantile groups are descriptive, not frozen Main/Extreme roles.",
                "Parent is the resampling unit; jammer families are averaged within parent.",
                "CFAR peak ratio and matched-filter peak are diagnostics, not a calibrated false-alarm guarantee."]}
    atomic_json(row_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    require_server()
    configure_precision()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read(CONFIG)
    config_sha = sha(CONFIG)
    source_drift = {name: (sha(ROOT / name), digest)
                    for name, digest in cfg["source_sha256"].items()
                    if sha(ROOT / name) != digest}
    if source_drift:
        raise RuntimeError(f"frozen source identity mismatch: {source_drift}")
    data_path = DATA / "dev.npz"
    data_sha = sha(data_path)
    diagnostic_sha = sha(Path(__file__))
    if read(DATA / "dev.receipt.json").get("sha256") != data_sha:
        raise RuntimeError("DEV data receipt mismatch")
    with np.load(data_path, allow_pickle=False) as data:
        s, j, e = (data[k].copy() for k in ("s", "j", "epsilon"))
        parent_id, doppler = data["parent_id"].copy(), data["doppler_hz"].copy()
        snr, jsr = data["snr_db"].copy(), data["jsr_db"].copy()
    identity = {"run": RUN, "data_sha256": data_sha, "config_sha256": config_sha,
                "diagnostic_source_sha256": diagnostic_sha,
                "models": list(args.models), "seeds": list(args.seeds), "epoch": EPOCH,
                "scope": "DEV_ONLY", "started_utc": now()}
    atomic_json(OUT / "RUN_IDENTITY.json", identity)
    results = []
    for model_name in args.models:
        for seed in args.seeds:
            result = evaluate_one(model_name, seed, data_sha, config_sha, diagnostic_sha, parent_id,
                                  s, j, e, doppler, snr, jsr)
            results.append({"model": model_name, "seed": seed,
                            "checkpoint_sha256": result["checkpoint_sha256"],
                            "result_file": f"{model_name}_seed_{seed}.json"})
            atomic_json(OUT / "STATUS.json", {**identity, "stage": "RUNNING",
                        "completed_models": results, "updated_utc": now()})
    atomic_json(OUT / "STATUS.json", {**identity, "stage": "COMPLETE",
                "completed_models": results, "completed_utc": now()})
    print(json.dumps({"stage": "COMPLETE", "runs": len(results), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
