"""Fixed DEV-only evaluation for the ReCoVP-Hybrid seed-17 checkpoint.

This entry point is intentionally path-locked to data/paper2_1103/dev.npz,
outputs/recovp_hybrid_seed17, and server B / physical GPU 0 or 1. It never trains,
loads another role, or calls the correction-model restore_model helper.
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from paper2.evaluation_1103_v4 import (group_metrics, radar_metrics, states_from_components,
                                       summarize)
from paper2.io import ROOT, now, read, sha, write
from tools.server_gpu_admission import require_visible_gpu
from tools.vp_1103_recovp_hybrid import ReCoVPHybrid


RUN = "recovp_hybrid_seed17"
SEED = 17
EPOCH = 256
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
DATA = ROOT / "data" / "paper2_1103" / "dev.npz"
DATA_RECEIPT = ROOT / "data" / "paper2_1103" / "dev.receipt.json"
BASE = ROOT / "outputs" / RUN / f"seed_{SEED}"
CHECKPOINT = BASE / f"epoch_{EPOCH:03d}.pt"
COMPLETE = BASE / "TRAINING_COMPLETE.json"
CONFIG = BASE.parent / "CONFIG.json"
PREFLIGHT = ROOT / "evidence" / "outputs" / RUN / "PREFLIGHT.json"
SCALES = (0.1, 10.0)
JAM_TYPES = 3
COMPLEX_SCALE_SEED = 17_256
COMPLEX_SCALE_RANGE = (0.2, 5.0)
SNR_BIN_DB = 5.0
JSR_BIN_DB = 10.0


def require_server_gpu() -> dict:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError("Hybrid DEV evaluation is restricted to server B / DESKTOP-RFFERM3")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"Hybrid DEV evaluation root must be {SERVER_ROOT}")
    selected = require_visible_gpu()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose the selected physical GPU 0 or 1; CUDA evaluation has no CPU/local fallback")
    return {**selected, "hostname": platform.node(), "root": str(ROOT),
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "utilization_percent_at_start": selected["utilization_percent"]}


def evaluation_sources() -> dict[str, str]:
    paths = (
        ROOT / "tools/vp_1103_recovp_hybrid_evaluate.py",
        ROOT / "tools/server_gpu_admission.py",
        ROOT / "tools/vp_1103_recovp_hybrid.py",
        ROOT / "tools/vp_1103_recovp_hybrid_train.py",
        ROOT / "paper2/evaluation_1103_v4.py",
        ROOT / "paper2/radar_eval.py",
        ROOT / "paper2/io.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def validate_checkpoint(source_sha: dict[str, str]) -> tuple[dict, str]:
    if not CHECKPOINT.is_file() or not CHECKPOINT.with_suffix(".json").is_file():
        raise FileNotFoundError(f"fixed epoch-256 checkpoint and receipt required: {CHECKPOINT}")
    complete = read(COMPLETE)
    receipt = read(CHECKPOINT.with_suffix(".json"))
    checkpoint_sha = sha(CHECKPOINT)
    if (complete.get("stage") != "TRAINING_COMPLETE" or complete.get("seed") != SEED
            or complete.get("epochs") != EPOCH):
        raise RuntimeError("seed-17 training completion record missing or invalid")
    identity = complete.get("identity", {})
    if (receipt.get("checkpoint_sha256") != checkpoint_sha
            or receipt.get("identity") != identity
            or int(receipt.get("epoch", -1)) != EPOCH
            or complete.get("milestones", {}).get(str(EPOCH)) != checkpoint_sha
            or complete.get("recovery_sha256", {}).get(str(EPOCH)) != checkpoint_sha):
        raise RuntimeError("epoch-256 checkpoint receipt/completion binding mismatch")
    for name, digest in identity.get("source_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"checkpoint source binding is stale or missing: {name}")
    for required in ("tools/vp_1103_recovp_hybrid.py", "tools/vp_1103_recovp_hybrid_train.py"):
        if required not in identity.get("source_sha256", {}):
            raise RuntimeError(f"checkpoint source binding missing required source: {required}")
    if identity.get("run") != RUN or identity.get("seed") != SEED:
        raise RuntimeError("checkpoint training identity mismatch")
    return complete, checkpoint_sha


def frozen_data_bindings(complete: dict) -> dict[str, str]:
    """Authenticate frozen run records before accepting any current DEV file."""
    identity = complete["identity"]
    if (sha(CONFIG) != identity.get("config_sha256")
            or sha(PREFLIGHT) != identity.get("preflight_sha256")):
        raise RuntimeError("frozen CONFIG/PREFLIGHT SHA does not match checkpoint identity")
    config, preflight = read(CONFIG), read(PREFLIGHT)
    if (config.get("run_id") != RUN or config.get("seed") != SEED
            or config.get("epochs") != EPOCH
            or config.get("preflight_sha256") != identity.get("preflight_sha256")
            or config.get("source_sha256") != identity.get("source_sha256")):
        raise RuntimeError("frozen CONFIG identity differs from checkpoint identity")
    if (preflight.get("stage") != "PREFLIGHT_PASSED" or preflight.get("status") != "PASS"
            or preflight.get("seed") != SEED
            or preflight.get("scope") != "TRAIN_AND_DEV_ONLY"):
        raise RuntimeError("frozen PREFLIGHT scope, seed or status is invalid")
    expected = config.get("data_sha256", {})
    preflight_data = preflight.get("data_sha256", preflight.get("dataset_sha256", {}))
    for name in ("train.npz", "train.receipt.json", "dev.npz", "dev.receipt.json"):
        if not expected.get(name) or preflight_data.get(name) != expected[name]:
            raise RuntimeError(f"frozen CONFIG/PREFLIGHT data binding mismatch: {name}")
    if expected["train.npz"] != identity.get("train_sha256"):
        raise RuntimeError("frozen TRAIN binding differs from checkpoint identity")
    return expected


def load_dev(complete: dict) -> tuple[dict[str, np.ndarray], str]:
    expected = frozen_data_bindings(complete)
    if not DATA.is_file() or not DATA_RECEIPT.is_file():
        raise FileNotFoundError(f"fixed DEV file and receipt required: {DATA}")
    data_sha = sha(DATA)
    if data_sha != expected["dev.npz"] or sha(DATA_RECEIPT) != expected["dev.receipt.json"]:
        raise RuntimeError("live DEV data/receipt differs from frozen CONFIG/PREFLIGHT binding")
    receipt = read(DATA_RECEIPT)
    if receipt.get("sha256", "").lower() != data_sha.lower():
        raise RuntimeError("DEV receipt SHA-256 mismatch")
    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ("s", "j", "epsilon", "parent_id", "doppler_hz", "snr_db", "jsr_db")}
    s, j, e = (arrays[key] for key in ("s", "j", "epsilon"))
    if s.ndim != 4 or s.shape[1:] != (JAM_TYPES, 8, 1800) or j.shape != s.shape or e.shape != s.shape:
        raise ValueError(f"fixed DEV must have [parent,3,8,1800] components; got {s.shape}")
    if len(arrays["parent_id"]) != len(s) or len(arrays["doppler_hz"]) != len(s):
        raise ValueError("DEV parent metadata length mismatch")
    if arrays["snr_db"].shape != (len(s), 8) or arrays["jsr_db"].shape != (len(s), JAM_TYPES, 8):
        raise ValueError("DEV native SNR/JSR metadata shape mismatch")
    return arrays, data_sha


def _iq(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack((x.real, x.imag), axis=1).astype(np.float32, copy=False))


def _scaled_states(s: np.ndarray, j: np.ndarray, e: np.ndarray,
                   scale: float | complex | np.ndarray) -> np.ndarray:
    """Scale flattened parent/variant groups without losing their pairing."""
    batch = len(s)
    s4, j4, e4 = (x.reshape(batch * JAM_TYPES, 8, 1800).astype(np.complex64, copy=False)
                  for x in (s, j, e))
    observed = states_from_components(s4, j4, e4)
    scale_array = np.asarray(scale)
    if scale_array.ndim == 0:
        scaled_observed = observed * scale_array
    elif scale_array.shape == (batch, JAM_TYPES):
        scaled_observed = observed * scale_array.reshape(batch * JAM_TYPES, 1, 1, 1)
    else:
        raise ValueError("scale must be scalar or [parent,jammer_variant]")
    return scaled_observed


def _factor_batch(model: ReCoVPHybrid, s: np.ndarray, j: np.ndarray, e: np.ndarray,
                  scale: float | complex | np.ndarray = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [parent,variant,state,pulse,time] final and Stage-1 restorations."""
    batch = len(s)
    flat = _scaled_states(s, j, e, scale).reshape(-1, 1800)
    iq = _iq(flat)
    with torch.inference_mode():
        factors = model.factor(iq.to("cuda"))
        final = factors["restored"].cpu().numpy()
        stage1 = factors["stage1_restored"].cpu().numpy()
        alpha = factors["alpha"].reshape(batch, JAM_TYPES, 4, 8).cpu().numpy()
    shape = (batch, JAM_TYPES, 4, 8, 1800)
    return final.reshape(shape), stage1.reshape(shape), alpha


def _metric_batch(s: np.ndarray, j: np.ndarray, e: np.ndarray, final: np.ndarray,
                  stage1: np.ndarray, doppler: np.ndarray) -> dict[str, np.ndarray]:
    batch = len(s)
    s4, j4, e4 = (x.reshape(batch * JAM_TYPES, 8, 1800).astype(np.complex128, copy=False)
                  for x in (s, j, e))
    output: dict[str, np.ndarray] = {}
    fd = np.repeat(doppler, JAM_TYPES)
    for label, restored in (("hybrid", final), ("stage1", stage1)):
        u = restored.reshape(batch * JAM_TYPES, 4, 8, 1800).astype(np.complex128, copy=False)
        metrics = group_metrics(s4, j4, e4, u)
        metrics.update(radar_metrics(s4, states_from_components(s4, j4, e4), u, fd))
        for name, values in metrics.items():
            output[f"{label}__{name}"] = np.asarray(values).reshape(batch, JAM_TYPES).mean(axis=1)
    return output


def _distribution(values: np.ndarray, percentiles=(10, 50, 90)) -> dict:
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "mean": float(values.mean()),
            **{f"p{p}": float(np.quantile(values, p / 100)) for p in percentiles}}


def _alpha_distribution(alpha: np.ndarray) -> dict:
    names = ("Y00", "Y10", "Y01", "Y11")
    return {name: _distribution(alpha[:, :, state, :].reshape(-1))
            for state, name in enumerate(names)}


def _measure_latency(model: ReCoVPHybrid, example: np.ndarray) -> dict:
    """CUDA-event latency for model.factor with one or eight pulse inputs."""
    timings = {}
    for count in (1, 8):
        x = _iq(np.repeat(example[:1], count, axis=0)).to("cuda")
        for _ in range(5):
            with torch.inference_mode():
                model.factor(x)
        torch.cuda.synchronize()
        elapsed_ms = []
        for _ in range(40):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                start.record()
                model.factor(x)
                end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
        timings[f"{count}_pulse"] = _distribution(np.asarray(elapsed_ms), (50, 95))
    return timings


def _stratified_summaries(raw_values: dict[str, np.ndarray], snr_db: np.ndarray,
                          jsr_db: np.ndarray) -> dict:
    """Summarize parent-level metrics by fixed-width bins of native DEV metadata."""
    metrics = {k: v for k, v in raw_values.items()
               if "__" in k and not k.startswith("scale_") and
               "abs_metric_delta" not in k and not k.endswith("equivariance_relative_l2")}
    result = {"snr_bin_width_db": SNR_BIN_DB, "jsr_bin_width_db": JSR_BIN_DB,
              "snr_db": {}, "jsr_db": {}}
    snr_parent = np.nanmean(snr_db, axis=1)
    # JSR is pulse-specific and undefined on unjammed pulses; first average each
    # variant over its jammed pulses, then average the three variants per parent.
    with np.errstate(invalid="ignore"):
        jsr_variant = np.nanmean(jsr_db, axis=2)
        jsr_parent = np.nanmean(jsr_variant, axis=1)
    for axis_name, values, width in (("snr_db", snr_parent, SNR_BIN_DB),
                                     ("jsr_db", jsr_parent, JSR_BIN_DB)):
        finite = np.isfinite(values)
        bins = np.zeros(values.shape, dtype=int)
        bins[finite] = np.floor(values[finite] / width).astype(int)
        for bin_id in sorted(set(bins[finite].tolist())):
            mask = (bins == bin_id) & finite
            lo = bin_id * width
            hi = lo + width
            result[axis_name][f"[{lo:g},{hi:g})"] = {
                "parents": int(mask.sum()),
                "metrics": {key: summarize(value[mask], seed=SEED * 50_000 + len(key) + bin_id)
                            for key, value in metrics.items()},
            }
    return result


def evaluate(batch_parents: int = 2) -> dict:
    if batch_parents < 1:
        raise ValueError("batch_parents must be positive")
    admission = require_server_gpu()
    source_sha = evaluation_sources()
    complete, checkpoint_sha = validate_checkpoint(source_sha)
    arrays, data_sha = load_dev(complete)
    torch.set_num_threads(4)
    model = ReCoVPHybrid().cuda().eval()
    payload = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    if (payload.get("format") != "recovp-hybrid-seed17-epoch-v1"
            or payload.get("identity") != complete.get("identity")
            or payload.get("seed") != SEED or payload.get("epoch") != EPOCH):
        raise RuntimeError("loaded epoch-256 checkpoint envelope mismatch")
    model.load_state_dict(payload["model"], strict=True)

    n = len(arrays["parent_id"])
    if n != 256:
        raise ValueError(f"fixed DEV evaluation requires 256 physical parents; got {n}")
    rng = np.random.default_rng(COMPLEX_SCALE_SEED)
    magnitude = rng.uniform(*COMPLEX_SCALE_RANGE, size=(n, JAM_TYPES))
    phase = rng.uniform(-np.pi, np.pi, size=(n, JAM_TYPES))
    complex_scales = (magnitude * np.exp(1j * phase)).astype(np.complex64)
    parent_values: dict[str, list[np.ndarray]] = {}
    equivariance: dict[str, list[np.ndarray]] = {str(value): [] for value in SCALES}
    complex_equivariance: list[np.ndarray] = []
    complex_metric_deltas: dict[str, list[np.ndarray]] = {}
    scale_metrics: dict[str, dict[str, list[np.ndarray]]] = {
        str(value): {} for value in SCALES}
    alpha_rows = []
    started = time.perf_counter()
    for lo in range(0, n, batch_parents):
        hi = min(lo + batch_parents, n)
        s, j, e = (arrays[key][lo:hi] for key in ("s", "j", "epsilon"))
        base_final, base_stage1, alpha = _factor_batch(model, s, j, e)
        base = _metric_batch(s, j, e, base_final, base_stage1, arrays["doppler_hz"][lo:hi])
        for key, values in base.items():
            parent_values.setdefault(key, []).append(values)
        alpha_rows.append(alpha)  # parent × jammer variant × Y state × pulse
        observed = states_from_components(
            s.reshape(-1, 8, 1800), j.reshape(-1, 8, 1800), e.reshape(-1, 8, 1800)
        ).reshape(hi - lo, JAM_TYPES, 4, 8, 1800)
        for label, restored in (("hybrid", base_final), ("stage1", base_stage1)):
            for state_index, state_name in ((0, "Y00"), (1, "Y10")):
                numerator = np.linalg.norm(
                    (restored[:, :, state_index] - observed[:, :, state_index]).reshape(hi - lo, -1), axis=1)
                denominator = np.linalg.norm(observed[:, :, state_index].reshape(hi - lo, -1), axis=1).clip(1e-24)
                parent_values.setdefault(f"{label}__e{'0' if state_index == 0 else 'S'}_{state_name}", []).append(
                    numerator / denominator)
        base_norm = np.linalg.norm(base_final.reshape(hi - lo, -1), axis=1).clip(1e-24)
        for factor in SCALES:
            scaled_final, scaled_stage1, _ = _factor_batch(model, s, j, e, factor)
            expected = base_final * factor
            error = np.linalg.norm((scaled_final - expected).reshape(hi - lo, -1), axis=1) / base_norm
            equivariance[str(factor)].append(error)
            scaled = _metric_batch(s * factor, j * factor, e * factor,
                                   scaled_final, scaled_stage1, arrays["doppler_hz"][lo:hi])
            for key, values in scaled.items():
                scale_metrics[str(factor)].setdefault(key, []).append(values)

        # One random complex scalar per parent × jammer variant, shared across
        # all four states and eight pulses to preserve within-group relations.
        complex_scale = complex_scales[lo:hi]
        random_final, random_stage1, _ = _factor_batch(model, s, j, e, complex_scale)
        expected = base_final * complex_scale[:, :, None, None, None]
        complex_error = np.linalg.norm((random_final - expected).reshape(hi - lo, -1), axis=1) / base_norm
        complex_equivariance.append(complex_error)
        random_metrics = _metric_batch(s * complex_scale[:, :, None, None],
                                      j * complex_scale[:, :, None, None],
                                      e * complex_scale[:, :, None, None],
                                      random_final, random_stage1, arrays["doppler_hz"][lo:hi])
        for key, values in random_metrics.items():
            if key.startswith("hybrid__"):
                delta_key = f"random_complex_scale__abs_metric_delta__{key[8:]}"
                complex_metric_deltas.setdefault(delta_key, []).append(np.abs(values - base[key]))

    parent_id = arrays["parent_id"].copy()
    raw_values = {key: np.concatenate(parts) for key, parts in parent_values.items()}
    raw_alpha = np.concatenate(alpha_rows, axis=0)
    for key, parts in complex_metric_deltas.items():
        raw_values[key] = np.concatenate(parts)
    raw_values["random_complex_scale__equivariance_relative_l2"] = np.concatenate(complex_equivariance)
    for factor in SCALES:
        factor_key = str(factor)
        for key, parts in scale_metrics[factor_key].items():
            scaled = np.concatenate(parts)
            raw_values[f"scale_{factor_key}__{key}"] = scaled
            if key.startswith("hybrid__"):
                raw_values[f"scale_{factor_key}__abs_metric_delta__{key[8:]}"] = np.abs(
                    scaled - raw_values[key])
        raw_values[f"scale_{factor_key}__equivariance_relative_l2"] = np.concatenate(
            equivariance[factor_key])

    summaries = {key: summarize(value, seed=SEED * 10_000 + index)
                 for index, (key, value) in enumerate(raw_values.items())}
    paired = {}
    for key, hybrid_values in raw_values.items():
        if key.startswith("hybrid__"):
            stage1_key = "stage1__" + key[len("hybrid__"):]
            if stage1_key in raw_values:
                paired[f"hybrid_minus_stage1__{key[len('hybrid__'):]}"] = summarize(
                    hybrid_values - raw_values[stage1_key],
                    seed=SEED * 20_000 + len(paired))
    for factor in SCALES:
        factor_key = str(factor)
        for key, values in raw_values.items():
            prefix = f"scale_{factor_key}__abs_metric_delta__hybrid__"
            if key.startswith(prefix):
                paired[key] = summarize(values, seed=SEED * 30_000 + len(paired))
        equiv_key = f"scale_{factor_key}__equivariance_relative_l2"
        paired[equiv_key] = summarize(raw_values[equiv_key], seed=SEED * 40_000 + len(paired))
    random_eq_key = "random_complex_scale__equivariance_relative_l2"
    paired[random_eq_key] = summarize(raw_values[random_eq_key], seed=SEED * 45_000)
    for key, values in raw_values.items():
        if key.startswith("random_complex_scale__abs_metric_delta__"):
            paired[key] = summarize(values, seed=SEED * 46_000 + len(paired))
    raw_path = BASE / "dev_epoch_256_parent_values.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, parent_id=parent_id, alpha_parent_variant_state_pulse=raw_alpha,
                        **raw_values)
    strata = _stratified_summaries(raw_values, arrays["snr_db"], arrays["jsr_db"])
    latency = _measure_latency(model, states_from_components(
        arrays["s"][:1].reshape(-1, 8, 1800), arrays["j"][:1].reshape(-1, 8, 1800),
        arrays["epsilon"][:1].reshape(-1, 8, 1800)).reshape(-1, 1800))
    result = {
        "stage": "DEV_ONLY_RECOVP_HYBRID", "run": RUN, "seed": SEED, "epoch": EPOCH,
        "parents": n, "jammer_variants": JAM_TYPES, "created_utc": now(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": f"cuda:0 (physical GPU {admission['physical_gpu_index']})",
        "execution_admission": admission, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_identity": complete["identity"], "data_sha256": data_sha,
        "data_receipt_sha256": sha(DATA_RECEIPT), "evaluator_source_sha256": source_sha,
        "metrics": summaries, "paired_parent_differences": paired,
        "alpha_distribution_by_state": _alpha_distribution(raw_alpha),
        "modification_amount_definition": {
            "e0": "||U00-Y00||_2 / ||Y00||_2; Y00 is observed epsilon-only state and U00 is its hybrid or Stage-1 restoration.",
            "eS": "||U10-Y10||_2 / ||Y10||_2; Y10 is observed target-plus-noise state and U10 is its hybrid or Stage-1 restoration.",
            "aggregation": "pulse and jammer-variant axes pooled per physical parent; norm computed over both axes",
        },
        "complex_scale_screen": {"magnitude_range": list(COMPLEX_SCALE_RANGE),
                                 "phase_range_rad": [-float(np.pi), float(np.pi)],
                                 "seed": COMPLEX_SCALE_SEED,
                                 "same_factor_for_all_states_and_pulses_within_parent_variant": True,
                                 "paired_zero_metric_changes": {key: paired[key] for key in paired
                                     if key.startswith("random_complex_scale__abs_metric_delta__")},
                                 "equivariance_relative_l2": paired[random_eq_key]},
        "latency_cuda_event_ms": latency,
        "native_snr_jsr_stratification": strata,
        "scale_factors": list(SCALES),
        "raw_parent_values_path": str(raw_path), "raw_parent_values_sha256": sha(raw_path),
        "metric_note": "v4 group_metrics and radar_metrics; three jammer variants averaged within physical parent; paired parent bootstrap intervals; DEV only.",
    }
    out = BASE / "dev_epoch_256_v1.json"
    write(out, result)
    evidence = ROOT / "evidence" / "outputs" / RUN / "seed_17" / out.name
    write(evidence, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-parents", type=int, default=2)
    args = parser.parse_args()
    print(evaluate(batch_parents=args.batch_parents), flush=True)


if __name__ == "__main__":
    main()
