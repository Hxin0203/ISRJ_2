"""Registered inference-time screening of a coherence-graph VP penalty.

This script evaluates a frozen seed-17, epoch-256 ReCoVP-Hybrid checkpoint.
It does not train or modify the model source. The graph penalty is added only
to the Stage-1 coefficient solve, followed by the frozen Stage-2 refiner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "data/hybrid_condition_master_20261127"
CALIBRATION = ROOT / "data/unet_ablation_3seed_256_20261115/equal_pfa/calibration/test.npz"
CHECKPOINT = ROOT / "outputs/recovp_hybrid_seed17/seed_17/epoch_256.pt"
CHECKPOINT_SIDECAR = CHECKPOINT.with_suffix(".json")
BASELINE_ROW = ROOT / "outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_17/ROW.json"
CONFIG = ROOT / "evidence/outputs/recovp_hybrid_seed17/CONFIG.json"
PREFLIGHT = ROOT / "evidence/outputs/recovp_hybrid_seed17/PREFLIGHT.json"
PRE_REGISTRATION = ROOT / "evidence/outputs/RECoVP_GRAPH_EXT_20261127/PREREGISTRATION.json"
OUTPUTS = ROOT / "outputs/recovp_graph_extension_20261127"
REPORTS = ROOT / "reports/RECoVP_graph_extension_20261127"

EXPECTED_CHECKPOINT_SHA256 = "e0b6cb1be9559ed5e75a9ce3c4e178e85aba22f0cec753f52d3f523cbaf05212"
EXPECTED_MASTER_SHA256 = "0c7f029b0d89c2fdaa9503dafd45c63e52874885337e139f63543622cafb6235"
EXPECTED_PARENT_COUNT = 128
EXPECTED_CALIBRATION_PARENT_COUNT = 1024
PARENT_BATCH = 4
SIGNAL_BATCH = 96
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261127
PERMUTATION_SEED = 20261128
CONDITIONS = ((-30, 10), (-30, 30), (-20, 10), (-20, 30))
PFA_TARGETS = (0.01, 0.05, 0.10)
VARIANTS = (
    "baseline",
    "graph_weak",
    "graph_strong",
    "permuted_graph_strong",
    "diagonal_strong",
)
BOOTSTRAP_METRICS = (
    "u_nmse",
    "damage",
    "sjrif_db",
    "amplitude",
    "pd_eq_0.01",
    "pd_eq_0.05",
    "pd_eq_0.1",
    "pfa_eq_0.01",
    "frame_pfa_jammer",
)


def protocol_definition() -> dict[str, Any]:
    return {
        "question": "Does a coherence-derived graph Laplacian penalty improve the frozen ReCoVP-Hybrid Stage-1 variable-projection solve?",
        "checkpoint": {"run": "recovp_hybrid_seed17", "seed": 17, "epoch": 256,
                       "sha256": EXPECTED_CHECKPOINT_SHA256},
        "data": {
            "master_component_sha256": EXPECTED_MASTER_SHA256,
            "primary_parent_indices": [0, EXPECTED_PARENT_COUNT],
            "physical_parent_count": EXPECTED_PARENT_COUNT,
            "calibration_parent_count": EXPECTED_CALIBRATION_PARENT_COUNT,
            "calibration_sha256": sha256(CALIBRATION),
            "conditions": [{"snr_db": snr, "jsr_db": jsr} for snr, jsr in CONDITIONS],
            "states": ["E", "S+E", "J+E", "S+J+E"],
            "jammer_types": ["direct", "repeat", "cyclic"],
            "repeated_measurement_aggregation": "average jammer types within each physical parent",
        },
        "extension": {
            "edge_weight": "squared magnitude of normalized dictionary-atom inner product",
            "neighbors_per_node_before_symmetrization": 2,
            "symmetrization": "union by elementwise maximum",
            "penalty": "D^-1 L D^-1, where L is the weighted graph Laplacian and D contains physical atom norms",
            "trace_normalization": "trace(Gamma) / trace(diag(precision)) equals the configured strength",
            "variants": {
                "baseline": "no additional penalty",
                "graph_weak": {"strength": 0.1},
                "graph_strong": {"strength": 1.0},
                "permuted_graph_strong": {"strength": 1.0, "permutation_seed": PERMUTATION_SEED},
                "diagonal_strong": {"strength": 1.0, "control": "retain graph penalty diagonal and remove node coupling"},
            },
            "stage2": "frozen checkpoint refiner is unchanged",
        },
        "thresholds": {
            "source": "independent 1024-parent calibration file",
            "targets": list(PFA_TARGETS),
            "policy": "recalibrate each variant separately using the higher empirical quantile of all parent-by-jammer-type H0 scores",
        },
        "estimands": {
            "primary": "parent-paired difference in macro mean U-NMSE, equal-weighting the four registered SNR x JSR cells",
            "secondary": ["target damage", "SJRI-F", "amplitude retention", "equal-PFA detection and false alarm", "standard frame false alarm", "latency"],
            "regimes": {"extreme": "-30 <= SNR < -20 dB", "ordinary": "-20 <= SNR <= 0 dB"},
            "uncertainty": "10,000 paired bootstrap resamples of physical parent IDs, jointly across conditions and jammer types",
        },
        "screening_gates": {
            "relative_u_nmse_reduction_at_least": 0.05,
            "u_nmse_paired_ci95_upper_below": 0.0,
            "damage_relative_increase_ci95_upper_at_most": 0.05,
            "pd_at_1pct_equal_pfa_ci95_lower_difference_at_least": -0.02,
            "pfa_at_1pct_equal_pfa_ci95_upper_difference_at_most": 0.01,
            "graph_specificity": "for graph_strong, paired macro U-NMSE CI95 upper must be below zero versus both permuted_graph_strong and diagonal_strong",
            "interpretation": "passing advances only to trained multi-seed ablation; it does not establish final adoption or generalization",
        },
        "compute": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_seed": PERMUTATION_SEED,
            "parent_batch": PARENT_BATCH,
            "signal_batch": SIGNAL_BATCH,
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    devices = []
    for row in rows:
        index, name, uuid, utilization, used, total = [item.strip() for item in row.split(",", 5)]
        devices.append({
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "utilization_percent": int(utilization),
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
        })
    if not devices:
        raise RuntimeError("nvidia-smi did not report a CUDA device")
    return {"host": platform.node(), "devices": devices}


def validate_bindings() -> dict[str, Any]:
    from tools.paper2_baseline_suite import equal_pfa_suite as equal_pfa

    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen seed-17 checkpoint SHA-256 changed")
    sidecar = json.loads(CHECKPOINT_SIDECAR.read_text(encoding="utf-8"))
    identity = sidecar.get("identity", {})
    if sidecar.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256 or sidecar.get("epoch") != 256:
        raise RuntimeError("checkpoint sidecar does not bind the frozen epoch-256 checkpoint")
    if identity.get("run") != "recovp_hybrid_seed17" or identity.get("seed") != 17:
        raise RuntimeError("checkpoint sidecar run or seed identity changed")
    for relative_path, expected in identity.get("source_sha256", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"checkpoint source binding changed: {relative_path}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if sha256(CONFIG) != identity.get("config_sha256"):
        raise RuntimeError("seed-17 CONFIG SHA-256 does not match checkpoint identity")
    if sha256(PREFLIGHT) != identity.get("preflight_sha256"):
        raise RuntimeError("seed-17 PREFLIGHT SHA-256 does not match checkpoint identity")
    if config.get("source_sha256") != identity.get("source_sha256"):
        raise RuntimeError("CONFIG source bindings differ from checkpoint sidecar")
    if preflight.get("status") != "PASS" or preflight.get("seed") != 17:
        raise RuntimeError("seed-17 preflight record is missing or invalid")

    baseline_row = json.loads(BASELINE_ROW.read_text(encoding="utf-8"))
    if baseline_row.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("baseline equal-PFA ROW is bound to a different checkpoint")

    registration_path = DATA_ROOT / "REGISTRATION.json"
    manifest_path = DATA_ROOT / "DATA_MANIFEST.json"
    component_path = DATA_ROOT / "components/test_0_0.npz"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(component_path) != EXPECTED_MASTER_SHA256:
        raise RuntimeError("registered condition-master component SHA-256 changed")
    if manifest.get("component_sha256") != EXPECTED_MASTER_SHA256:
        raise RuntimeError("condition-master manifest does not bind its component file")
    if registration.get("primary_parent_indices") != [0, EXPECTED_PARENT_COUNT]:
        raise RuntimeError("registered primary parent range changed")
    for relative_path, expected in registration.get("source_sha256", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"registered condition source changed: {relative_path}")

    if not CALIBRATION.is_file():
        raise FileNotFoundError(f"independent calibration set is missing: {CALIBRATION}")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA inference is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("screening requires the selected local CUDA device to be the only visible device")

    equal_pfa.ROOT = ROOT
    equal_pfa.OUTPUTS = ROOT / "outputs"
    equal_pfa.configure_precision()
    torch.set_num_threads(4)

    with np.load(CALIBRATION, allow_pickle=False) as archive:
        calibration_ids = archive["parent_id"].copy()
        calibration_shapes = {key: list(archive[key].shape) for key in ("s", "j", "epsilon", "doppler_hz")}
    with np.load(component_path, allow_pickle=False) as archive:
        primary_ids = archive["parent_id"][:EXPECTED_PARENT_COUNT].copy()
        primary_shapes = {key: list(archive[key][:EXPECTED_PARENT_COUNT].shape) for key in ("s", "j", "epsilon")}
    if len(np.unique(calibration_ids)) != len(calibration_ids):
        raise RuntimeError("calibration parent IDs are not unique")
    if len(calibration_ids) != EXPECTED_CALIBRATION_PARENT_COUNT:
        raise RuntimeError("independent calibration parent count changed")
    if calibration_shapes != {
        "s": [EXPECTED_CALIBRATION_PARENT_COUNT, 3, 8, 1800],
        "j": [EXPECTED_CALIBRATION_PARENT_COUNT, 3, 8, 1800],
        "epsilon": [EXPECTED_CALIBRATION_PARENT_COUNT, 3, 8, 1800],
        "doppler_hz": [EXPECTED_CALIBRATION_PARENT_COUNT],
    }:
        raise RuntimeError(f"independent calibration array shapes changed: {calibration_shapes}")
    if np.intersect1d(primary_ids, calibration_ids).size:
        raise RuntimeError("registered primary DEV parents overlap the independent calibration set")
    if any(count for count in manifest.get("protected_id_overlap", {}).values()):
        raise RuntimeError("condition-master manifest records overlap with a protected data split")

    return {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_sidecar_sha256": sha256(CHECKPOINT_SIDECAR),
        "checkpoint_identity": identity,
        "config_sha256": sha256(CONFIG),
        "preflight_sha256": sha256(PREFLIGHT),
        "baseline_row_sha256": sha256(BASELINE_ROW),
        "master_registration_sha256": sha256(registration_path),
        "master_manifest_sha256": sha256(manifest_path),
        "master_component_sha256": EXPECTED_MASTER_SHA256,
        "master_parent_count": manifest.get("physical_independent_parents"),
        "primary_parent_ids_sha256": hashlib.sha256(primary_ids.tobytes()).hexdigest(),
        "primary_shapes": primary_shapes,
        "calibration_sha256": sha256(CALIBRATION),
        "calibration_parent_count": int(len(calibration_ids)),
        "calibration_parent_ids_sha256": hashlib.sha256(calibration_ids.tobytes()).hexdigest(),
        "calibration_shapes": calibration_shapes,
        "calibration_primary_overlap": 0,
        "host": platform.node(),
    }


def load_model():
    from tools.paper2_baseline_suite import equal_pfa_suite as equal_pfa

    equal_pfa.ROOT = ROOT
    equal_pfa.OUTPUTS = ROOT / "outputs"
    equal_pfa.configure_precision()
    torch.set_num_threads(4)
    model, binding = equal_pfa.load("ReCoVP-Hybrid", 17)
    return model, binding


def graph_permutation() -> list[int]:
    return np.random.default_rng(PERMUTATION_SEED).permutation(16).tolist()


def graph_regularizer(
    stage1: torch.nn.Module,
    factors: dict[str, torch.Tensor],
    variant: str,
    permutation: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    basis = factors["dictionary"]
    gram = factors["gram"]
    batch, _, nodes = basis.shape
    weights = gram.abs().square()
    weights = weights.clone()
    weights.diagonal(dim1=-2, dim2=-1).zero_()
    values, neighbors = torch.topk(weights, k=2, dim=-1)
    adjacency = torch.zeros_like(weights)
    adjacency.scatter_(-1, neighbors, values)
    adjacency = torch.maximum(adjacency, adjacency.transpose(-1, -2))
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    degree = adjacency.sum(-1).real
    laplacian = torch.diag_embed(degree).to(torch.complex128) - adjacency

    delay = factors["delay"].double()
    frequency = stage1.frequency.to(device=basis.device)
    unit_spectrum = stage1.unit_spectrum.to(device=basis.device)
    phase = -2j * torch.pi * frequency[None, None] * delay[..., None]
    template = torch.fft.ifft(
        unit_spectrum[None, None] * torch.exp(phase), n=stage1.pulse_spectrum.numel()
    )[..., :basis.shape[1]]
    atom_norm = torch.linalg.vector_norm(factors["mask"].double() * template, dim=-1) + 1e-12
    reconstructed = (factors["mask"].double() * template / atom_norm[..., None]).transpose(1, 2)
    basis_error = torch.linalg.vector_norm(reconstructed - basis) / torch.linalg.vector_norm(basis).clamp_min(1e-30)
    if float(basis_error) > 2e-6:
        raise RuntimeError(f"physical-atom norm reconstruction mismatch: {float(basis_error)}")

    inverse_norm = atom_norm.reciprocal()
    raw = inverse_norm[..., :, None] * laplacian * inverse_norm[..., None, :]
    raw_trace = torch.diagonal(raw, dim1=-2, dim2=-1).real.sum(-1)
    precision_trace = factors["precision"].sum(-1)

    if variant == "graph_weak":
        strength = 0.1
    elif variant in ("graph_strong", "permuted_graph_strong", "diagonal_strong"):
        strength = 1.0
    else:
        raise ValueError(f"graph regularizer requested for unsupported variant {variant}")
    gamma = raw * (strength * precision_trace / raw_trace.clamp_min(1e-30))[:, None, None]
    if variant == "permuted_graph_strong":
        index = torch.as_tensor(permutation, device=basis.device, dtype=torch.long)
        gamma = gamma.index_select(-2, index).index_select(-1, index)
    elif variant == "diagonal_strong":
        diagonal = torch.diagonal(gamma, dim1=-2, dim2=-1).real
        gamma = torch.diag_embed(diagonal).to(torch.complex128)

    gamma = 0.5 * (gamma + gamma.mH)
    diagnostics = {
        "atom_basis_relative_reconstruction_error": float(basis_error),
        "gamma_trace_ratio_mean": float((torch.diagonal(gamma, dim1=-2, dim2=-1).real.sum(-1) / precision_trace).mean()),
        "mean_dictionary_coherence_sq": float(weights.sum((-2, -1)).mean() / (nodes * (nodes - 1))),
    }
    return gamma, diagnostics


def factor_variant(
    model: torch.nn.Module,
    iq: torch.Tensor,
    variant: str,
    permutation: list[int],
) -> dict[str, torch.Tensor]:
    if variant == "baseline":
        return model.factor(iq)

    one = model.stage1(iq)
    gamma, _ = graph_regularizer(model.stage1, one, variant, permutation)
    return solve_stage1_with_penalty(model, iq, one, gamma)


def solve_stage1_with_penalty(
    model: torch.nn.Module,
    iq: torch.Tensor,
    one: dict[str, torch.Tensor],
    gamma: torch.Tensor,
) -> dict[str, torch.Tensor]:
    observed = torch.complex(iq[:, 0], iq[:, 1])
    basis = one["dictionary"]
    gram = one["gram"]
    precision = one["precision"]
    normal = 0.5 * (gram + gram.mH) + torch.diag_embed(precision).to(torch.complex128) + gamma
    right = torch.linalg.solve(normal, basis.mH)
    correction_norm = (basis @ (right @ one["normalized_input"].to(torch.complex128)[..., None]))[..., 0]
    correction = correction_norm.to(torch.complex64) * one["scale"]
    residual = observed - correction
    refined = model.stage2(residual, correction, observed, one["scale"], model.stage1.matched_spectrum)
    final = residual - refined["delta"]
    return {**one, **refined, "right": right, "correction": correction,
            "restored": final, "stage1_restored": residual, "residual": residual}


def states_for(s: np.ndarray, j: np.ndarray, e: np.ndarray) -> np.ndarray:
    from paper2.evaluation_1103_v4 import states_from_components

    parent_count = len(s)
    states = states_from_components(
        s.reshape(parent_count * 3, 8, 1800),
        j.reshape(parent_count * 3, 8, 1800),
        e.reshape(parent_count * 3, 8, 1800),
    )
    return states.reshape(parent_count, 3, 4, 8, 1800)


def restore_flat(model: torch.nn.Module, flat: np.ndarray, variant: str, permutation: list[int]) -> np.ndarray:
    restored = np.empty(flat.shape, dtype=np.complex64)
    for start in range(0, len(flat), SIGNAL_BATCH):
        stop = min(start + SIGNAL_BATCH, len(flat))
        iq = torch.from_numpy(np.stack((flat[start:stop].real, flat[start:stop].imag), axis=1).astype(np.float32, copy=False)).cuda()
        with torch.inference_mode():
            factors = factor_variant(model, iq, variant, permutation)
        restored[start:stop] = factors["restored"].cpu().numpy()
    return restored


def run_signature(bindings: dict[str, Any], registration_sha: str, script_sha: str) -> str:
    payload = json.dumps({
        "bindings": bindings,
        "conditions": CONDITIONS,
        "variants": VARIANTS,
        "registration_sha256": registration_sha,
        "script_sha256": script_sha,
        "pre_registration_sha256": sha256(PRE_REGISTRATION),
        "signal_batch": SIGNAL_BATCH,
        "parent_batch": PARENT_BATCH,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_preregistration() -> None:
    if PRE_REGISTRATION.exists():
        raise FileExistsError(f"pre-registration is already frozen: {PRE_REGISTRATION}")
    bindings = validate_bindings()
    permutation = graph_permutation()
    registration = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "FROZEN_BEFORE_FULL_EVALUATION",
        "script_sha256": sha256(Path(__file__)),
        "permutation": permutation,
        "protocol": protocol_definition(),
        "bindings": bindings,
        "binding_sha256": hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    atomic_json(PRE_REGISTRATION, registration)
    print(json.dumps({
        "status": registration["status"],
        "path": str(PRE_REGISTRATION),
        "script_sha256": registration["script_sha256"],
        "binding_sha256": registration["binding_sha256"],
    }, ensure_ascii=False, indent=2))


def calibration_file(variant: str) -> Path:
    return OUTPUTS / variant / "calibration_partial.npz"


def calculate_calibration(model: torch.nn.Module, variant: str, permutation: list[int], signature: str) -> np.ndarray:
    from tools.hybrid_condition_benchmark import score

    with np.load(CALIBRATION, allow_pickle=False) as archive:
        s, j, e = (archive[key].copy() for key in ("s", "j", "epsilon"))
        ids = archive["parent_id"].copy()
        doppler = archive["doppler_hz"].copy()
    result_path = calibration_file(variant)
    h0 = np.zeros((len(ids), 3), dtype=np.float64)
    completed = 0
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as archive:
            if str(archive["signature"].item()) != signature or str(archive["variant"].item()) != variant:
                raise RuntimeError(f"stale calibration checkpoint exists: {result_path}")
            completed = int(archive["completed_stop"].item())
            if not 0 <= completed <= len(ids):
                raise RuntimeError("invalid calibration resume cursor")
            h0[:completed] = archive["score_h0"][:completed]
            if not np.array_equal(archive["parent_id"], ids):
                raise RuntimeError("calibration resume IDs changed")

    for start in range(completed, len(ids), PARENT_BATCH):
        stop = min(start + PARENT_BATCH, len(ids))
        observed_states = states_for(s[start:stop], j[start:stop], e[start:stop])
        selected = observed_states[:, :, 2:4]
        flat = selected.reshape(-1, 1800)
        restored = restore_flat(model, flat, variant, permutation).reshape(stop - start, 3, 2, 8, 1800)
        scored_states = observed_states.copy()
        scored_states[:, :, 2:4] = restored
        model_output = scored_states.transpose(0, 2, 1, 3, 4)
        scores = score(s[start:stop], j[start:stop], e[start:stop], model_output, doppler[start:stop], {})
        h0[start:stop] = scores["score_h0"]
        atomic_npz(result_path, signature=np.asarray(signature), variant=np.asarray(variant),
                   completed_stop=np.asarray([stop]), parent_id=ids, score_h0=h0)
        print(f"CAL {variant} {stop}/{len(ids)}", flush=True)

    return h0


def condition_file(variant: str, snr: int, jsr: int) -> Path:
    return OUTPUTS / variant / f"snr_{snr}_jsr_{jsr}_partial.npz"


def calculate_condition(
    model: torch.nn.Module,
    variant: str,
    snr: int,
    jsr: int,
    thresholds: dict[str, float],
    permutation: list[int],
    signature: str,
) -> dict[str, np.ndarray]:
    from tools.hybrid_condition_benchmark import select, score

    selected = select(snr, jsr, 0, EXPECTED_PARENT_COUNT)
    s, j, e = selected["s"], selected["j"], selected["epsilon"]
    ids, doppler = selected["parent_id"], selected["doppler_hz"]
    result_path = condition_file(variant, snr, jsr)
    arrays: dict[str, np.ndarray] = {}
    completed = 0
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as archive:
            if str(archive["signature"].item()) != signature or str(archive["variant"].item()) != variant:
                raise RuntimeError(f"stale condition checkpoint exists: {result_path}")
            completed = int(archive["completed_stop"].item())
            if not np.array_equal(archive["parent_id"], ids):
                raise RuntimeError("condition resume IDs changed")
            arrays = {key: archive[key].copy() for key in archive.files if key.startswith("metric__")}
            if arrays:
                arrays = {key.removeprefix("metric__"): value for key, value in arrays.items()}
            if not 0 <= completed <= len(ids):
                raise RuntimeError("invalid condition resume cursor")

    for start in range(completed, len(ids), PARENT_BATCH):
        stop = min(start + PARENT_BATCH, len(ids))
        observations = states_for(s[start:stop], j[start:stop], e[start:stop])
        restored = restore_flat(model, observations.reshape(-1, 1800), variant, permutation)
        restored = restored.reshape(stop - start, 3, 4, 8, 1800)
        model_output = restored.transpose(0, 2, 1, 3, 4)
        scores = score(s[start:stop], j[start:stop], e[start:stop], model_output,
                       doppler[start:stop], thresholds)
        if not arrays:
            arrays = {name: np.full((len(ids),), np.nan, dtype=np.float64) for name in scores}
        for name, values in scores.items():
            arrays[name][start:stop] = np.asarray(values).mean(axis=1)
        serialized = {f"metric__{name}": value for name, value in arrays.items()}
        atomic_npz(result_path, signature=np.asarray(signature), variant=np.asarray(variant),
                   completed_stop=np.asarray([stop]), parent_id=ids, **serialized)
        print(f"DEV {variant} SNR={snr} JSR={jsr} {stop}/{len(ids)}", flush=True)

    return arrays


def measure_latency(model: torch.nn.Module, example: np.ndarray, permutation: list[int]) -> dict[str, dict[str, float]]:
    results = {}
    for variant in VARIANTS:
        results[variant] = {}
        for count in (1, 8):
            sample = example[:count]
            iq = torch.from_numpy(np.stack((sample.real, sample.imag), axis=1).astype(np.float32, copy=False)).cuda()
            for _ in range(3):
                with torch.inference_mode():
                    factor_variant(model, iq, variant, permutation)
            torch.cuda.synchronize()
            timings = []
            for _ in range(12):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    start_event.record()
                    factor_variant(model, iq, variant, permutation)
                    end_event.record()
                end_event.synchronize()
                timings.append(float(start_event.elapsed_time(end_event)))
            key = f"{count}_pulse"
            results[variant][key] = {
                "median_ms": float(np.median(timings)),
                "p95_ms": float(np.quantile(timings, 0.95)),
                "samples": len(timings),
            }
    return results


def validate_numerics(model: torch.nn.Module, permutation: list[int]) -> dict[str, Any]:
    from tools.hybrid_condition_benchmark import select

    chosen = select(-30, 10, 0, 1)
    states = states_for(chosen["s"], chosen["j"], chosen["epsilon"])
    flat = states[0, 0, 3]
    iq = torch.from_numpy(np.stack((flat.real, flat.imag), axis=1).astype(np.float32, copy=False)).cuda()
    with torch.inference_mode():
        baseline = model.factor(iq)
        zero_stage1 = model.stage1(iq)
        graph, graph_diagnostics = graph_regularizer(model.stage1, zero_stage1, "graph_strong", permutation)
        weak, weak_diagnostics = graph_regularizer(model.stage1, zero_stage1, "graph_weak", permutation)
        permuted, permuted_diagnostics = graph_regularizer(model.stage1, zero_stage1, "permuted_graph_strong", permutation)
        diagonal, diagonal_diagnostics = graph_regularizer(model.stage1, zero_stage1, "diagonal_strong", permutation)
        zero_final = solve_stage1_with_penalty(
            model, iq, zero_stage1, torch.zeros_like(zero_stage1["gram"])
        )
        zero_gamma = torch.zeros_like(zero_stage1["gram"])
        zero_trace_ratio = float(
            torch.diagonal(zero_gamma, dim1=-2, dim2=-1).real.sum(-1).mean()
        )
    baseline_delta = float((baseline["restored"] - zero_final["restored"]).abs().max())
    if baseline_delta > 2e-6:
        raise RuntimeError(f"zero-penalty parity failed with max absolute error {baseline_delta}")

    diagnostics = {}
    for name, gamma in (("strong", graph), ("weak", weak), ("permuted", permuted), ("diagonal", diagonal)):
        hermitian_error = float((gamma - gamma.mH).abs().max())
        minimum_eigenvalue = float(torch.linalg.eigvalsh(gamma).amin())
        if hermitian_error > 1e-10 or minimum_eigenvalue < -1e-8:
            raise RuntimeError(f"{name} graph penalty failed the Hermitian PSD check")
        diagnostics[name] = {
            "hermitian_max_abs_error": hermitian_error,
            "minimum_eigenvalue": minimum_eigenvalue,
        }

    precision_trace = zero_stage1["precision"].sum(-1)
    trace_ratios = {
        name: float((torch.diagonal(gamma, dim1=-2, dim2=-1).real.sum(-1) / precision_trace).mean())
        for name, gamma in (("strong", graph), ("weak", weak),
                             ("permuted", permuted), ("diagonal", diagonal))
    }
    if abs(trace_ratios["strong"] - 1.0) > 1e-6 or abs(trace_ratios["weak"] - 0.1) > 1e-6:
        raise RuntimeError(f"graph trace normalization failed: {trace_ratios}")

    return {
        "zero_penalty_max_abs_output_error": baseline_delta,
        "zero_penalty_pass": baseline_delta <= 2e-6,
        "zero_penalty_trace": zero_trace_ratio,
        "graph_diagnostics": diagnostics,
        "trace_ratios": trace_ratios,
        "regularizer_diagnostics": {
            "strong": graph_diagnostics,
            "weak": weak_diagnostics,
            "permuted": permuted_diagnostics,
            "diagonal": diagonal_diagnostics,
        },
        "permutation": permutation,
    }


def bootstrap_summary(results: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, EXPECTED_PARENT_COUNT, size=(BOOTSTRAP_REPLICATES, EXPECTED_PARENT_COUNT))
    metrics = sorted(set.intersection(*(set(data[CONDITIONS[0]]) for data in results.values())))
    condition_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    macro_groups = {
        "macro_extreme": tuple(condition for condition in CONDITIONS if condition[0] < -20),
        "macro_ordinary": tuple(condition for condition in CONDITIONS if -20 <= condition[0] <= 0),
        "macro_mean": CONDITIONS,
    }

    for variant in VARIANTS:
        summarized_conditions = [*CONDITIONS, *macro_groups]
        for condition in summarized_conditions:
            for metric in metrics:
                if isinstance(condition, str):
                    values = np.mean(np.stack([results[variant][cell][metric] for cell in macro_groups[condition]]), axis=0)
                    label = condition
                    snr, jsr = "", ""
                else:
                    values = results[variant][condition][metric]
                    label = f"snr_{condition[0]}_jsr_{condition[1]}"
                    snr, jsr = condition
                condition_rows.append({
                    "variant": variant,
                    "condition": label,
                    "snr_db": snr,
                    "jsr_db": jsr,
                    "metric": metric,
                    "parents": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                })

    comparisons = [(variant, "baseline", "candidate_minus_baseline")
                   for variant in VARIANTS if variant != "baseline"]
    comparisons.extend(("graph_strong", control, f"graph_strong_minus_{control}")
                       for control in ("permuted_graph_strong", "diagonal_strong"))

    for candidate, reference, comparison_name in comparisons:
        comparison_metrics = BOOTSTRAP_METRICS if reference == "baseline" else ("u_nmse", "damage")
        for metric in comparison_metrics:
            if metric not in metrics:
                continue
            for condition in (*CONDITIONS, *macro_groups):
                if isinstance(condition, str):
                    candidate_values = np.mean(np.stack([
                        results[candidate][cell][metric] for cell in macro_groups[condition]
                    ]), axis=0)
                    reference_values = np.mean(np.stack([
                        results[reference][cell][metric] for cell in macro_groups[condition]
                    ]), axis=0)
                    label = condition
                    snr, jsr = "", ""
                else:
                    candidate_values = results[candidate][condition][metric]
                    reference_values = results[reference][condition][metric]
                    label = f"snr_{condition[0]}_jsr_{condition[1]}"
                    snr, jsr = condition
                paired_difference = candidate_values - reference_values
                boot = paired_difference[indices].mean(axis=1)
                lower, upper = np.quantile(boot, (0.025, 0.975))
                bootstrap_rows.append({
                    "variant": candidate,
                    "comparison": comparison_name,
                    "condition": label,
                    "snr_db": snr,
                    "jsr_db": jsr,
                    "metric": metric,
                    "mean_difference": float(np.mean(paired_difference)),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "physical_parents": EXPECTED_PARENT_COUNT,
                })

    return condition_rows, bootstrap_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_decision(condition_rows: list[dict], bootstrap_rows: list[dict]) -> dict[str, Any]:
    by_metric = {
        (r["variant"], r["comparison"], r["condition"], r["metric"]): r
        for r in bootstrap_rows
    }
    candidates = {}
    for variant in ("graph_weak", "graph_strong"):
        comparison = "candidate_minus_baseline"
        eu = by_metric.get((variant, comparison, "macro_mean", "u_nmse"))
        damage = by_metric.get((variant, comparison, "macro_mean", "damage"))
        pd_1pct = by_metric.get((variant, comparison, "macro_mean", "pd_eq_0.01"))
        pfa_1pct = by_metric.get((variant, comparison, "macro_mean", "pfa_eq_0.01"))
        if not (eu and damage and pd_1pct and pfa_1pct):
            continue
        baseline_eu = np.mean([
            row["mean"] for row in condition_rows
            if row["variant"] == "baseline" and row["metric"] == "u_nmse"
            and str(row["condition"]).startswith("snr_")
        ])
        baseline_damage = np.mean([
            row["mean"] for row in condition_rows
            if row["variant"] == "baseline" and row["metric"] == "damage"
            and str(row["condition"]).startswith("snr_")
        ])
        eu_reduction = -eu["mean_difference"] / max(baseline_eu, 1e-30)
        damage_increase = damage["mean_difference"] / max(baseline_damage, 1e-30)
        damage_upper_relative = damage["ci95_upper"] / max(baseline_damage, 1e-30)
        performance_gate = bool(
            eu["ci95_upper"] < 0
            and eu_reduction >= 0.05
            and damage_upper_relative <= 0.05
            and pd_1pct["ci95_lower"] >= -0.02
            and pfa_1pct["ci95_upper"] <= 0.01
        )
        graph_specificity = None
        if variant == "graph_strong":
            control_rows = {}
            for control in ("permuted_graph_strong", "diagonal_strong"):
                control_row = by_metric.get((
                    "graph_strong", f"graph_strong_minus_{control}", "macro_mean", "u_nmse"
                ))
                control_rows[control] = control_row
            graph_specificity = {
                "controls": {
                    control: None if row is None else {
                        "mean_difference": row["mean_difference"],
                        "ci95": [row["ci95_lower"], row["ci95_upper"]],
                    }
                    for control, row in control_rows.items()
                },
                "pass": all(row is not None and row["ci95_upper"] < 0 for row in control_rows.values()),
            }
        candidates[variant] = {
            "macro_eu_relative_reduction": float(eu_reduction),
            "macro_damage_relative_increase": float(damage_increase),
            "macro_damage_relative_increase_ci95_upper": float(damage_upper_relative),
            "macro_pd_1pct_difference": float(pd_1pct["mean_difference"]),
            "macro_pfa_1pct_difference": float(pfa_1pct["mean_difference"]),
            "eu_ci95": [eu["ci95_lower"], eu["ci95_upper"]],
            "damage_ci95": [damage["ci95_lower"], damage["ci95_upper"]],
            "pd_1pct_ci95": [pd_1pct["ci95_lower"], pd_1pct["ci95_upper"]],
            "pfa_1pct_ci95": [pfa_1pct["ci95_lower"], pfa_1pct["ci95_upper"]],
            "performance_gate_pass": performance_gate,
            "graph_specificity": graph_specificity,
            "screening_gate_pass": bool(
                performance_gate and (graph_specificity is None or graph_specificity["pass"])
            ),
        }
    return {
        "scope": "single-checkpoint development screen",
        "criteria": [
            "macro E_U improvement at least 5 percent and paired parent-bootstrap CI95 excludes zero",
            "macro E_S increase CI95 upper no greater than 5 percent of baseline",
            "macro 1 percent equal-PFA detection decline no greater than 2 percentage points",
            "macro 1 percent equal-PFA false-alarm increase CI95 upper no greater than 1 percentage point",
            "graph_strong must outperform both permuted-graph and diagonal controls in paired macro E_U CI95",
            "passing only supports advancing to a trained multi-seed ablation, not final adoption",
        ],
        "candidates": candidates,
    }


def final_outputs(
    results: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
    thresholds: dict[str, dict[str, float]],
    latencies: dict[str, dict[str, float]],
    bindings: dict[str, Any],
    validation: dict[str, Any],
    signature: str,
    permutation: list[int],
    gpu_snapshot_before: dict[str, Any],
) -> None:
    condition_rows, bootstrap_rows = bootstrap_summary(results)
    write_csv(REPORTS / "CONDITION_METRICS.csv", condition_rows)
    write_csv(REPORTS / "PAIRED_BOOTSTRAP.csv", bootstrap_rows)
    latency_rows = []
    for variant, records in latencies.items():
        for batch, summary in records.items():
            latency_rows.append({"variant": variant, "batch": batch, **summary})
    write_csv(REPORTS / "LATENCY.csv", latency_rows)
    decision = make_decision(condition_rows, bootstrap_rows)
    atomic_json(REPORTS / "DECISION.json", decision)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "root": str(ROOT),
        "device": torch.cuda.get_device_name(0),
        "gpu_snapshot_before_run": gpu_snapshot_before,
        "gpu_snapshot_after_run": gpu_snapshot(),
        "run_signature": signature,
        "script_sha256": sha256(Path(__file__)),
        "pre_registration_sha256": sha256(PRE_REGISTRATION),
        "bindings": bindings,
        "conditions": [{"snr_db": snr, "jsr_db": jsr} for snr, jsr in CONDITIONS],
        "parents_per_condition": EXPECTED_PARENT_COUNT,
        "physical_parent_unit": "parent_id; jammer types are repeated measurements averaged within parent",
        "variants": list(VARIANTS),
        "permutation_seed": PERMUTATION_SEED,
        "permutation": permutation,
        "calibration_thresholds": thresholds,
        "zero_gamma_and_psd_validation": validation,
        "latency": latencies,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "elapsed_outputs": "See per-variant calibration and condition partial files under outputs/.",
        "local_arrays_directory": str(OUTPUTS),
    }
    atomic_json(REPORTS / "RUN_MANIFEST.json", manifest)
    atomic_npz(OUTPUTS / "PARENT_METRICS.npz", **{
        f"{variant}__snr_{snr}__jsr_{jsr}__{metric}": values
        for variant, conditions in results.items()
        for (snr, jsr), metrics in conditions.items()
        for metric, values in metrics.items()
    })


def execute() -> None:
    bindings = validate_bindings()
    if not PRE_REGISTRATION.is_file():
        raise FileNotFoundError("registered PREREGISTRATION.json is required before full data evaluation")
    pre_registration = json.loads(PRE_REGISTRATION.read_text(encoding="utf-8"))
    permutation = graph_permutation()
    script_sha = sha256(Path(__file__))
    registration_sha = sha256(DATA_ROOT / "REGISTRATION.json")
    signature = run_signature(bindings, registration_sha, script_sha)
    if pre_registration.get("script_sha256") != script_sha:
        raise RuntimeError("script SHA-256 differs from frozen pre-registration")
    if pre_registration.get("permutation") != permutation:
        raise RuntimeError("pre-registered graph permutation differs from the frozen seed")
    if pre_registration.get("status") != "FROZEN_BEFORE_FULL_EVALUATION":
        raise RuntimeError("pre-registration is not marked as frozen")
    expected_binding_sha = hashlib.sha256(
        json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if pre_registration.get("protocol") != protocol_definition():
        raise RuntimeError("current protocol differs from the frozen pre-registration")
    if pre_registration.get("binding_sha256") != expected_binding_sha:
        raise RuntimeError("current checkpoint, data, or environment binding differs from the pre-registration")

    gpu = gpu_snapshot()
    selected_device = gpu["devices"][0]
    if selected_device["utilization_percent"] >= 85:
        raise RuntimeError(f"selected local GPU utilization is at least 85 percent: {selected_device}")
    model, model_binding = load_model()
    validation = validate_numerics(model, permutation)
    if not validation["zero_penalty_pass"]:
        raise RuntimeError("zero-penalty parity failed")

    thresholds = {}
    for variant in VARIANTS:
        calibration_h0 = calculate_calibration(model, variant, permutation, signature)
        thresholds[variant] = {
            str(pfa): float(np.quantile(calibration_h0.reshape(-1), 1.0 - pfa, method="higher"))
            for pfa in PFA_TARGETS
        }

    eval_jobs = [(variant, snr, jsr) for variant in VARIANTS for snr, jsr in CONDITIONS]
    rng = np.random.default_rng(PERMUTATION_SEED)
    rng.shuffle(eval_jobs)
    results: dict[str, dict[tuple[int, int], dict[str, np.ndarray]]] = {variant: {} for variant in VARIANTS}
    for variant, snr, jsr in eval_jobs:
        threshold_rows = {f"{pfa}": value for pfa, value in thresholds[variant].items()}
        results[variant][(snr, jsr)] = calculate_condition(
            model, variant, snr, jsr, threshold_rows, permutation, signature
        )

    first_condition = next(iter(CONDITIONS))
    selected = __import__("tools.hybrid_condition_benchmark", fromlist=["select"]).select(
        first_condition[0], first_condition[1], 0, 1
    )
    observations = states_for(selected["s"], selected["j"], selected["epsilon"])
    example = observations[0, 0, 3]
    latencies = measure_latency(model, example, permutation)
    final_outputs(results, thresholds, latencies, {**bindings, "loaded_model": model_binding},
                  validation, signature, permutation, gpu)


def validate_only() -> None:
    bindings = validate_bindings()
    permutation = graph_permutation()
    model, model_binding = load_model()
    validation = validate_numerics(model, permutation)
    print(json.dumps({
        "status": "VALIDATION_ONLY_PASS" if validation["zero_penalty_pass"] else "FAIL",
        "bindings": bindings,
        "model_binding": model_binding,
        "gpu": gpu_snapshot(),
        "validation": validation,
        "script_sha256": sha256(Path(__file__)),
    }, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="verify bindings and numerical parity without full evaluation")
    parser.add_argument("--write-preregistration", action="store_true", help="freeze the validated protocol before full evaluation")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
    elif args.write_preregistration:
        write_preregistration()
    else:
        execute()


if __name__ == "__main__":
    main()
