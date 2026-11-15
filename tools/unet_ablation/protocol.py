"""Frozen protocol and verified reuse of the completed original model."""
from __future__ import annotations

from pathlib import Path
import platform
import numpy as np
import torch

from paper2.io import ROOT, read, sha
from tools.durable_io import atomic_copy, atomic_json
from .models import VARIANTS, architecture
from .runtime import require_server, configure_precision, SERVER_HOST, SERVER_ROOT, GPU_INDEX, BUDGET_SECONDS

RUN_ID = "unet_ablation_3seed_256_20261115"
BASE = ROOT / "outputs" / RUN_ID
EVIDENCE = ROOT / "evidence" / "outputs" / RUN_ID
CONFIG = ROOT / "configs" / f"{RUN_ID}.json"
ORIGINAL = ROOT / "outputs" / "paper2_256_epoch_8seed_20261114"
SEEDS = (17, 29, 41)
MILESTONES = tuple(range(32, 257, 32))
DATA = ROOT / "data" / "paper2_1103"


def source_hashes():
    paths = [*sorted((ROOT / "paper2").rglob("*.py")),
             *sorted(Path(__file__).parent.glob("*.py")), ROOT / "tools/durable_io.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def require_equal_file(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"immutable identity changed: {path}")
    else:
        atomic_json(path, value)


def baseline_bindings():
    result = {}
    for seed in SEEDS:
        folder = ORIGINAL / f"seed_{seed}"
        complete = read(folder / "TRAINING_COMPLETE.json")
        cp = folder / "epoch_256.pt"
        digest = sha(cp)
        if complete["milestones"]["256"] != digest or read(cp.with_suffix(".json"))["checkpoint_sha256"] != digest:
            raise RuntimeError(f"original checkpoint damaged: {cp}")
        dev = folder / "dev_epoch_256_v4.json"
        dev_result = read(dev)
        if dev_result["checkpoint_sha256"] != digest or dev_result["data_sha256"] != sha(DATA / "dev.npz"):
            raise RuntimeError(f"original DEV identity mismatch: {dev}")
        raw = dev.with_suffix(".npz")
        if sha(raw) != dev_result["local_parent_values_sha256"]:
            raise RuntimeError(f"original DEV arrays damaged: {raw}")
        for name, expected in complete["identity"]["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError(f"original source drift: {name}")
        result[str(seed)] = {"checkpoint": cp.relative_to(ROOT).as_posix(), "checkpoint_sha256": digest,
                             "complete_sha256": sha(folder / "TRAINING_COMPLETE.json"),
                             "dev": dev.relative_to(ROOT).as_posix(), "dev_sha256": sha(dev),
                             "dev_raw_sha256": sha(raw), "training_data_sha256": complete["identity"]["dataset_sha256"]}
    return result


def freeze():
    require_server()
    precision = configure_precision()
    torch.set_num_threads(4)
    BASE.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    preflight = BASE / "NUMERICAL_PREFLIGHT.json"
    if read(preflight)["source_sha256"] != sources or read(preflight)["stage"] != "NUMERICAL_PREFLIGHT_PASSED":
        raise RuntimeError("server numerical preflight required for the exact current sources")
    data = {f"{role}{suffix}": sha(DATA / f"{role}{suffix}")
            for role in ("train", "dev") for suffix in (".npz", ".receipt.json")}
    for role in ("train", "dev"):
        if read(DATA / f"{role}.receipt.json")["sha256"] != data[f"{role}.npz"]:
            raise RuntimeError(f"dataset receipt mismatch: {role}")
    reused = baseline_bindings()
    if any(row["training_data_sha256"] != data["train.npz"] for row in reused.values()):
        raise RuntimeError("original training data mismatch")
    # U-Net has first priority; ablations are blocked by seed with a fixed shuffled order.
    queue = [{"variant": "unet", "seed": s} for s in SEEDS]
    rng = np.random.default_rng(2026111507)
    for seed in SEEDS:
        queue.extend({"variant": str(v), "seed": seed} for v in rng.permutation(VARIANTS[1:]))
    cfg = {"run_id": RUN_ID, "seeds": list(SEEDS), "variants": list(VARIANTS), "queue": queue,
           "numerical_preflight_sha256": sha(preflight),
           "epochs": 256, "milestones": list(MILESTONES), "batch_size": 32, "steps_per_epoch": 48,
           "microbatch_records": {v: 8 if v == "unet" else 32 for v in VARIANTS},
           "accumulation": "U-Net: four microbatches of8, weighted loss, one clip and Adam step per32 records. GroupNorm is per waveform; no BatchNorm or dropout. Mathematical batch objective preserved; floating-point reduction order may differ.",
           "total_updates_per_run": 12288, "source_sha256": sources, "data_sha256": data,
           "original_epoch256": reused,
           "optimizer": {"class": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0, "clip_grad_norm": 5},
           "lr_schedule": "identical train_256_8seed.learning_rate: warmup 1 epoch, cosine to 2e-5 at128, cosine to2e-6 at256",
           "precision": precision, "parameter_matching": {"original_registered": 112756, "original_in_graph": 59714,
                "unet_registered_and_in_graph": 59714, "active_difference": 0, "relative_active_difference_percent": 0,
                "widths": [18, 42, 74],
                "note": "Exact active-parameter matching. Original 53042 unused gate/propagator parameters retained for frozen identity; no dummy U-Net parameters."},
           "loss_weights": {v: {"endpoint": 1, "pc": 0 if v == "no_pc" else .1,
                                "damage": 0 if v == "no_damage" else 1} for v in VARIANTS},
           "selection": "Fixed epoch256 and all three seeds; milestone DEV is descriptive only; no seed/epoch winner selection.",
           "confirmation": {"role": "fresh independent simulation, held back until all training completes", "parents": 512,
                            "seed": 2026111501, "generator_role": "test", "snr_jsr": "same uniform DEV distribution, no retuning",
                            "directory": f"data/{RUN_ID}/confirmation", "epoch": 256},
           "analysis": {"primary": ["u_nmse", "damage"], "required": ["sjrif_db", "pd"],
                        "secondary": ["false_peak_suppression_db", "frame_pfa_jammer", "cell_pfa_jammer", "phase_deg", "amplitude"],
                        "statistics": "seed mean and sample SD; differences paired on seed and physical parent; seed-averaged parent bootstrap95%; no confirmatory p-values with n=3 seeds",
                        "radar": "unchanged evaluation_1103_v4: 600-sample PC, 8-pulse Doppler, CA-CFAR Pfa=1e-6"},
           "recovery": "immutable full checkpoint each epoch (model, optimizer, RNG, metrics); immutable every32 epochs; latest pointer; bounded controller retries only for transient IO or process interruption",
           "execution": {"ssh_host": "rsdr-b", "hostname": SERVER_HOST, "root": SERVER_ROOT,
                         "cuda_visible_devices": GPU_INDEX, "local_training_allowed": False,
                         "study_time_limit_seconds": BUDGET_SECONDS,
                         "controls": "PAUSE/STOP files under run directory; check before each epoch/new job; keep completed checkpoints"},
           "environment": {"hostname": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}}
    require_equal_file(CONFIG, cfg)
    require_equal_file(BASE / "CONFIG.json", cfg)
    # Freeze exact sources locally. Training data remain in their original verified local location.
    for name, digest in sources.items():
        target = BASE / "source_snapshot" / name
        if not target.exists():
            atomic_copy(ROOT / name, target)
        if sha(target) != digest:
            raise RuntimeError(f"source snapshot mismatch: {name}")
    for v in ("full", *VARIANTS):
        require_equal_file(BASE / v / "ARCHITECTURE.json", architecture(v))
    atomic_json(EVIDENCE / "CONFIG.json", cfg)
    return cfg


def validate_config():
    cfg = read(CONFIG)
    if read(BASE / "CONFIG.json") != cfg or cfg["source_sha256"] != source_hashes():
        raise RuntimeError("frozen source/config mismatch; preserve run and use a new identity")
    if sha(BASE / "NUMERICAL_PREFLIGHT.json") != cfg["numerical_preflight_sha256"]:
        raise RuntimeError("numerical preflight evidence changed")
    for name, digest in cfg["data_sha256"].items():
        if sha(DATA / name) != digest:
            raise RuntimeError(f"frozen dataset mismatch: {name}")
    return cfg


def folder_for(variant, seed):
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError((variant, seed))
    return BASE / variant / f"seed_{seed}"
