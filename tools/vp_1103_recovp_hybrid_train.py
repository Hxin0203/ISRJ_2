"""Crash-recoverable, server-B-only ReCoVP-Hybrid seed-17 training.

Training is intentionally gated on the completed, hash-bound Hybrid preflight.
The model returns restored RAW_IQ; all objectives here receive the corresponding
correction C = y - restored. No CONFIRM or historical TEST paths are opened.
"""
from __future__ import annotations

import argparse
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read, sha
from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import rng_snapshot, restore_rng
from tools.durable_io import ProcessLock, atomic_copy, atomic_json, atomic_write
from tools.vp_1103_recovp_hybrid_objective import hybrid_objective
from tools.vp_1103_recovp_hybrid import ReCoVPHybrid, effective_real_parameters


RUN = "recovp_hybrid_seed17"
SEED = 17
EPOCHS = 256
RECORD_BATCH = 32
MICROBATCH_RECORDS = 8
STEPS_PER_EPOCH = 48
MILESTONES = tuple(range(32, 257, 32))
SERVER_HOST = "DESKTOP-RFFERM3"
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
GPU_INDEX = "1"
GPU_UUID = "GPU-bb60dd3a-738e-2398-908f-7f77ac05549b"
MAX_PARAMS = 59_700
DATA = ROOT / "data" / "paper2_1103"
BASE = ROOT / "outputs" / RUN
EVIDENCE = ROOT / "evidence" / "outputs" / RUN
PREFLIGHT = EVIDENCE / "PREFLIGHT.json"
FORMAT = "recovp-hybrid-seed17-epoch-v1"


def configure_precision() -> dict:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"dtype": "float32", "vp_solve": "complex128",
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def require_server_and_gpu() -> dict:
    """Refuse any local, wrong-root, wrong-device, or wrong-UUID execution."""
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError("Hybrid training is restricted to rsdr-b / DESKTOP-RFFERM3")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"Hybrid training root must be {SERVER_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_INDEX:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=1 before starting Hybrid training")
    if not torch.cuda.is_available():
        raise RuntimeError("Server CUDA unavailable; CPU/local fallback is prohibited")
    query = subprocess.check_output([
        "nvidia-smi", "-i", GPU_INDEX, "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip()
    parts = [p.strip() for p in query.split(",")]
    if len(parts) != 5 or parts[0] != GPU_UUID:
        raise RuntimeError(f"Physical GPU-1 UUID mismatch: {query}")
    free_mib, total_mib, util = map(float, parts[2:])
    if util >= 85:
        raise RuntimeError(f"GPU-1 resource admission failed; preserve existing jobs: {query}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only physical GPU 1")
    return {"hostname": platform.node(), "root": str(ROOT), "visible_devices": GPU_INDEX,
            "physical_gpu_index": 1, "gpu_uuid": parts[0], "gpu_name": parts[1],
            "free_mib_at_start": free_mib, "total_mib": total_mib,
            "utilization_percent_at_start": util,
            "admission_policy": "concurrent use allowed; start only when selected GPU utilization is below 85 percent"}


def source_hashes() -> dict[str, str]:
    names = (
        "tools/vp_1103_recovp_hybrid.py",
        "tools/vp_1103_recovp_hybrid_train.py",
        "tools/vp_1103_recovp_hybrid_objective.py",
        "paper2/train_1103.py",
        "paper2/network.py",
        "paper2/train_256_8seed.py",
        "paper2/io.py",
        "tools/unet_tail_loss/objective.py",
        "tools/durable_io.py",
    )
    return {name: sha(ROOT / name) for name in names}


def data_hashes() -> dict[str, str]:
    names = ("train.npz", "train.receipt.json", "dev.npz", "dev.receipt.json")
    values = {name: sha(DATA / name) for name in names}
    for role in ("train", "dev"):
        receipt = read(DATA / f"{role}.receipt.json")
        if receipt.get("sha256", "").lower() != values[f"{role}.npz"].lower():
            raise RuntimeError(f"{role} dataset receipt does not match the NPZ content")
    return values


def validate_preflight(src: dict[str, str], data: dict[str, str]) -> dict:
    result = read(PREFLIGHT)
    if result.get("stage") != "PREFLIGHT_PASSED" or result.get("status") != "PASS":
        raise RuntimeError("Hybrid preflight has not passed; training remains gated")
    recorded_src = result.get("source_sha256", result.get("source_hashes", {}))
    recorded_data = result.get("data_sha256", result.get("dataset_sha256", {}))
    all_sources = {**src, "tools/vp_1103_recovp_hybrid_preflight.py":
                   sha(ROOT / "tools/vp_1103_recovp_hybrid_preflight.py")}
    for name, digest in all_sources.items():
        if recorded_src.get(name) != digest:
            raise RuntimeError(f"preflight source binding missing or stale: {name}")
    for name in data:
        if recorded_data.get(name) != data[name]:
            raise RuntimeError(f"preflight data binding missing or stale: {name}")
    if result.get("scope") != "TRAIN_AND_DEV_ONLY" or result.get("seed") != SEED:
        raise RuntimeError("preflight scope or seed identity mismatch")
    if result.get("guard", {}).get("training_started") is not False:
        raise RuntimeError("preflight must finish without starting training")
    if result.get("parameter_report", {}).get("total_effective_real_parameters", MAX_PARAMS + 1) > MAX_PARAMS:
        raise RuntimeError("preflight parameter audit failed")
    if result.get("equivariance", {}).get("pass") is not True:
        raise RuntimeError("preflight equivariance check did not pass")
    if result.get("real_train_microbatch", {}).get("all_trainable_tensors_finite_nonzero") is not True:
        raise RuntimeError("preflight B=8 gradient check did not pass")
    return result


def config_payload(src: dict, data: dict, preflight_sha: str,
                   admission: dict, precision: dict) -> dict:
    return {
        "run_id": RUN, "scope": "TRAIN_512_DEV_256_ONLY", "seed": SEED,
        "epochs": EPOCHS, "milestones": list(MILESTONES),
        "training_parents": 512, "jammer_variants": ["direct", "repeat", "cyclic"],
        "records_per_epoch": 1536, "record_batch": RECORD_BATCH,
        "microbatch_records": MICROBATCH_RECORDS, "steps_per_epoch": STEPS_PER_EPOCH,
        "architecture": "ReCoVPHybrid K=16; parameter count bound by preflight",
        "objective": {"formula": "L_new(final) + 0.25 * L_new(Stage1)",
                      "base": "paper2.network.training_loss B4 for each stage",
                      "weights": {"endpoint": 1.0, "pulse_compression": 0.1,
                                  "target_damage": 1.0, "matched_filter_tail": 0.05,
                                  "stage1_auxiliary": 0.25},
                      "tail": "top-8 smooth range-Doppler tail on target-absent jammer state; surrogate only",
                      "correction_semantics": "C_final = observed - factor()['restored']; C_stage1 = observed - factor()['stage1_restored'] = factor()['correction']",
                      "accumulation": "four 8-record microbatches weighted by 8/32; one clip and Adam update per 32 records"},
        "optimizer": {"class": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.0, "clip_grad_norm": 5.0},
        "lr_schedule": "one-epoch linear warmup to 1e-3; cosine to 2e-5 by epoch128 and 2e-6 by epoch256",
        "source_sha256": src, "data_sha256": data,
        "preflight_sha256": preflight_sha,
        "recovery": "immutable epoch_000..256, latest pointer, every-32-epoch milestone, optimizer and RNG state; SHA-256 receipts",
        "execution": {**admission, "local_training": False},
        "precision": precision,
        "data_scope": "Only data/paper2_1103/train.npz and dev.npz are bound; no CONFIRM or TEST access",
        "prepared_utc": now(),
    }


def freeze_config(src: dict, data: dict, preflight: dict, admission: dict,
                  precision: dict) -> dict:
    BASE.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    cfg = config_payload(src, data, sha(PREFLIGHT), admission, precision)
    config_path = BASE / "CONFIG.json"
    evidence_path = EVIDENCE / "CONFIG.json"
    for path in (config_path, evidence_path):
        if path.exists():
            old = read(path)
            stable_old = {k: v for k, v in old.items() if k != "prepared_utc"}
            stable_new = {k: v for k, v in cfg.items() if k != "prepared_utc"}
            if stable_old != stable_new:
                raise RuntimeError(f"frozen run config changed: {path}")
            if path == config_path:
                cfg = old
        else:
            atomic_json(path, cfg)
    return cfg


def learning_rate(epoch: int, batch_index: int) -> float:
    global_step = (epoch - 1) * STEPS_PER_EPOCH + batch_index + 1
    if epoch == 1:
        return 1e-3 * global_step / STEPS_PER_EPOCH
    if epoch <= 128:
        progress = (global_step - STEPS_PER_EPOCH) / max(1, 128 * STEPS_PER_EPOCH - STEPS_PER_EPOCH)
        return float(2e-5 + 0.5 * (1e-3 - 2e-5) * (1 + np.cos(np.pi * progress)))
    extension_step = (epoch - 129) * STEPS_PER_EPOCH + batch_index
    extension_steps = (EPOCHS - 128) * STEPS_PER_EPOCH
    progress = extension_step / max(1, extension_steps - 1)
    return float(2e-6 + 0.5 * (2e-5 - 2e-6) * (1 + np.cos(np.pi * progress)))


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_checkpoint(path: Path, expected: dict) -> dict | None:
    try:
        receipt = read(path.with_suffix(".json"))
        if receipt.get("checkpoint_sha256") != sha(path):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("identity") != expected:
            return None
        if int(payload.get("epoch", -1)) != int(receipt.get("epoch", -2)):
            return None
        if payload.get("global_step") != payload["epoch"] * STEPS_PER_EPOCH:
            return None
        return payload
    except (OSError, ValueError, KeyError, RuntimeError, EOFError):
        return None


def choose_resume(folder: Path, expected: dict) -> dict:
    candidates = []
    for path in [*sorted((folder / "recovery").glob("epoch_*.pt"), reverse=True), folder / "latest.pt"]:
        if path.exists():
            payload = validate_checkpoint(path, expected)
            if payload is not None:
                candidates.append(payload)
    if candidates:
        return max(candidates, key=lambda row: int(row["epoch"]))
    if list((folder / "recovery").glob("*.pt")) or (folder / "latest.pt").exists():
        raise RuntimeError("existing recovery files are invalid; preserve and audit them before restart")
    return {"epoch": -1}


def publish_checkpoint(folder: Path, payload: dict) -> str:
    epoch = int(payload["epoch"])
    archive = folder / "recovery" / f"epoch_{epoch:03d}.pt"
    archive.parent.mkdir(parents=True, exist_ok=True)
    existing = validate_checkpoint(archive, payload["identity"]) if archive.exists() else None
    if existing is not None:
        if existing["epoch_record"] != payload.get("epoch_record"):
            raise RuntimeError(f"immutable recovery record differs: {archive}")
        digest = sha(archive)
    else:
        if archive.exists() or archive.with_suffix(".json").exists():
            raise RuntimeError(f"invalid immutable recovery file exists: {archive}")
        atomic_write(archive, lambda stream: torch.save(payload, stream))
        digest = sha(archive)
        atomic_json(archive.with_suffix(".json"), {
            "epoch": epoch, "identity": payload["identity"],
            "checkpoint_sha256": digest, "updated_utc": now()})
    for target in (folder / "latest.pt", *((folder / f"epoch_{epoch:03d}.pt",) if epoch in MILESTONES else ())):
        if target.exists() and target.name != "latest.pt" and sha(target) != digest:
            raise RuntimeError(f"immutable milestone changed: {target}")
        atomic_copy(archive, target)
        atomic_json(target.with_suffix(".json"), {
            "epoch": epoch, "identity": payload["identity"],
            "checkpoint_sha256": digest, "updated_utc": now()})
    record = payload.get("epoch_record")
    if record is not None:
        atomic_json(folder / f"training_epoch_{epoch:03d}.json",
                    {**record, "checkpoint_sha256": digest})
    return digest


def grouped_iq(correction: torch.Tensor, record_count: int) -> torch.Tensor:
    """[record*8,4,2,T] -> [record,8,4,2,T] for the tail objective."""
    return correction.reshape(record_count, 8, 4, 2, correction.shape[-1])


def microbatch_loss(net: ReCoVPHybrid, batch: dict, record_count: int) -> tuple[dict, dict]:
    if len(batch["s"]) != record_count * 8:
        raise ValueError("record_count disagrees with the B=8 batch")
    return hybrid_objective(net, batch, need_details=True)


def optimizer_step(net: ReCoVPHybrid, optimizer, arrays: dict, ids: np.ndarray,
                   epoch: int, batch_index: int, device: torch.device) -> dict:
    optimizer.zero_grad(set_to_none=True)
    sums = {k: 0.0 for k in ("total", "endpoint", "pc", "damage", "tail",
                             "final_total", "stage1_total", "final_tail", "stage1_tail")}
    for lo in range(0, len(ids), MICROBATCH_RECORDS):
        subids = ids[lo:lo + MICROBATCH_RECORDS]
        batch = make_batch(arrays, subids, device)
        losses, tail_diag = microbatch_loss(net, batch, len(subids))
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"nonfinite loss at epoch={epoch}, update={batch_index}")
        weight = len(subids) / len(ids)
        (losses["total"] * weight).backward()
        for key, value in losses.items():
            sums[key] += float(value.detach()) * weight
        sums["tail_peak_db"] = sums.get("tail_peak_db", 0.0) + float(tail_diag["tail_peak_db_mean"]) * weight
        sums["stage1_tail_peak_db"] = sums.get("stage1_tail_peak_db", 0.0) + float(tail_diag["stage1_tail_peak_db"]) * weight
        del batch, losses, tail_diag
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
    optimizer.step()
    return {**sums, "gradient_norm": float(grad_norm), "lr": learning_rate(epoch, batch_index)}


def train(seed: int) -> dict:
    if seed != SEED:
        raise ValueError("This registered run permits seed 17 only")
    admission = require_server_and_gpu()
    precision = configure_precision()
    torch.set_num_threads(4)
    src = source_hashes()
    data = data_hashes()
    preflight = validate_preflight(src, data)
    cfg = freeze_config(src, data, preflight, admission, precision)
    config_sha = sha(BASE / "CONFIG.json")
    identity = {"run": RUN, "seed": seed, "config_sha256": config_sha,
                "train_sha256": data["train.npz"], "source_sha256": src,
                "preflight_sha256": sha(PREFLIGHT)}
    folder = BASE / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessLock(folder / "PROCESS.lock"):
        complete_path = folder / "TRAINING_COMPLETE.json"
        if complete_path.exists():
            result = read(complete_path)
            if result.get("identity") != identity:
                raise RuntimeError("completed Hybrid training identity mismatch")
            return result
        arrays = load_arrays(DATA / "train.npz")
        if len(arrays["s"]) != 512:
            raise RuntimeError(f"expected 512 TRAIN parents, got {len(arrays['s'])}")
        if arrays["s"].shape[-2:] != (8, 1800):
            raise RuntimeError(f"unexpected TRAIN shape: {arrays['s'].shape}")
        seed_rng(seed)
        model = ReCoVPHybrid().cuda()
        if effective_real_parameters(model) > 59_700:
            raise RuntimeError("model exceeds the registered effective-parameter budget")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0.0)
        recovered = choose_resume(folder, identity)
        if int(recovered["epoch"]) >= 0:
            model.load_state_dict(recovered["model"], strict=True)
            optimizer.load_state_dict(recovered["optimizer"])
            restore_rng(recovered["rng"])
            start_epoch = int(recovered["epoch"])
            publish_checkpoint(folder, recovered)
        else:
            start_epoch = 0
            initial = {"format": FORMAT, "identity": identity, "seed": seed,
                       "epoch": 0, "global_step": 0, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "rng": rng_snapshot(),
                       "epoch_record": None, "architecture": model.parameter_report(),
                       "created_utc": now()}
            publish_checkpoint(folder, initial)
        previous_time = time.perf_counter()
        for epoch in range(start_epoch + 1, EPOCHS + 1):
            model.train()
            epoch_rows = []
            for batch_index, ids in enumerate(batches(len(arrays["s"]), epoch, seed)):
                lr = learning_rate(epoch, batch_index)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                row = optimizer_step(model, optimizer, arrays, ids, epoch, batch_index,
                                     torch.device("cuda"))
                epoch_rows.append(row)
                if batch_index == 0 or (batch_index + 1) % 12 == 0:
                    status = {"stage": "TRAINING", "seed": seed, "epoch": epoch,
                              "epochs": EPOCHS, "update": batch_index + 1,
                              "updates_per_epoch": STEPS_PER_EPOCH,
                              "last_committed_epoch": epoch - 1, "metrics": row,
                              "updated_utc": now()}
                    atomic_json(BASE / "STATUS.json", status)
                    atomic_json(EVIDENCE / "STATUS.json", status)
            if len(epoch_rows) != STEPS_PER_EPOCH:
                raise RuntimeError(f"expected {STEPS_PER_EPOCH} updates; got {len(epoch_rows)}")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - previous_time
            previous_time = time.perf_counter()
            metrics = {key: float(np.mean([row[key] for row in epoch_rows]))
                       for key in ("total", "final_total", "stage1_total", "endpoint", "pc",
                                   "damage", "tail", "tail_peak_db", "stage1_tail_peak_db",
                                   "gradient_norm", "lr")}
            record = {"run": RUN, "seed": seed, "epoch": epoch,
                      "global_step": epoch * STEPS_PER_EPOCH,
                      "updates_this_epoch": STEPS_PER_EPOCH, **metrics,
                      "epoch_elapsed_seconds": elapsed, "updated_utc": now()}
            payload = {"format": FORMAT, "identity": identity, "seed": seed,
                       "epoch": epoch, "global_step": epoch * STEPS_PER_EPOCH,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng_snapshot(), "epoch_record": record,
                       "architecture": model.parameter_report(), "created_utc": now()}
            digest = publish_checkpoint(folder, payload)
            status = {"stage": "EPOCH_SAVED", "seed": seed, "epoch": epoch,
                      "checkpoint_sha256": digest, "metrics": metrics, "updated_utc": now()}
            atomic_json(BASE / "STATUS.json", status)
            atomic_json(EVIDENCE / "STATUS.json", status)
            print(f"ReCoVP-Hybrid seed=17 epoch={epoch}/256 loss={metrics['total']:.7g} sec={elapsed:.1f}", flush=True)
        recovery = {}
        milestones = {}
        for epoch in range(0, EPOCHS + 1):
            path = folder / "recovery" / f"epoch_{epoch:03d}.pt"
            payload = validate_checkpoint(path, identity)
            if payload is None:
                raise RuntimeError(f"recovery audit failed: {path}")
            recovery[str(epoch)] = sha(path)
            if epoch in MILESTONES:
                milestone = folder / f"epoch_{epoch:03d}.pt"
                if sha(milestone) != recovery[str(epoch)]:
                    raise RuntimeError(f"milestone hash mismatch: {milestone}")
                milestones[str(epoch)] = sha(milestone)
        result = {"stage": "TRAINING_COMPLETE", "identity": identity,
                  "seed": seed, "epochs": EPOCHS, "updates": EPOCHS * STEPS_PER_EPOCH,
                  "milestones": milestones, "recovery_sha256": recovery,
                  "completed_utc": now()}
        atomic_json(complete_path, result)
        atomic_json(EVIDENCE / "TRAINING_COMPLETE.json", result)
        atomic_json(BASE / "STATUS.json", {"stage": "TRAINING_COMPLETE",
                    "seed": seed, "identity": identity, "updated_utc": now()})
        atomic_json(EVIDENCE / "STATUS.json", read(BASE / "STATUS.json"))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(17,),
                        help="the registered single seed is 17")
    args = parser.parse_args()
    result = train(args.seed)
    print({"stage": result["stage"], "seed": result["seed"],
           "epochs": result["epochs"], "updates": result["updates"]}, flush=True)


if __name__ == "__main__":
    main()
