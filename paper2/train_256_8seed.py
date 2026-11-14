"""Crash-resilient extension of the 1103 CorrectionNet training.

This is a separate run from ``train_1103``. It preserves the original run,
copies and verifies its 32-epoch checkpoints for seeds 17/29/41, then extends
those seeds and trains five predeclared seeds through epoch 256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .data_1103 import generate, verify
from .dev_256_8seed_v4 import evaluate_checkpoint
from .io import ROOT, now, read, sha, write
from .network import CorrectionNet, training_loss
from .train_1103 import BATCH, JAM_TYPES, batches, load_arrays, make_batch
from tools.durable_io import (ProcessLock, atomic_copy, atomic_json, atomic_write,
                              replace_retry)

OLD_BASE = ROOT / "outputs" / "paper2_1103_retry_20261114"
RUN_ID = "paper2_256_epoch_8seed_20261114"
BASE = ROOT / "outputs" / RUN_ID
OLD_SEEDS = (17, 29, 41)
NEW_SEEDS = (53, 67, 79, 97, 109)
SEEDS = OLD_SEEDS + NEW_SEEDS
MILESTONES = tuple(range(32, 257, 32))
TOTAL_EPOCHS = 256
OLD_EPOCHS = 128
STEPS_PER_EPOCH = 48
CHECKPOINT_FORMAT = "paper2-256-8seed-v1"


def source_hashes() -> dict[str, str]:
    values = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
              for p in sorted((ROOT / "paper2").rglob("*.py"))}
    for name in ("tools/aggregate_convergence_256_8seed.py", "tools/durable_io.py",
                 "tools/training_progress_256.py"):
        values[name] = sha(ROOT / name)
    return values


def config_for(dataset_sha: str, data_receipt_sha: str, old_training_sha: str,
               sources: dict[str, str], old_history_sha256: dict[str, str]) -> dict:
    return {
        "run_id": RUN_ID,
        "seeds": list(SEEDS),
        "continued_seeds": list(OLD_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "epochs": TOTAL_EPOCHS,
        "milestone_epochs": list(MILESTONES),
        "batch_size": BATCH,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epoch_definition": "one pass over each of 512 parents x 3 jammer variants",
        "optimizer": "Adam",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "loss": "B4 endpoint + 0.1 pulse-compression + target-response difference",
        "lr_schedule": {
            "epochs_1_128": "preserve original 1-epoch warm-up and cosine 1e-3 to 2e-5",
            "epochs_129_256": "cosine 2e-5 to 2e-6, inclusive endpoints",
        },
        "recovery": "immutable full recovery archive every epoch; latest copy; immutable 32-epoch milestones; all SHA-verified",
        "architecture": "CorrectionNet B4; architecture JSON, module repr and exact sources snapshotted locally",
        "training_data_sha256": dataset_sha,
        "training_data_receipt_sha256": data_receipt_sha,
        "parent_training_complete_sha256": old_training_sha,
        "parent_training_history_sha256": old_history_sha256,
        "source_sha256": sources,
        "selection_rule": "No TEST-based selection; evaluate on the fixed DEV v4 protocol every 32 epochs.",
    }


def atomic_torch_save(value, path: Path) -> str:
    atomic_write(path, lambda stream: torch.save(value, stream))
    return sha(path)


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_identity(seed: int, dataset_sha: str, config_sha: str,
                        source_sha: dict[str, str]) -> dict:
    return {"seed": seed, "dataset_sha256": dataset_sha,
            "config_sha256": config_sha, "source_sha256": source_sha}


def validate_candidate(path: Path, receipt_path: Path, expected: dict):
    if not path.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = read(receipt_path)
        actual_sha = sha(path)
        if receipt.get("checkpoint_sha256") != actual_sha:
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != CHECKPOINT_FORMAT or payload.get("identity") != expected:
            return None
        if int(payload.get("epoch", -1)) != int(receipt.get("epoch", -2)):
            return None
        return payload, actual_sha
    except Exception:
        return None


def choose_resume(folder: Path, expected: dict):
    """Return the highest verified epoch/segment, including immutable archives.

    A stale latest pointer must never hide a newer durable recovery archive.
    Check newest candidates first to avoid deserializing all past epochs.
    """
    candidates = []
    paths = [folder / "latest.pt", *folder.glob("epoch_*.pt"),
             *folder.glob("recovery/epoch_*_segment_*.pt")]
    for path in paths:
        try:
            receipt = read(path.with_suffix(".json"))
            epoch, segment = int(receipt["epoch"]), int(receipt.get("segment", 0))
            candidates.append((epoch, segment, path.name == "latest.pt", path))
        except (OSError, KeyError, ValueError):
            continue
    for epoch, segment, _, path in sorted(candidates, reverse=True):
        valid = validate_candidate(path, path.with_suffix(".json"), expected)
        if valid is not None and (int(valid[0]["epoch"]), int(valid[0].get("segment", 0))) == (epoch, segment):
            return path, valid[0], valid[1]
    return None


def optimizer_for(net):
    return torch.optim.Adam(net.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                            weight_decay=0)


def learning_rate(epoch: int, batch_index: int) -> float:
    global_step = (epoch - 1) * STEPS_PER_EPOCH + batch_index + 1
    if epoch == 1:
        return 1e-3 * global_step / STEPS_PER_EPOCH
    if epoch <= OLD_EPOCHS:
        total_steps = OLD_EPOCHS * STEPS_PER_EPOCH
        progress = (global_step - STEPS_PER_EPOCH) / max(1, total_steps - STEPS_PER_EPOCH)
        return float(2e-5 + .5 * (1e-3 - 2e-5) * (1 + np.cos(np.pi * progress)))
    extension_steps = (TOTAL_EPOCHS - OLD_EPOCHS) * STEPS_PER_EPOCH
    extension_step = (epoch - OLD_EPOCHS - 1) * STEPS_PER_EPOCH + batch_index
    progress = extension_step / max(1, extension_steps - 1)
    return float(2e-6 + .5 * (2e-5 - 2e-6) * (1 + np.cos(np.pi * progress)))


def rng_snapshot() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_rng(seed: int) -> None:
    """Seed all training RNGs before model construction for new runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pack_checkpoint(seed: int, epoch: int, segment: int, net, optimizer,
                    identity: dict, epoch_record: dict | None) -> dict:
    return {"format": CHECKPOINT_FORMAT, "seed": seed, "epoch": epoch,
            "segment": segment, "identity": identity,
            "model": net.state_dict(), "optimizer": optimizer.state_dict(),
            "global_step": epoch * STEPS_PER_EPOCH,
            "rng": rng_snapshot(), "epoch_record": epoch_record,
            "architecture": architecture(net),
            "created_utc": now()}


def architecture(net):
    return {"class": "paper2.network.CorrectionNet", "objective": "B4",
            "parameter_count": sum(p.numel() for p in net.parameters()),
            "module_repr": str(net),
            "state_shapes": {key: list(value.shape) for key, value in net.state_dict().items()}}


def write_recovery(folder: Path, payload: dict, milestone: bool = False) -> tuple[str, str | None]:
    epoch, segment = int(payload["epoch"]), int(payload.get("segment", 0))
    archive = folder / "recovery" / f"epoch_{epoch:03d}_segment_{segment:03d}.pt"
    receipt = archive.with_suffix(".json")
    valid = validate_candidate(archive, receipt, payload["identity"])
    if valid is None:
        # Preserve incomplete/corrupted files as forensic evidence before retry.
        for path in (archive, receipt):
            if path.exists():
                replace_retry(path, path.with_name(path.name + f".invalid_{time.time_ns()}"))
        latest_sha = atomic_torch_save(payload, archive)
        record = {"seed": payload["seed"], "epoch": epoch, "segment": segment,
                  "checkpoint_sha256": latest_sha, "identity": payload["identity"],
                  "updated_utc": now()}
        atomic_json(receipt, record)
    else:
        # Never overwrite a committed recovery archive with a different state.
        if (valid[0].get("epoch_record") != payload.get("epoch_record") or any(
                not torch.equal(value.cpu(), valid[0]["model"][name].cpu())
                for name, value in payload.get("model", {}).items())):
            raise RuntimeError(f"immutable recovery epoch record changed: {archive}")
        latest_sha = valid[1]
    atomic_copy(archive, folder / "latest.pt")
    atomic_json(folder / "latest.json", {"epoch": epoch, "segment": segment,
                 "checkpoint_sha256": latest_sha, "archive": str(archive.relative_to(folder)),
                 "config_sha256": payload["identity"]["config_sha256"], "updated_utc": now()})
    milestone_sha = None
    if milestone and epoch in MILESTONES:
        cp = folder / f"epoch_{epoch:03d}.pt"
        receipt = cp.with_suffix(".json")
        if cp.exists() or receipt.exists():
            valid = validate_candidate(cp, receipt, payload["identity"])
            if valid is not None and valid[1] == sha(cp):
                milestone_sha = valid[1]
            else:
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{time.time_ns()}"
                if cp.exists():
                    replace_retry(cp, cp.with_name(f"{cp.stem}.invalid_{stamp}{cp.suffix}"))
                if receipt.exists():
                    replace_retry(receipt, receipt.with_name(f"{receipt.stem}.invalid_{stamp}{receipt.suffix}"))
                atomic_copy(archive, cp)
                milestone_sha = latest_sha
                atomic_json(receipt, {"seed": payload["seed"], "epoch": epoch, "segment": segment,
                                      "checkpoint_sha256": milestone_sha,
                                      "config_sha256": payload["identity"]["config_sha256"],
                                      "dataset_sha256": payload["identity"]["dataset_sha256"],
                                      "source_sha256": payload["identity"]["source_sha256"],
                                      "updated_utc": now()})
        else:
            atomic_copy(archive, cp)
            milestone_sha = latest_sha
            atomic_json(receipt, {"seed": payload["seed"], "epoch": epoch, "segment": segment,
                                  "checkpoint_sha256": milestone_sha,
                                  "config_sha256": payload["identity"]["config_sha256"],
                                  "dataset_sha256": payload["identity"]["dataset_sha256"],
                                  "source_sha256": payload["identity"]["source_sha256"],
                                  "updated_utc": now()})
    return latest_sha, milestone_sha


def verify_old_run(dataset_sha: str) -> tuple[dict, dict[int, dict]]:
    complete_path = OLD_BASE / "TRAINING_COMPLETE.json"
    complete = read(complete_path)
    if complete.get("data_sha256") != dataset_sha:
        raise RuntimeError("training data differs from the verified epoch-128 run")
    by_seed = {int(run["seed"]): run for run in complete["runs"]}
    if set(by_seed) != set(OLD_SEEDS):
        raise RuntimeError("unexpected seed roster in the source training run")
    verified = {}
    for seed in OLD_SEEDS:
        run = by_seed[seed]
        folder = OLD_BASE / f"seed_{seed}"
        for epoch in (32, 64, 96, 128):
            cp = folder / f"epoch_{epoch:03d}.pt"
            receipt = read(cp.with_suffix(".json"))
            actual = sha(cp)
            if actual != run["checkpoints"][str(epoch)] or actual != receipt["checkpoint_sha256"]:
                raise RuntimeError(f"source checkpoint hash mismatch: {cp}")
            if receipt["dataset_sha256"] != dataset_sha:
                raise RuntimeError(f"source checkpoint data identity mismatch: {cp}")
            verified[(seed, epoch)] = {"path": cp, "sha256": actual, "receipt": receipt}
    return complete, verified


def install_ancestry(folder: Path, seed: int, verified: dict, identity: dict) -> dict:
    rows = []
    # Preserve the original 128-epoch checkpoint in its source run. The new
    # run stores a recovery-envelope copy at epoch 128 for exact continuation.
    for epoch in (32, 64, 96):
        source = verified[(seed, epoch)]
        target = folder / f"epoch_{epoch:03d}.pt"
        target_receipt = target.with_suffix(".json")
        if target.exists() and target_receipt.exists():
            saved = read(target_receipt)
            if saved.get("ancestry_source_sha256") == source["sha256"]:
                valid = validate_candidate(target, target_receipt, identity)
                if valid is not None:
                    rows.append({"epoch": epoch, "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                                 "sha256": valid[1], "ancestry_source_sha256": source["sha256"]})
                    continue
        if target.exists():
            if sha(target) != source["sha256"]:
                raise RuntimeError(f"refusing to overwrite a different existing checkpoint: {target}")
        else:
            shutil.copy2(source["path"], target)
        if sha(target) != source["sha256"]:
            raise RuntimeError(f"copy verification failed: {target}")
        if target_receipt.exists():
            saved = read(target_receipt)
            if saved.get("checkpoint_sha256") != source["sha256"]:
                raise RuntimeError(f"refusing to overwrite a different receipt: {target_receipt}")
        else:
            atomic_json(target_receipt, {**source["receipt"],
                                         "copied_from": str(source["path"].relative_to(ROOT)).replace("\\", "/"),
                                         "checkpoint_sha256": source["sha256"]})
        rows.append({"epoch": epoch, "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha(target), "ancestry_source_sha256": source["sha256"]})
    atomic_json(folder / "ANCESTRY.json", {"seed": seed, "checkpoints": rows,
                 "original_run": str(OLD_BASE.relative_to(ROOT)).replace("\\", "/"),
                 "epoch_128_continuation_source_sha256": verified[(seed, 128)]["sha256"],
                 "preserved_source": True, "created_utc": now()})
    return {"checkpoints": rows}


def copy_legacy_epoch128_into_recovery(folder: Path, seed: int, dataset_sha: str,
                                       config_sha: str, sources: dict[str, str],
                                       verified: dict, net, optimizer) -> tuple[int, int]:
    cp = verified[(seed, 128)]["path"]
    legacy = torch.load(cp, map_location="cpu", weights_only=False)
    if legacy.get("seed") != seed or legacy.get("epoch") != 128 or legacy.get("dataset_sha256") != dataset_sha:
        raise RuntimeError(f"legacy checkpoint metadata mismatch: {cp}")
    net.load_state_dict(legacy["state"])
    optimizer.load_state_dict(legacy["optimizer"])
    # The legacy checkpoint does not contain RNG state. Sampling is epoch-seeded;
    # initialize and record deterministic RNG state for any stochastic layers.
    torch.manual_seed(seed + 128)
    np.random.seed(seed + 128)
    random.seed(seed + 128)
    identity = checkpoint_identity(seed, dataset_sha, config_sha, sources)
    payload = pack_checkpoint(seed, 128, 0, net, optimizer, identity,
                              {"epoch": 128, "source": "verified_legacy_checkpoint",
                               "legacy_checkpoint_sha256": verified[(seed, 128)]["sha256"]})
    milestone_path = folder / "epoch_128.pt"
    if milestone_path.exists() or milestone_path.with_suffix(".json").exists():
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{time.time_ns()}"
        if milestone_path.exists():
            os.replace(milestone_path, milestone_path.with_name(
                f"{milestone_path.stem}.invalid_{stamp}{milestone_path.suffix}"))
        receipt_path = milestone_path.with_suffix(".json")
        if receipt_path.exists():
            os.replace(receipt_path, receipt_path.with_name(
                f"{receipt_path.stem}.invalid_{stamp}{receipt_path.suffix}"))
    milestone_sha = atomic_torch_save(payload, milestone_path)
    atomic_json(milestone_path.with_suffix(".json"), {
        "seed": seed, "epoch": 128, "checkpoint_sha256": milestone_sha,
        "ancestry_source_sha256": verified[(seed, 128)]["sha256"],
        "config_sha256": config_sha, "dataset_sha256": dataset_sha,
        "source_sha256": sources, "updated_utc": now()})
    ancestry_path = folder / "EPOCH_128_ANCESTRY.json"
    atomic_json(ancestry_path, {
        "seed": seed, "epoch": 128,
        "source_checkpoint": str(cp.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": verified[(seed, 128)]["sha256"],
        "continuation_checkpoint": str(milestone_path.relative_to(ROOT)).replace("\\", "/"),
        "continuation_checkpoint_sha256": milestone_sha,
        "weights_loaded_from_source_checkpoint": True,
        "created_utc": now()})
    write_recovery(folder, payload, milestone=True)
    return 128, 0


def append_epoch_if_missing(folder: Path, record: dict, checkpoint_sha: str) -> None:
    path = folder / "events.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (old.get("event") == "epoch_complete" and old.get("seed") == record.get("seed")
                    and old.get("epoch") == record.get("epoch")
                    and old.get("segment") == record.get("segment")
                    and old.get("checkpoint_sha256") == checkpoint_sha):
                return
    append_event(path, {"event": "epoch_complete", **record,
                        "checkpoint_sha256": checkpoint_sha,
                        "recorded_utc": now()})


def write_epoch_receipt(folder: Path, record: dict, checkpoint_sha: str) -> None:
    """Persist immutable per-epoch metrics separately from milestone weights."""
    path = folder / (f"training_epoch_{int(record['epoch']):03d}_"
                     f"segment_{int(record['segment']):03d}.json")
    value = {**record, "checkpoint_sha256": checkpoint_sha,
             "recorded_utc": now()}
    if path.exists():
        if read(path) != value:
            # A crash can happen after the atomic write but before the event
            # append. In that case retain the original timestamp and require
            # the scientific record itself to match exactly.
            saved = read(path)
            if any(saved.get(key) != item for key, item in value.items()
                   if key != "recorded_utc"):
                raise RuntimeError(f"immutable training epoch receipt changed: {path}")
        return
    atomic_json(path, value)


def preserve_invalid_latest(folder: Path) -> None:
    path = folder / "latest.pt"
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = folder / f"latest.invalid_{stamp}.pt"
    if not target.exists():
        shutil.copy2(path, target)


def train_one(seed: int, arrays: dict, dataset_sha: str, config_sha: str,
              sources: dict[str, str], verified_old: dict) -> dict:
    folder = BASE / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    complete_path = folder / "TRAINING_COMPLETE.json"
    identity = checkpoint_identity(seed, dataset_sha, config_sha, sources)
    if complete_path.exists():
        old = read(complete_path)
        if old.get("identity") != identity:
            raise RuntimeError(f"completed seed identity mismatch: {seed}")
        for epoch in MILESTONES:
            if sha(folder / f"epoch_{epoch:03d}.pt") != old["milestones"][str(epoch)]:
                raise RuntimeError(f"completed checkpoint hash mismatch: seed={seed} epoch={epoch}")
            evaluate_checkpoint(seed, epoch)
        if sha(folder / "events.jsonl") != old["training_events_sha256"]:
            raise RuntimeError(f"completed seed training event hash mismatch: {seed}")
        for name, digest in old["training_epoch_record_sha256"].items():
            if sha(folder / f"training_epoch_{name}.json") != digest:
                raise RuntimeError(f"completed seed epoch metric hash mismatch: {seed} {name}")
        return old

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CorrectionNet().to(device)
    optimizer = optimizer_for(net)
    segment = 0
    recovery_payload = None
    if seed in OLD_SEEDS:
        install_ancestry(folder, seed, verified_old, identity)

    chosen = choose_resume(folder, identity)
    if chosen is None:
        if (folder / "latest.pt").exists():
            preserve_invalid_latest(folder)
            if seed not in OLD_SEEDS:
                append_event(folder / "events.jsonl", {"event": "restart_from_scratch_no_valid_checkpoint",
                              "seed": seed, "segment": segment, "recorded_utc": now()})
        if seed in OLD_SEEDS:
            start_epoch, segment = copy_legacy_epoch128_into_recovery(
                folder, seed, dataset_sha, config_sha, sources, verified_old, net, optimizer)
        else:
            start_epoch = 0
    else:
        source_path, payload, recovery_sha = chosen
        segment = int(payload.get("segment", 0))
        if source_path.name != "latest.pt":
            preserve_invalid_latest(folder)
            existing = [int(p.stem.rsplit("_", 1)[1]) for p in
                        folder.glob("training_epoch_*_segment_*.json")]
            segment = max([segment, *existing]) + 1
            append_event(folder / "events.jsonl", {"event": "resume_fallback", "seed": seed,
                          "restored_epoch": int(payload["epoch"]), "segment": segment,
                          "recovery_source": source_path.name,
                          "recovery_sha256": recovery_sha, "recorded_utc": now()})
        net.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        start_epoch = int(payload["epoch"])
        recovery_payload = payload
        if payload.get("epoch_record", {}).get("updates_this_epoch") and start_epoch > 0:
            write_epoch_receipt(folder, payload["epoch_record"], recovery_sha)
            append_epoch_if_missing(folder, payload["epoch_record"], recovery_sha)

    # Close the crash window between writing an epoch-32k recovery point and
    # its immutable milestone copy before any new optimization starts.
    if start_epoch in MILESTONES and start_epoch > 0:
        if recovery_payload is None:
            recovery_payload = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
        write_recovery(folder, recovery_payload, milestone=True)

    # Resume or interruption must never leave a milestone without its DEV
    # convergence measurement. Existing matching results are reused by SHA.
    for epoch in MILESTONES:
        if epoch <= start_epoch:
            evaluate_checkpoint(seed, epoch)
    from tools.training_progress_256 import refresh as refresh_convergence
    refresh_convergence()

    if start_epoch > TOTAL_EPOCHS:
        raise RuntimeError(f"seed {seed} is at epoch {start_epoch} but has no completion receipt")
    n = arrays["s"].shape[0]
    steps = int(np.ceil(n * len(JAM_TYPES) / BATCH))
    if steps != STEPS_PER_EPOCH:
        raise RuntimeError(f"unexpected steps per epoch: {steps}")
    run_start = time.perf_counter()
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        epoch_start = time.perf_counter()
        epoch_losses = []
        for batch_index, ids in enumerate(batches(n, epoch, seed)):
            lr = learning_rate(epoch, batch_index)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = make_batch(arrays, ids, device)
            optimizer.zero_grad(set_to_none=True)
            y = batch["y"]
            corr = net(y).reshape(y.shape[0] // 4, 4, 2, y.shape[-1])
            losses = training_loss(corr, batch["s"], batch["j"], batch["e"], "B4")
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"nonfinite loss at seed={seed} epoch={epoch}")
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            row = {key: float(value.detach()) for key, value in losses.items()}
            row.update(lr=float(lr), gradient_norm=float(grad_norm))
            epoch_losses.append(row)

        if device.type == "cuda":
            torch.cuda.synchronize()
        mean_losses = {key: float(np.mean([item[key] for item in epoch_losses]))
                       for key in ("total", "endpoint", "pc", "damage", "lr", "gradient_norm")}
        step = epoch * STEPS_PER_EPOCH
        record = {"seed": seed, "epoch": epoch, "segment": segment, "step": step,
                  "steps_per_epoch": STEPS_PER_EPOCH, "updates_this_epoch": len(epoch_losses),
                  **mean_losses, "epoch_elapsed_seconds": time.perf_counter() - epoch_start,
                  "elapsed_seconds_this_process": time.perf_counter() - run_start,
                  "device": str(device)}
        payload = pack_checkpoint(seed, epoch, segment, net, optimizer, identity, record)
        latest_sha, milestone_sha = write_recovery(folder, payload, epoch in MILESTONES)
        write_epoch_receipt(folder, record, latest_sha)
        append_epoch_if_missing(folder, record, latest_sha)
        if milestone_sha is not None:
            atomic_json(folder / f"epoch_{epoch:03d}.json", {"seed": seed, "epoch": epoch, "segment": segment,
                         "checkpoint_sha256": milestone_sha, "dataset_sha256": dataset_sha,
                         "config_sha256": config_sha, "source_sha256": sources,
                         "training_metrics": mean_losses, "updated_utc": now()})
        atomic_json(BASE / "STATUS.json", {"stage": "TRAINING", "seed": seed, "epoch": epoch,
                     "epochs": TOTAL_EPOCHS, "checkpoint_sha256": latest_sha,
                     "milestone_sha256": milestone_sha, "updated_utc": now()})
        if milestone_sha is not None:
            atomic_json(BASE / "STATUS.json", {"stage": "DEV_EVALUATION", "seed": seed,
                         "epoch": epoch, "epochs": TOTAL_EPOCHS,
                         "checkpoint_sha256": milestone_sha, "updated_utc": now()})
            evaluate_checkpoint(seed, epoch)
            refresh_convergence()
            atomic_json(BASE / "STATUS.json", {"stage": "TRAINING", "seed": seed,
                         "epoch": epoch, "epochs": TOTAL_EPOCHS,
                         "checkpoint_sha256": latest_sha,
                         "milestone_sha256": milestone_sha, "updated_utc": now()})

    milestone_hashes = {}
    for epoch in MILESTONES:
        cp, receipt = folder / f"epoch_{epoch:03d}.pt", folder / f"epoch_{epoch:03d}.json"
        if seed in OLD_SEEDS and epoch in (32, 64, 96):
            expected = verified_old[(seed, epoch)]["sha256"]
            saved = read(receipt)
            if sha(cp) != expected or saved.get("checkpoint_sha256") != expected:
                raise RuntimeError(f"ancestral milestone checkpoint failed final validation: {cp}")
            milestone_hashes[str(epoch)] = expected
        else:
            valid = validate_candidate(cp, receipt, identity)
            if valid is None:
                raise RuntimeError(f"milestone checkpoint failed final validation: {cp}")
            milestone_hashes[str(epoch)] = valid[1]
        dev_result = folder / f"dev_epoch_{epoch:03d}_v4.json"
        if not dev_result.is_file():
            evaluate_checkpoint(seed, epoch)
    result = {"seed": seed, "identity": identity, "milestones": milestone_hashes,
              "training_epoch_record_sha256": {
                  path.stem.removeprefix("training_epoch_"): sha(path)
                  for path in sorted(folder.glob("training_epoch_*_segment_*.json"))},
              "training_events_sha256": sha(folder / "events.jsonl"),
              "recovery_archives_sha256": {str(path.relative_to(folder)).replace("\\", "/"): sha(path)
                   for path in sorted((folder / "recovery").glob("*.pt"))},
              "completed_utc": now(), "device": str(device),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "torch": torch.__version__, "python": platform.python_version(),
              "steps_per_epoch": STEPS_PER_EPOCH, "total_updates": TOTAL_EPOCHS*STEPS_PER_EPOCH}
    atomic_json(complete_path, result)
    return result


def acquire_lock():
    return ProcessLock(BASE / "PROCESS.lock").acquire()


def snapshot_inputs(cfg):
    """Keep byte-verified local data and exact model/source definitions."""
    manifest = {}
    paths = dict(cfg["source_sha256"])
    for role in ("train", "dev"):
        for name in (f"data/paper2_1103/{role}.npz", f"data/paper2_1103/{role}.receipt.json"):
            if (ROOT / name).is_file():
                paths[name] = sha(ROOT / name)
    for name, digest in paths.items():
        source, target = ROOT / name, BASE / "input_snapshot" / name
        if sha(source) != digest:
            raise RuntimeError(f"source/data changed before snapshot: {name}")
        if not target.exists():
            atomic_copy(source, target)
        if sha(target) != digest:
            raise RuntimeError(f"immutable input snapshot mismatch: {name}")
        manifest[name] = digest
    # Dataset generation/identity receipts are snapshotted with the arrays;
    # no augmentation or additional training examples are introduced.
    arch_path = BASE / "ARCHITECTURE.json"
    # Avoid consuming the training RNG while capturing the module definition.
    with torch.random.fork_rng():
        model_arch = architecture(CorrectionNet())
    if arch_path.exists() and read(arch_path) != model_arch:
        raise RuntimeError("model architecture snapshot changed")
    if not arch_path.exists():
        atomic_json(arch_path, model_arch)
    manifest["ARCHITECTURE.json"] = sha(arch_path)
    path = BASE / "INPUT_SNAPSHOT.json"
    value = {"sha256": manifest, "config_sha256": sha(BASE / "CONFIG.json")}
    if path.exists() and read(path) != value:
        raise RuntimeError("input snapshot identity changed")
    if not path.exists():
        atomic_json(path, value)


def prepare() -> tuple[dict, dict[int, dict], Path]:
    path, data_receipt = generate("train")
    data_identity = verify(path)
    dataset_sha = sha(path)
    old_complete, verified = verify_old_run(dataset_sha)
    sources = source_hashes()
    old_history_sha256 = {
        str(seed): sha(OLD_BASE / f"seed_{seed}" / "history.json")
        for seed in OLD_SEEDS
    }
    cfg = config_for(dataset_sha, sha(data_receipt), sha(OLD_BASE / "TRAINING_COMPLETE.json"),
                     sources, old_history_sha256)
    config_path = BASE / "CONFIG.json"
    if config_path.exists():
        if read(config_path) != cfg:
            raise RuntimeError("frozen 256-epoch run configuration or source identity changed")
    else:
        atomic_json(config_path, cfg)
    config_sha = sha(config_path)
    manifest_path = BASE / "RUN_MANIFEST.json"
    manifest = {"run_id": RUN_ID, "created_utc": now(), "config_sha256": config_sha,
                "dataset_sha256": dataset_sha, "dataset_identity": data_identity,
                "old_training_complete_sha256": sha(OLD_BASE / "TRAINING_COMPLETE.json"),
                "old_seeds": list(OLD_SEEDS), "new_seeds": list(NEW_SEEDS),
                "checkpoint_policy": "immutable full recovery every epoch; latest copy; immutable 32-epoch milestones",
                "network": "CorrectionNet / B4, same implementation as verified epoch-128 run"}
    if manifest_path.exists():
        old = read(manifest_path)
        for key in ("config_sha256", "dataset_sha256", "old_training_complete_sha256", "old_seeds", "new_seeds"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"frozen run manifest identity changed: {key}")
    else:
        atomic_json(manifest_path, manifest)
    snapshot_inputs(cfg)
    return cfg, verified, config_path


def run() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock()
    try:
        cfg, verified_old, config_path = prepare()
        data_path = ROOT / "data" / "paper2_1103" / "train.npz"
        arrays = load_arrays(data_path)
        dataset_sha, config_sha = sha(data_path), sha(config_path)
        sources = cfg["source_sha256"]
        results = []
        for seed in SEEDS:
            results.append(train_one(seed, arrays, dataset_sha, config_sha, sources, verified_old))
        receipt = {"run_id": RUN_ID, "seeds": list(SEEDS), "config_sha256": config_sha,
                   "dataset_sha256": dataset_sha,
                   "completed_utc": now(), "runs": results}
        atomic_json(BASE / "TRAINING_COMPLETE.json", receipt)
        atomic_json(BASE / "STATUS.json", {"stage": "TRAINING_COMPLETE", "updated_utc": now()})
        from tools.aggregate_convergence_256_8seed import main as aggregate_convergence
        aggregate_convergence()
        atomic_json(BASE / "STATUS.json", {"stage": "PIPELINE_COMPLETE",
                     "convergence_manifest_sha256": sha(
                         ROOT / "evidence" / "outputs" / "CONVERGENCE_256_8SEED_20261114" / "MANIFEST.json"),
                     "updated_utc": now()})
    except BaseException as exc:
        completed_training = (BASE / "TRAINING_COMPLETE.json").is_file()
        atomic_json(BASE / f"FAILURE_{time.time_ns()}.json",
                    {"error": repr(exc), "traceback": traceback.format_exc(), "updated_utc": now()})
        atomic_json(BASE / "STATUS.json", {"stage": "POSTPROCESSING_FAILED" if completed_training
                     else "FAILED_OR_INTERRUPTED", "training_complete": completed_training,
                     "error": repr(exc),
                     "updated_utc": now()})
        raise
    finally:
        lock.release()


def recovery_self_test() -> None:
    """Exercise checksum-gated latest recovery and 32-epoch fallback without training."""
    with tempfile.TemporaryDirectory(prefix="paper2_256_recovery_") as tmp:
        folder = Path(tmp)
        identity = checkpoint_identity(53, "data", "config", {"source": "sha"})
        baseline = {"format": CHECKPOINT_FORMAT, "identity": identity, "epoch": 32,
                    "seed": 53, "segment": 0, "state": torch.tensor([1.0])}
        baseline_sha = atomic_torch_save(baseline, folder / "epoch_032.pt")
        atomic_json(folder / "epoch_032.json", {"epoch": 32, "checkpoint_sha256": baseline_sha})
        latest = {**baseline, "epoch": 47, "state": torch.tensor([2.0])}
        latest_sha = atomic_torch_save(latest, folder / "latest.pt")
        atomic_json(folder / "latest.json", {"epoch": 47, "checkpoint_sha256": latest_sha})
        selected = choose_resume(folder, identity)
        assert selected and selected[1]["epoch"] == 47
        (folder / "latest.pt").write_bytes(b"deliberately damaged")
        selected = choose_resume(folder, identity)
        assert selected and selected[1]["epoch"] == 32 and selected[2] == baseline_sha
    print("recovery self-test passed: valid latest selected; corrupt latest fell back to verified epoch 032")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-self-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.recovery_self_test:
        recovery_self_test()
    elif args.prepare_only:
        BASE.mkdir(parents=True, exist_ok=True)
        with ProcessLock(BASE / "PROCESS.lock"):
            prepare()
    else:
        run()


if __name__ == "__main__":
    main()
