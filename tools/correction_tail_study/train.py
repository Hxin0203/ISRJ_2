"""Crash-recoverable original-CorrectionNet loss contrast on server B."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read, sha
from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import (atomic_torch_save, seed_rng, rng_snapshot,
                                  restore_rng, optimizer_for, learning_rate)
from tools.durable_io import ProcessLock, atomic_json, atomic_copy, replace_retry
from tools.unet_ablation.protocol import DATA as ABLATION_DATA
from .objective import study_objective
from .reference import reference_binding
from .runtime import (require_server, configure_precision,
                                         resource_sample, user_control,
                                         begin_budget, end_budget)
from paper2.network import CorrectionNet as StudyModel
from .runtime import admit, wait_dependency
from .preflight import preflight

from .settings import RUN, ARM, GPU_INDEX, GPU_UUID
BASE = ROOT / "outputs" / RUN
EVIDENCE = ROOT / "evidence" / "outputs" / RUN
DATA = ABLATION_DATA
SEEDS = (17, 29, 41)
MILESTONES = tuple(range(32, 257, 32))
STEPS = 48
FORMAT = "correction-tail-study-epoch-v1"


def local_sources() -> dict[str, str]:
    paths = [*sorted((ROOT / "paper2").rglob("*.py")),
             *sorted((ROOT / "tools/unet_ablation").glob("*.py")),
             *sorted((ROOT / "tools/correction_tail_study").glob("*.py")),
             ROOT / "tools/unet_tail_loss/objective.py",
             ROOT / "tools/durable_io.py", ROOT / "tools/start_correction_tail_study.ps1",
             ROOT / "tools/deploy_correction_tail_study.ps1",
             ROOT / "evidence/outputs/CORRECTIONNET_TAIL_STUDY_PLAN_20261117.md"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def prepare() -> dict:
    require_server()
    configure_precision()
    torch.set_num_threads(4)
    BASE.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    data = {f"{role}{ext}": sha(DATA / f"{role}{ext}")
            for role in ("train", "dev") for ext in (".npz", ".receipt.json")}
    for role in ("train", "dev"):
        receipt = read(DATA / f"{role}.receipt.json")
        if receipt["sha256"] != data[f"{role}.npz"]:
            raise RuntimeError(f"{role} receipt mismatch")
    baseline = {}
    for seed in SEEDS:
        baseline[str(seed)] = reference_binding(seed, data['train.npz'])
    sources = local_sources()
    preflight_record = read(BASE / "NUMERICAL_PREFLIGHT.json")
    if preflight_record["source_sha256"] != sources or preflight_record["stage"] != "NUMERICAL_PREFLIGHT_PASSED":
        raise RuntimeError("Exact-source preflight required")
    cfg = {
        "run_id": RUN, "scope": "DEV_SCREEN_ONLY", "seeds": list(SEEDS),
        "epochs": 256, "updates_per_seed": 12288,
        "milestones": list(MILESTONES), "records_per_epoch": 1536,
        "batch_records": 32, "microbatch_records": 8, "steps_per_epoch": STEPS,
        "architecture": "Original CorrectionNet; dechirp/RMS/core unchanged; 112756 registered, 59714 active, 53042 inactive parameters",
        "arm": ARM,
        "baseline": baseline, "training_data_sha256": data["train.npz"],
        "training_data_receipt_sha256": data["train.receipt.json"],
        "dev_data_sha256": data["dev.npz"], "dev_receipt_sha256": data["dev.receipt.json"],
        "source_sha256": sources,
        "optimizer": {"class": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0, "clip_grad_norm": 5},
        "lr_schedule": "identical to frozen 256-epoch U-Net screen: one-epoch warmup; cosine 1e-3 to 2e-5 by 128 and to 2e-6 by 256",
        "loss": {"baseline": "endpoint + 0.1 pulse-compression + target-damage",
                 "tail_weight": 0.05 if ARM == 'tail' else 0.0,
                 "tail": "exact frozen U-Net smooth-top8 matched-filter RD surrogate; training-noise mean RD power normalization; original arm also records unweighted tail",
                 "limitation": "risk surrogate; does not calibrate or guarantee CFAR false-alarm probability"},
        "preflight_sha256": sha(BASE / "NUMERICAL_PREFLIGHT.json"),
        "comparison": ("Historical Full reference is descriptive only; this fresh original-loss arm is the training control" if ARM == 'original' else "Fresh original-loss CorrectionNet control; same initial tensors, seeds, data, microbatch8, optimizer, LR, updates and precision; only tail weight changes"),
        "recovery": "atomic immutable per-epoch checkpoints, every-32-epoch milestone, optimizer and RNG state; preserve all records",
        "execution": {"host": "rsdr-b", "root": "D:/code/Xin/ISRJ_222",
                      "cuda_visible_devices": GPU_INDEX, "physical_gpu_uuid": GPU_UUID, "local_training": False},
        "confirmation": "new independent parent set required after arm screening; DEV cannot confirm",
        "prepared_utc": now(),
    }
    config = BASE / "CONFIG.json"
    if config.exists() and read(config) != cfg:
        # Exclude only the volatile preparation timestamp from an identical retry.
        old = read(config)
        if {k: v for k, v in old.items() if k != "prepared_utc"} != {k: v for k, v in cfg.items() if k != "prepared_utc"}:
            raise RuntimeError("frozen screening config changed; create a new run identity")
        cfg = old
    else:
        atomic_json(config, cfg)
    atomic_json(EVIDENCE / "CONFIG.json", cfg)
    atomic_json(BASE / "STATUS.json", {"stage": "PREPARED", "config_sha256": sha(config), "updated_utc": now()})
    atomic_json(EVIDENCE / "STATUS.json", read(BASE / "STATUS.json"))
    return cfg


def validate_config() -> dict:
    cfg = read(BASE / "CONFIG.json")
    for name, digest in cfg["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"source drift: {name}")
    if sha(BASE / "NUMERICAL_PREFLIGHT.json") != cfg["preflight_sha256"]:
        raise RuntimeError("preflight identity changed")
    for name, digest in (("train.npz", cfg["training_data_sha256"]),
                         ("train.receipt.json", cfg["training_data_receipt_sha256"]),
                         ("dev.npz", cfg["dev_data_sha256"]),
                         ("dev.receipt.json", cfg["dev_receipt_sha256"])):
        if sha(DATA / name) != digest:
            raise RuntimeError(f"frozen data changed: {name}")
    return cfg


def valid_checkpoint(path, expected):
    try:
        receipt = read(path.with_suffix(".json"))
        if receipt["checkpoint_sha256"] != sha(path):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["format"] != FORMAT or payload["identity"] != expected or payload["epoch"] != receipt["epoch"]:
            return None
        if payload["global_step"] != payload["epoch"] * 48:
            return None
        return payload
    except Exception:
        return None


def choose_resume(folder, expected):
    candidates = []
    for path in [*folder.glob("recovery/epoch_*.pt"), folder / "latest.pt", *folder.glob("epoch_*.pt")]:
        try:
            candidates.append((int(read(path.with_suffix(".json"))["epoch"]), str(path), path))
        except (OSError, ValueError, KeyError):
            pass
    for _, _, path in sorted(candidates, reverse=True):
        value = valid_checkpoint(path, expected)
        if value is not None:
            return value
    if any((folder / "recovery").glob("*.pt")) or (folder / "latest.pt").exists():
        raise RuntimeError("no valid recovery checkpoint; preserve this run for inspection")
    return None


def preserve_invalid(path):
    if path.exists():
        replace_retry(path, path.with_name(path.name + f".invalid_{time.time_ns()}"))


def publish(folder, payload):
    epoch = payload["epoch"]
    archive = folder / "recovery" / f"epoch_{epoch:03d}.pt"
    existing = valid_checkpoint(archive, payload["identity"])
    if existing is not None:
        if existing["epoch_record"] != payload["epoch_record"] or any(
                not torch.equal(v.cpu(), existing["model"][k].cpu()) for k, v in payload["model"].items()):
            raise RuntimeError("immutable recovery checkpoint changed")
        digest = sha(archive)
    else:
        preserve_invalid(archive)
        preserve_invalid(archive.with_suffix(".json"))
        digest = atomic_torch_save(payload, archive)
        atomic_json(archive.with_suffix(".json"), {"epoch": epoch, "identity": payload["identity"], "checkpoint_sha256": digest})
    targets = [folder / "latest.pt"]
    if epoch in MILESTONES:
        targets.append(folder / f"epoch_{epoch:03d}.pt")
    for path in targets:
        if path.name != "latest.pt" and path.exists() and sha(path) != digest:
            raise RuntimeError(f"immutable milestone changed: {path}")
        atomic_copy(archive, path)
        atomic_json(path.with_suffix(".json"), {"epoch": epoch, "identity": payload["identity"], "checkpoint_sha256": digest})
    record = payload["epoch_record"]
    if record is not None:
        path = folder / f"training_epoch_{epoch:03d}.json"
        value = {**record, "checkpoint_sha256": digest}
        if path.exists() and read(path) != value:
            raise RuntimeError("immutable epoch metrics changed")
        atomic_json(path, value)
    return digest



def train_seed(seed: int) -> dict:
    require_server()
    cfg = validate_config()
    folder = BASE / "arm" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    expected = {"run": RUN, "seed": seed, "config_sha256": sha(BASE / "CONFIG.json"),
                "data_sha256": cfg["training_data_sha256"], "source_sha256": cfg["source_sha256"]}
    with ProcessLock(folder / "PROCESS.lock"):
        complete = folder / "TRAINING_COMPLETE.json"
        if complete.exists():
            result = read(complete)
            if result["identity"] != expected:
                raise RuntimeError("completed run identity mismatch")
            for epoch, digest in result["recovery_sha256"].items():
                cp = folder / "recovery" / f"epoch_{int(epoch):03d}.pt"
                if sha(cp) != digest or read(cp.with_suffix(".json"))["checkpoint_sha256"] != digest:
                    raise RuntimeError(f"completed recovery changed: {cp}")
            for epoch, digest in result["milestones"].items():
                if sha(folder / f"epoch_{int(epoch):03d}.pt") != digest:
                    raise RuntimeError("completed milestone changed")
            return result
        configure_precision()
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed_rng(seed)
        model = StudyModel().cuda()
        optimizer = optimizer_for(model)
        payload = choose_resume(folder, expected)
        if payload is not None:
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            start = int(payload["epoch"])
            publish(folder, payload)
        else:
            start = 0
            payload = {"format": FORMAT, "identity": expected, "seed": seed, "epoch": 0,
                       "global_step": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng_snapshot(), "epoch_record": None}
            publish(folder, payload)
        arrays = load_arrays(DATA / "train.npz")
        if len(arrays["s"]) != 512:
            raise RuntimeError("training parent count changed")
        for epoch in range(start + 1, 257):
            user_control(BASE)
            totals = {k: 0.0 for k in ('total', 'endpoint', 'pc', 'damage', 'tail', 'tail_peak_db_mean', 'tail_top8_db_mean')}
            started = time.perf_counter()
            model.train()
            for step, ids in enumerate(batches(len(arrays["s"]), epoch, seed), start=1):
                lr = learning_rate(epoch, step - 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                ids = np.asarray(ids)
                for lo in range(0, len(ids), 8):
                    subids = ids[lo:lo + 8]
                    batch = make_batch(arrays, subids, "cuda")
                    corr = model(batch["y"]).reshape(-1, 4, 2, 1800)
                    base = study_objective(corr, batch, len(subids))
                    total = base["total"]
                    if not torch.isfinite(total):
                        raise RuntimeError(f"non-finite objective at seed={seed}, epoch={epoch}, step={step}")
                    weight = len(subids) / len(ids)
                    (total * weight).backward()
                    vals = {k: base[k] for k in totals}
                    for key, value in vals.items():
                        totals[key] += float(value.detach()) * weight / STEPS
                    del batch, corr, base, total
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                if step % 12 == 0:
                    atomic_json(BASE / "STATUS.json", {"stage": "TRAINING", "seed": seed, "epoch": epoch,
                                "step": step, "steps": STEPS, "last_committed_epoch": epoch - 1,
                                "resource": resource_sample(), "updated_utc": now()})
            torch.cuda.synchronize()
            record = {"run": RUN, "seed": seed, "epoch": epoch, "global_step": epoch * STEPS,
                      "updates_this_epoch": STEPS, **totals,
                      "epoch_elapsed_seconds": time.perf_counter() - started}
            payload = {"format": FORMAT, "identity": expected, "seed": seed, "epoch": epoch,
                       "global_step": epoch * STEPS, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "rng": rng_snapshot(), "epoch_record": record}
            digest = publish(folder, payload)
            atomic_json(BASE / "STATUS.json", {"stage": "EPOCH_SAVED", "seed": seed, "epoch": epoch,
                        "checkpoint_sha256": digest, "metrics": totals, "updated_utc": now()})
        recovery = {}
        milestones = {}
        for epoch in range(257):
            cp = folder / "recovery" / f"epoch_{epoch:03d}.pt"
            if not cp.exists() or valid_checkpoint(cp, expected) is None:
                raise RuntimeError(f"recovery audit failed: {cp}")
            recovery[str(epoch)] = sha(cp)
            if epoch in MILESTONES:
                milestone = folder / f"epoch_{epoch:03d}.pt"
                if sha(milestone) != recovery[str(epoch)] or read(milestone.with_suffix(".json"))["checkpoint_sha256"] != recovery[str(epoch)]:
                    raise RuntimeError(f"milestone identity mismatch: {milestone}")
                milestones[str(epoch)] = recovery[str(epoch)]
        result = {"identity": expected, "seed": seed, "epochs": 256, "updates": 12288,
                  "milestones": milestones, "recovery_sha256": recovery, "completed_utc": now()}
        atomic_json(complete, result)
        atomic_json(EVIDENCE / "arm" / f"seed_{seed}" / "TRAINING_COMPLETE.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    require_server()
    if args.preflight:
        preflight(BASE)
        return
    if args.prepare:
        cfg = prepare()
        print({"stage": "PREPARED", "config_sha256": sha(BASE / "CONFIG.json"), "seeds": cfg["seeds"]}, flush=True)
        return
    if args.seed is not None:
        try:
            train_seed(args.seed)
        except BaseException as exc:
            folder = BASE / "arm" / f"seed_{args.seed}"
            atomic_json(folder / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                        "utc": now(), "pid": os.getpid()})
            raise
        return
    with ProcessLock(BASE / "CONTROLLER.lock"):
        wait_dependency(BASE)
        admit(BASE)
        preflight(BASE)
        prepare()
        budget = begin_budget(BASE)
        try:
            for seed in SEEDS:
                user_control(BASE)
                train_seed(seed)
            from .evaluate import evaluate_all
            dev_screen = evaluate_all()
            if ARM == 'tail':
                from .fourcell import summarize_four_cells
                summarize_four_cells()
            done = {"stage": "DEV_SCREEN_COMPLETE", "seeds": list(SEEDS),
                    "config_sha256": sha(BASE / "CONFIG.json"),
                    "dev_screen_sha256": sha(EVIDENCE / "DEV_SCREEN_SUMMARY.json"),
                    "dev_screen_gate": dev_screen["screening_gate"], "completed_utc": now()}
            atomic_json(BASE / "PIPELINE_STATUS.json", done)
            atomic_json(EVIDENCE / "PIPELINE_STATUS.json", done)
            if ARM == 'tail':
                from .runtime import release_dilation
                release_dilation(BASE)
        finally:
            end_budget(BASE, budget)


if __name__ == "__main__":
    main()
