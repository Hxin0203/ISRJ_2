"""Continue the eight completed Hybrid runs in separate 256-to-512 directories.

The original trainer, model and objective remain hash-bound dependencies. Parent
endpoints are archived with optimizer/RNG state before any continuation update.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from paper2.io import ROOT, now, read
from tools.stable_hash import sha
from tools import server_gpu_admission as admission
from tools import vp_1103_recovp_hybrid_train as trainer
from tools.durable_io import ProcessLock, atomic_copy, atomic_json


SEEDS = (17, 29, 41, 53, 67, 79, 97, 109)
PARENT_EPOCH = 256
FINAL_EPOCH = 512
CONTINUATION_LR = 2e-6
PIPELINE = "recovp_hybrid_continue512_20261127"
STATE = ROOT / "evidence" / "outputs" / PIPELINE
ARCHIVE = ROOT / "backups" / "recovp_hybrid_256_before512_20261127"
EXTRA_SOURCES = (
    "tools/stable_hash.py",
    "tools/recovp_hybrid_continue512.py",
    "tools/run_recovp_hybrid_continue512.py",
    "tools/start_recovp_hybrid_continue512.ps1",
    "tools/server_gpu_admission.py",
)


def paths(seed: int, continuation: bool = False) -> tuple[Path, Path, Path]:
    if seed not in SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    run = f"recovp_hybrid_{'512_' if continuation else ''}seed{seed}"
    base = ROOT / "outputs" / run
    return base, ROOT / "evidence" / "outputs" / run, base / f"seed_{seed}"


def parent_ready(seed: int) -> bool:
    _, evidence, folder = paths(seed)
    targets = (evidence / "TRAINING_COMPLETE.json", folder / "TRAINING_COMPLETE.json")
    if not all(path.is_file() for path in targets):
        return False
    result = read(targets[0])
    return (sha(targets[0]) == sha(targets[1]) and result.get("seed") == seed
            and result.get("epochs") == PARENT_EPOCH
            and len(result.get("recovery_sha256", {})) == PARENT_EPOCH + 1)


def copy_verified(source: Path, destination: Path) -> str:
    digest = sha(source)
    if destination.exists():
        if sha(destination) != digest:
            raise RuntimeError(f"immutable archive differs: {destination}")
    else:
        atomic_copy(source, destination)
    if sha(destination) != digest:
        raise RuntimeError(f"archive copy verification failed: {destination}")
    return digest


def audit_parent(seed: int) -> tuple[dict, dict, dict]:
    """Read every parent recovery SHA; load and validate the full final state."""
    admission.require_server_identity()
    if not parent_ready(seed):
        raise RuntimeError(f"seed {seed} has not completed 256 epochs")
    base, evidence, folder = paths(seed)
    config = read(base / "CONFIG.json")
    complete = read(folder / "TRAINING_COMPLETE.json")
    preflight_path = evidence / "PREFLIGHT.json"
    if config.get("seed") != seed or config.get("epochs") != PARENT_EPOCH:
        raise RuntimeError("parent config seed/epoch mismatch")
    if sha(base / "CONFIG.json") != sha(evidence / "CONFIG.json"):
        raise RuntimeError("parent config mirrors differ")
    for rel, digest in config["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"parent frozen source mismatch: {rel}")
    data = trainer.data_hashes()
    if data != config["data_sha256"]:
        raise RuntimeError("parent data binding mismatch")
    trainer.SEED = seed
    trainer.PREFLIGHT = preflight_path
    trainer.validate_preflight(config["source_sha256"], data)
    identity = {"run": config["run_id"], "seed": seed,
                "config_sha256": sha(base / "CONFIG.json"),
                "train_sha256": data["train.npz"],
                "source_sha256": config["source_sha256"],
                "preflight_sha256": sha(preflight_path)}
    if complete["identity"] != identity or config["preflight_sha256"] != sha(preflight_path):
        raise RuntimeError("parent complete/config/preflight identity mismatch")
    physical = config["execution"]["physical_gpu_index"]
    if config["execution"]["gpu_uuid"] != admission.GPU_UUIDS[physical]:
        raise RuntimeError("parent GPU identity mismatch")
    expected_epochs = set(range(PARENT_EPOCH + 1))
    recovery_paths = list((folder / "recovery").glob("epoch_*.pt"))
    if {int(path.stem.split("_")[-1]) for path in recovery_paths} != expected_epochs:
        raise RuntimeError("parent recovery sequence has a gap or unexpected epoch")
    for epoch in sorted(expected_epochs):
        checkpoint = folder / "recovery" / f"epoch_{epoch:03d}.pt"
        digest = sha(checkpoint)
        receipt = read(checkpoint.with_suffix(".json"))
        if (digest != complete["recovery_sha256"][str(epoch)]
                or receipt.get("checkpoint_sha256") != digest
                or receipt.get("epoch") != epoch or receipt.get("identity") != identity):
            raise RuntimeError(f"parent recovery identity/SHA mismatch: {checkpoint}")
        if epoch in range(32, PARENT_EPOCH + 1, 32):
            if sha(folder / f"epoch_{epoch:03d}.pt") != digest:
                raise RuntimeError(f"parent milestone differs at epoch {epoch}")
    endpoint = folder / "recovery" / "epoch_256.pt"
    trainer.FORMAT = f"recovp-hybrid-seed{seed}-epoch-v1"
    payload = trainer.validate_checkpoint(endpoint, identity)
    if payload is None or payload.get("seed") != seed:
        raise RuntimeError("parent final checkpoint failed validation")
    if sha(folder / "latest.pt") != sha(endpoint):
        raise RuntimeError("parent latest differs from final checkpoint")
    if not payload["optimizer"].get("state"):
        raise RuntimeError("parent optimizer state is missing")
    if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(payload["rng"]):
        raise RuntimeError("parent full RNG state is missing")
    if len(payload["rng"]["torch_cuda"]) != 1:
        raise RuntimeError("parent must have exactly one visible CUDA RNG state")

    archive = ARCHIVE / f"seed_{seed}"
    files = {
        "epoch_256.pt": endpoint, "epoch_256.json": endpoint.with_suffix(".json"),
        "CONFIG.json": base / "CONFIG.json", "PREFLIGHT.json": preflight_path,
        "TRAINING_COMPLETE.json": folder / "TRAINING_COMPLETE.json",
    }
    for rel in config["source_sha256"]:
        files[f"source/{rel}"] = ROOT / rel
    files["source/tools/vp_1103_recovp_hybrid_preflight.py"] = (
        ROOT / "tools/vp_1103_recovp_hybrid_preflight.py")
    archived = {name: copy_verified(path, archive / name) for name, path in files.items()}
    manifest = {"seed": seed, "parent_identity": identity,
                "parent_checkpoint": str(endpoint), "parent_sha256": sha(endpoint),
                "archive": str(archive), "archived_files_sha256": archived,
                "retained_original_recovery_sha256": complete["recovery_sha256"],
                "retained_original_milestones": complete["milestones"],
                "optimizer_and_rng_present": True}
    target = archive / "MANIFEST.json"
    if target.exists() and read(target) != manifest:
        raise RuntimeError("256-epoch archive manifest changed")
    if not target.exists():
        atomic_json(target, manifest)
    return config, payload, manifest


def freeze_extension(seed: int, gpu: int, parent: dict, manifest: dict,
                     precision: dict) -> tuple[dict, dict]:
    base, evidence, _ = paths(seed, True)
    sources = {**parent["source_sha256"],
               **{rel: sha(ROOT / rel) for rel in EXTRA_SOURCES}}
    config = {**copy.deepcopy(parent), "run_id": base.name, "epochs": FINAL_EPOCH,
              "scope": "EXISTING_TRAIN_CONTINUATION_257_TO_512",
              "milestones": list(range(288, FINAL_EPOCH + 1, 32)),
              "source_sha256": sources, "parent": manifest,
              "lr_schedule": "epochs 257..512 constant 2e-6; no warmup or reset",
              "continuation_lr": CONTINUATION_LR, "precision": precision,
              "execution": {"hostname": admission.SERVER_HOST, "root": str(ROOT),
                            "physical_gpu_index": gpu, "gpu_uuid": admission.GPU_UUIDS[gpu],
                            "visible_devices": str(gpu), "local_training": False,
                            "admission_policy": admission.ADMISSION_POLICY},
              "recovery": "new immutable epoch_256..512 plus every-32 milestone; original 0..256 retained"}
    config_path = base / "CONFIG.json"
    if config_path.exists() and read(config_path) != config:
        raise RuntimeError("continuation config changed, including frozen GPU binding")
    if not config_path.exists():
        atomic_json(config_path, config)
    copy_verified(config_path, evidence / "CONFIG.json")
    identity = {"run": base.name, "seed": seed, "config_sha256": sha(config_path),
                "train_sha256": parent["data_sha256"]["train.npz"],
                "source_sha256": sources, "preflight_sha256": parent["preflight_sha256"],
                "parent_checkpoint_sha256": manifest["parent_sha256"]}
    return config, identity


def restore_state(model, optimizer, payload: dict) -> None:
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    trainer.restore_rng(payload["rng"])


def repair_checkpoint_receipts(folder: Path, identity: dict) -> None:
    """Finish an atomic checkpoint commit interrupted before its JSON receipt."""
    for path in sorted((folder / "recovery").glob("epoch_*.pt")):
        receipt_path = path.with_suffix(".json")
        if receipt_path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(path.stem.split("_")[-1])
        if (payload.get("identity") != identity or payload.get("format") != trainer.FORMAT
                or payload.get("epoch") != epoch or not PARENT_EPOCH <= epoch <= FINAL_EPOCH
                or payload.get("global_step") != epoch * trainer.STEPS_PER_EPOCH
                or not {"model", "optimizer", "rng", "epoch_record"}.issubset(payload)):
            raise RuntimeError(f"unreceipted checkpoint identity is invalid: {path}")
        atomic_json(receipt_path, {"epoch": epoch, "identity": identity,
                                  "checkpoint_sha256": sha(path),
                                  "recovered_receipt_utc": now()})


def train(seed: int, gpu: int, verify_only: bool = False) -> dict:
    admission.require_server_identity()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("visible physical GPU differs from dispatch")
    base, evidence, folder = paths(seed, True)
    with ProcessLock(folder / "PROCESS.lock"):
        atomic_json(base / "STATUS.json", {"stage": "VERIFYING_PARENT", "seed": seed,
                                          "pid": os.getpid(), "updated_utc": now()})
        parent, parent_payload, manifest = audit_parent(seed)
        precision = trainer.configure_precision()
        torch.set_num_threads(4)
        # Admission is rechecked after the CPU archive audit, immediately before CUDA.
        admission.require_visible_gpu()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one visible server CUDA device is required")
        model = trainer.ReCoVPHybrid().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONTINUATION_LR,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        restore_state(model, optimizer, parent_payload)
        if verify_only:
            for name, value in model.state_dict().items():
                if not torch.equal(value.cpu(), parent_payload["model"][name]):
                    raise RuntimeError(f"restored model tensor differs: {name}")
            for parameter_id, state in optimizer.state_dict()["state"].items():
                expected_state = parent_payload["optimizer"]["state"][parameter_id]
                for name, value in state.items():
                    if not torch.equal(value.cpu(), expected_state[name].cpu()):
                        raise RuntimeError(f"restored optimizer slot differs: {parameter_id}/{name}")
                if int(state["step"].item()) != PARENT_EPOCH * trainer.STEPS_PER_EPOCH:
                    raise RuntimeError("restored Adam step counter is not 12288")
            if (not torch.equal(torch.get_rng_state(), parent_payload["rng"]["torch_cpu"])
                    or not torch.equal(torch.cuda.get_rng_state(), parent_payload["rng"]["torch_cuda"][0])):
                raise RuntimeError("restored CPU/CUDA RNG state differs")
            result = {"stage": "CONTINUATION_RESTORE_VERIFIED", "seed": seed,
                      "parent_sha256": manifest["parent_sha256"],
                      "optimizer_slots": len(optimizer.state), "rng_restored": True,
                      "model_restored": True, "gpu": gpu, "updated_utc": now()}
            atomic_json(base / "STATUS.json", result)
            return result
        _, identity = freeze_extension(seed, gpu, parent, manifest, precision)
        trainer.FORMAT = f"recovp-hybrid-seed{seed}-continue512-v1"
        trainer.MILESTONES = tuple(range(288, FINAL_EPOCH + 1, 32))
        trainer.learning_rate = lambda _epoch, _batch: CONTINUATION_LR
        repair_checkpoint_receipts(folder, identity)
        recovered = trainer.choose_resume(folder, identity)
        if recovered["epoch"] < 0:
            recovered = {**parent_payload, "format": trainer.FORMAT, "identity": identity,
                         "created_utc": now(), "parent_checkpoint_sha256": manifest["parent_sha256"]}
        restore_state(model, optimizer, recovered)
        trainer.publish_checkpoint(folder, recovered)
        arrays = trainer.load_arrays(trainer.DATA / "train.npz")
        if len(arrays["s"]) != 512 or arrays["s"].shape[-2:] != (8, 1800):
            raise RuntimeError("training data shape differs from the frozen protocol")
        start_epoch = int(recovered["epoch"])
        if not PARENT_EPOCH <= start_epoch <= FINAL_EPOCH:
            raise RuntimeError("continuation recovery epoch is outside the registered interval")
        del parent_payload, recovered
        for epoch in range(start_epoch + 1, FINAL_EPOCH + 1):
            started = time.perf_counter()
            model.train()
            rows = []
            for batch_index, ids in enumerate(trainer.batches(512, epoch, seed)):
                for group in optimizer.param_groups:
                    group["lr"] = CONTINUATION_LR
                row = trainer.optimizer_step(model, optimizer, arrays, ids, epoch,
                                             batch_index, torch.device("cuda"))
                rows.append(row)
                if batch_index == 0 or (batch_index + 1) % 12 == 0:
                    atomic_json(base / "STATUS.json", {
                        "stage": "TRAINING", "seed": seed, "epoch": epoch,
                        "epochs": FINAL_EPOCH, "last_committed_epoch": epoch - 1,
                        "update": batch_index + 1, "metrics": row,
                        "gpu": gpu, "pid": os.getpid(), "updated_utc": now()})
            if len(rows) != trainer.STEPS_PER_EPOCH:
                raise RuntimeError("unexpected number of updates")
            torch.cuda.synchronize()
            metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            record = {"run": base.name, "seed": seed, "epoch": epoch,
                      "global_step": epoch * trainer.STEPS_PER_EPOCH,
                      "updates_this_epoch": trainer.STEPS_PER_EPOCH, **metrics,
                      "epoch_elapsed_seconds": time.perf_counter() - started,
                      "updated_utc": now()}
            payload = {"format": trainer.FORMAT, "identity": identity, "seed": seed,
                       "epoch": epoch, "global_step": epoch * trainer.STEPS_PER_EPOCH,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": trainer.rng_snapshot(), "epoch_record": record,
                       "architecture": model.parameter_report(), "created_utc": now()}
            digest = trainer.publish_checkpoint(folder, payload)
            atomic_json(base / "STATUS.json", {
                "stage": "EPOCH_SAVED", "seed": seed, "epoch": epoch,
                "checkpoint_sha256": digest, "gpu": gpu, "pid": os.getpid(),
                "metrics": metrics, "updated_utc": now()})
            print(f"Hybrid seed={seed} epoch={epoch}/512 loss={metrics['total']:.7g}", flush=True)
        recovery = {}
        for epoch in range(PARENT_EPOCH, FINAL_EPOCH + 1):
            path = folder / "recovery" / f"epoch_{epoch:03d}.pt"
            if trainer.validate_checkpoint(path, identity) is None:
                raise RuntimeError(f"continuation recovery audit failed: {path}")
            recovery[str(epoch)] = sha(path)
            if epoch in trainer.MILESTONES and sha(folder / f"epoch_{epoch:03d}.pt") != recovery[str(epoch)]:
                raise RuntimeError(f"continuation milestone differs: {epoch}")
        result = {"stage": "TRAINING_COMPLETE", "identity": identity, "seed": seed,
                  "epochs": FINAL_EPOCH, "additional_epochs": 256,
                  "updates": FINAL_EPOCH * trainer.STEPS_PER_EPOCH,
                  "parent": manifest, "recovery_sha256": recovery,
                  "milestones": {str(e): recovery[str(e)] for e in trainer.MILESTONES},
                  "completed_utc": now()}
        atomic_json(folder / "TRAINING_COMPLETE.json", result)
        atomic_json(evidence / "TRAINING_COMPLETE.json", result)
        atomic_json(base / "STATUS.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        _, _, result = audit_parent(args.seed)
        atomic_json(STATE / f"PARENT_AUDIT_seed{args.seed}.json", result)
        print({"stage": "PARENT_ARCHIVED", "seed": args.seed,
               "parent_sha256": result["parent_sha256"]}, flush=True)
        return
    if args.gpu_index is None:
        parser.error("--gpu-index is required for continuation or restore verification")
    try:
        result = train(args.seed, args.gpu_index, args.verify_only)
    except admission.NoGpuAvailable as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
    if args.verify_only:
        atomic_json(STATE / f"RESTORE_VERIFIED_seed{args.seed}.json", result)
    print({"stage": result["stage"], "seed": args.seed}, flush=True)


if __name__ == "__main__":
    main()
