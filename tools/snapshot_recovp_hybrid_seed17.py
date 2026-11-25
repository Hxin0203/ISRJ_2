"""Read-only server-B audit of the active ReCoVP-Hybrid seed-17 run.

Only LIVE_SNAPSHOT.json is written. Training/model entry points are never
imported, datasets are hashed without loading arrays, and checkpoints are
deserialized on CPU. Run as ``python -B -m tools.snapshot_recovp_hybrid_seed17``
from the repository root. No CONFIRM or historical TEST path is opened.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = Path("D:/code/Xin/ISRJ_222")
SERVER_HOST = "DESKTOP-RFFERM3"
RUN = "recovp_hybrid_seed17"
BASE = ROOT / "outputs" / RUN
FOLDER = BASE / "seed_17"
EVIDENCE = ROOT / "evidence" / "outputs" / RUN
DATA = ROOT / "data" / "paper2_1103"
DATA_NAMES = {"train.npz", "train.receipt.json", "dev.npz", "dev.receipt.json"}
GPU_UUIDS = {
    0: "GPU-f12e73b1-3bed-5a1e-bcb0-f9726dd1621d",
    1: "GPU-bb60dd3a-738e-2398-908f-7f77ac05549b",
}
FORMAT = "recovp-hybrid-seed17-epoch-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def require_server() -> None:
    if platform.node().casefold() != SERVER_HOST.casefold():
        raise RuntimeError("Snapshot is restricted to rsdr-b / DESKTOP-RFFERM3")
    if ROOT.resolve() != SERVER_ROOT.resolve():
        raise RuntimeError(f"Snapshot root must be {SERVER_ROOT}")


def snapshot() -> dict:
    require_server()
    # This process must not allocate a CUDA context while training is active.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import torch
    from tools.durable_io import atomic_json, process_alive

    torch.set_num_threads(1)
    started = now()
    errors: list[str] = []
    config = read(BASE / "CONFIG.json")
    preflight = read(EVIDENCE / "PREFLIGHT.json")
    config_sha = sha(BASE / "CONFIG.json")
    preflight_sha = sha(EVIDENCE / "PREFLIGHT.json")

    def check(condition: bool, message: str) -> bool:
        if not condition:
            errors.append(message)
        return bool(condition)

    check(config.get("run_id") == RUN and config.get("seed") == 17,
          "CONFIG run or seed mismatch")
    check(config.get("epochs") == 256 and config.get("steps_per_epoch") == 48,
          "CONFIG epoch/update protocol mismatch")
    check(config.get("scope") == "TRAIN_512_DEV_256_ONLY", "CONFIG scope mismatch")
    check(preflight.get("stage") == "PREFLIGHT_PASSED"
          and preflight.get("status") == "PASS" and preflight.get("seed") == 17
          and preflight.get("scope") == "TRAIN_AND_DEV_ONLY", "PREFLIGHT identity failed")
    check(config.get("preflight_sha256") == preflight_sha, "CONFIG PREFLIGHT SHA mismatch")
    mirror_checks = {}
    for label, first, second in (
        ("config", BASE / "CONFIG.json", EVIDENCE / "CONFIG.json"),
        ("preflight", EVIDENCE / "PREFLIGHT.json", BASE / "PREFLIGHT.json"),
    ):
        match = second.exists() and sha(first) == sha(second)
        mirror_checks[label] = check(match, f"{label} evidence/output mirror mismatch")

    source_rows, data_rows = {}, {}
    for owner, record in (("config", config), ("preflight", preflight)):
        source_bindings = record.get("source_sha256", {})
        data_bindings = record.get("data_sha256", {})
        check(bool(source_bindings), f"{owner} missing source bindings")
        check(set(data_bindings) == DATA_NAMES, f"{owner} data allowlist mismatch")
        for name, expected in source_bindings.items():
            path = ROOT / name
            safe = (not Path(name).is_absolute() and ".." not in Path(name).parts
                    and path.resolve().is_relative_to(ROOT.resolve())
                    and name.replace("\\", "/").startswith(("tools/", "paper2/"))
                    and path.suffix == ".py")
            if not check(safe, f"{owner} unsafe source binding: {name}"):
                continue
            if name not in source_rows:
                source_rows[name] = {"actual_sha256": sha(path), "bound_by": {}}
            row = source_rows[name]
            match = row["actual_sha256"] == expected
            row["bound_by"][owner] = {"expected_sha256": expected, "match": match}
            check(match, f"{owner} source SHA mismatch: {name}")
        for name, expected in data_bindings.items():
            if name not in DATA_NAMES:
                continue
            if name not in data_rows:
                data_rows[name] = {"actual_sha256": sha(DATA / name), "bound_by": {}}
            row = data_rows[name]
            match = row["actual_sha256"] == expected
            row["bound_by"][owner] = {"expected_sha256": expected, "match": match}
            check(match, f"{owner} data SHA mismatch: {name}")
    for role in ("train", "dev"):
        receipt_match = read(DATA / f"{role}.receipt.json").get("sha256") == data_rows[f"{role}.npz"]["actual_sha256"]
        data_rows[f"{role}.npz"]["dataset_receipt_match"] = check(receipt_match, f"{role} data receipt mismatch")

    identity = {
        "run": RUN, "seed": 17, "config_sha256": config_sha,
        "train_sha256": config["data_sha256"]["train.npz"],
        "source_sha256": config["source_sha256"], "preflight_sha256": preflight_sha,
    }

    def inspect_checkpoint(path: Path, expected_epoch: int | None = None,
                           mutable: bool = False) -> dict:
        # latest.pt and latest.json publish separately. Retry a moving pointer;
        # immutable recovery points are audited independently in every case.
        for _ in range(3 if mutable else 1):
            receipt_bytes = path.with_suffix(".json").read_bytes()
            receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if mutable and (receipt_bytes != path.with_suffix(".json").read_bytes()
                            or actual != receipt.get("checkpoint_sha256")):
                continue
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            epoch = int(payload.get("epoch", -1))
            checks = {
                "sha256_matches_receipt": actual == receipt.get("checkpoint_sha256"),
                "receipt_identity_matches": receipt.get("identity") == identity,
                "embedded_identity_matches": payload.get("identity") == identity,
                "format_matches": payload.get("format") == FORMAT,
                "epoch_matches": epoch == receipt.get("epoch")
                and (expected_epoch is None or epoch == expected_epoch),
                "seed_matches": payload.get("seed") == 17,
                "global_step_matches": payload.get("global_step") == epoch * 48,
                "recovery_state_present": all(k in payload for k in ("model", "optimizer", "rng")),
            }
            for label, match in checks.items():
                check(match, f"{relative(path)}: {label}")
            record = payload.get("epoch_record")
            if epoch > 0:
                row_path = FOLDER / f"training_epoch_{epoch:03d}.json"
                checks["training_receipt_present"] = row_path.exists()
                if row_path.exists():
                    row = read(row_path)
                    checks["training_receipt_matches"] = row == {**record, "checkpoint_sha256": actual}
                else:
                    checks["training_receipt_matches"] = False
                # A just-published recovery point can precede its training row.
                # Exclude it from committed progress until publication finishes.
                if checks["training_receipt_present"]:
                    check(checks["training_receipt_matches"], f"epoch {epoch} training receipt mismatch")
            return {"path": relative(path), "epoch": epoch, "checkpoint_sha256": actual,
                    "checks": checks, "verified": all(checks.values()), "epoch_record": record}
        return {"path": relative(path), "verified": False, "concurrent_publication": True}

    recovery, training_rows, pending = [], [], []
    for path in sorted((FOLDER / "recovery").glob("epoch_*.pt")):
        epoch = int(path.stem.split("_")[-1])
        if not path.with_suffix(".json").exists():
            pending.append(relative(path))
            continue
        row = inspect_checkpoint(path, epoch)
        record = row.pop("epoch_record")
        recovery.append(row)
        if row["verified"] and record:
            training_rows.append({**record, "checkpoint_sha256": row["checkpoint_sha256"]})
    valid_epochs = sorted(row["epoch"] for row in recovery if row["verified"])
    highest = max(valid_epochs, default=-1)
    missing = sorted(set(range(highest + 1)) - set(valid_epochs))
    check(not missing, f"Missing verified recovery epochs: {missing}")
    milestones = []
    recovery_by_epoch = {row["epoch"]: row for row in recovery}
    for epoch in config["milestones"]:
        path = FOLDER / f"epoch_{epoch:03d}.pt"
        if epoch <= highest:
            if check(path.exists() and path.with_suffix(".json").exists(), f"Missing milestone {epoch}"):
                row = inspect_checkpoint(path, epoch)
                row.pop("epoch_record")
                check(row["checkpoint_sha256"] == recovery_by_epoch[epoch]["checkpoint_sha256"],
                      f"Milestone {epoch} differs from recovery")
                milestones.append(row)
    latest = inspect_checkpoint(FOLDER / "latest.pt", mutable=True)
    latest.pop("epoch_record", None)
    status = read(BASE / "STATUS.json")
    status_evidence = read(EVIDENCE / "STATUS.json")
    lock_path = FOLDER / "PROCESS.lock"
    # Byte zero is a mandatory Windows byte-range lock. Read only metadata.
    with lock_path.open("rb") as stream:
        stream.seek(1)
        lock = json.loads(stream.read().decode("utf-8"))
    lock["pid_alive"] = process_alive(lock.get("pid"))

    fields = "index,uuid,name,utilization.gpu,memory.used,memory.total"
    gpu_raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}",
                                       "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for cells in csv.reader(gpu_raw.splitlines()):
        cells = [cell.strip() for cell in cells]
        index = int(cells[0])
        if index not in GPU_UUIDS:
            continue
        row = dict(zip(fields.split(","), cells))
        for field in ("index", "utilization.gpu", "memory.used", "memory.total"):
            row[field] = int(row[field])
        row["expected_uuid_matches"] = check(cells[1] == GPU_UUIDS[index], f"GPU {index} UUID mismatch")
        gpus.append(row)
    check(len(gpus) == 2, "Expected two server GPUs")

    elapsed = [r["epoch_elapsed_seconds"] for r in training_rows[-8:]
               if math.isfinite(r["epoch_elapsed_seconds"]) and r["epoch_elapsed_seconds"] > 0]
    average = statistics.mean(elapsed) if elapsed else None
    fractional_epoch = float(highest)
    if status.get("stage") == "TRAINING":
        fractional_epoch = max(fractional_epoch, status["epoch"] - 1
                               + status["update"] / status["updates_per_epoch"])
    remaining = max(0., 256 - fractional_epoch) * average if average else None
    finished = now()
    publication_pending = bool(pending or not latest["verified"]
                               or any(not row["verified"] for row in recovery))
    result = {
        "format": "recovp-hybrid-readonly-live-snapshot-v1",
        "snapshot_started_utc": started, "checked_utc": finished,
        "verification": "FAIL" if errors else "PARTIAL" if publication_pending else "PASS",
        "errors": errors, "publication_pending": publication_pending,
        "scope": "TRAIN_AND_DEV_HASHES_AND_CPU_CHECKPOINT_METADATA_ONLY",
        "execution": {"hostname": platform.node(), "root": str(ROOT),
                      "checkpoint_map_location": "cpu", "cuda_visible_devices": "",
                      "training_started_or_stopped": False, "model_inference": False},
        "snapshot_tool_sha256": sha(Path(__file__)),
        "config_sha256": config_sha, "preflight_sha256": preflight_sha,
        "metadata_mirror_checks": mirror_checks,
        "source_verification": source_rows, "data_verification": data_rows,
        "expected_checkpoint_identity": identity,
        "active_status": status, "evidence_status": status_evidence,
        "process_lock": lock, "highest_committed_saved_epoch": highest,
        "verified_recovery_count": len(valid_epochs), "missing_recovery_epochs": missing,
        "recovery": recovery, "pending_publication_paths": pending,
        "latest": latest, "saved_milestones": milestones,
        "future_milestones": [epoch for epoch in config["milestones"] if epoch > highest],
        "training_epochs": training_rows,
        "eta": {"recent_epoch_count": len(elapsed), "mean_epoch_seconds": average,
                "progress_epochs": fractional_epoch, "remaining_seconds_estimate": remaining,
                "completion_utc_estimate": (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat()
                if remaining is not None else None,
                "limitation": "Estimate from recent completed epochs; shared GPU load may change."},
        "gpus": gpus,
        "guard": {"confirmation_accessed": False, "historical_test_accessed": False,
                  "allowed_dataset_files": sorted(DATA_NAMES),
                  "only_written_path": relative(EVIDENCE / "LIVE_SNAPSHOT.json")},
    }
    atomic_json(EVIDENCE / "LIVE_SNAPSHOT.json", result)
    return result


if __name__ == "__main__":
    result = snapshot()
    print(json.dumps({key: result[key] for key in (
        "checked_utc", "verification", "errors", "highest_committed_saved_epoch",
        "verified_recovery_count", "active_status", "eta", "gpus")}, indent=2))
    raise SystemExit(0 if result["verification"] == "PASS" else 1)
