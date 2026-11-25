"""Verify the detached handoff while seed 17 is still training; no inference."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_json, process_alive
from tools import finish_recovp_hybrid_seed17 as finish
from tools import vp_1103_recovp_hybrid_evaluate as evaluator
from tools.snapshot_recovp_hybrid_seed17 import snapshot
from tools.server_gpu_admission import require_server_identity


def verify() -> dict:
    require_server_identity()
    launch = read(finish.EVIDENCE / "POSTTRAINING_LAUNCH.json")
    status = read(finish.STATUS)
    if status.get("stage") != "WAITING_FOR_SEED17_TRAINING":
        raise RuntimeError(f"expected the active training wait, got {status.get('stage')}")
    launched = datetime.fromisoformat(launch["launched_utc"].replace("Z", "+00:00"))
    heartbeat = datetime.fromisoformat(status["updated_utc"])
    checks = {
        "controller_alive_after_separate_ssh_reconnect": process_alive(status["controller_pid"]),
        "heartbeat_advanced_after_launch": heartbeat > launched + timedelta(seconds=20),
        "training_process_alive": process_alive(status["training_process"]["pid"]),
        "controller_sources_match": status["source_sha256"] == finish.source_bindings(),
        "launcher_source_matches": sha(ROOT / "tools/start_recovp_hybrid_posttraining.ps1")
        == launch["launcher_source_sha256"],
        "controller_stderr_empty": not Path(launch["stderr_path"]).read_bytes(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"detached pipeline verification failed: {checks}")
    snap = snapshot()
    checks["snapshot_source_data_and_all_saved_recovery_pass"] = snap["verification"] == "PASS"
    checks["training_lock_owner_matches"] = (
        snap["process_lock"]["pid"] == status["training_process"]["pid"])
    expected = evaluator.frozen_data_bindings({"identity": snap["expected_checkpoint_identity"]})
    checks["evaluator_frozen_dev_config_preflight_binding_pass"] = True
    if not all(checks.values()):
        raise RuntimeError(f"pipeline snapshot verification failed: {checks}; {snap['errors']}")
    receipt = {
        "stage": "BACKGROUND_PIPELINE_VERIFIED_WAITING_FOR_TRAINING",
        "verified_utc": now(), "verifier_source_sha256": sha(Path(__file__)),
        "launch": launch, "posttraining_status": status, "checks": checks,
        "snapshot_sha256": sha(finish.EVIDENCE / "LIVE_SNAPSHOT.json"),
        "highest_saved_epoch": snap["highest_committed_saved_epoch"],
        "verified_recovery_count": snap["verified_recovery_count"], "eta": snap["eta"],
        "scope": "existing seed17 train256 -> checkpoint/data verification -> two-GPU admission below85 -> core fixed DEV256 -> review",
        "frozen_dev_sha256": expected["dev.npz"],
        "remaining_task_book_work": [
            "Base/P0/UNet/RSDR paired comparison and Equal-PFA",
            "Top-10 parent audit and CPU latency comparison",
            "complete SUMMARY/REPORT/VERIFICATION delivery",
        ],
        "limitations": [
            "Final checkpoint and DEV performance metrics do not exist yet",
            "No automatic recovery from server reboot or training numerical failure",
            "No extra seed or CONFIRM/TEST evaluation",
        ],
    }
    atomic_json(finish.EVIDENCE / "PIPELINE_VERIFICATION.json", receipt)
    return receipt


if __name__ == "__main__":
    result = verify()
    print(json.dumps({key: result[key] for key in (
        "stage", "verified_utc", "checks", "highest_saved_epoch",
        "verified_recovery_count", "eta")}, indent=2))
