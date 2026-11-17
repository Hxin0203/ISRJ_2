"""Read-only server audit; publish only lightweight text evidence."""
from __future__ import annotations

import csv
import io
import platform
import subprocess
from pathlib import Path

from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_copy, atomic_json

RUNS = ("unet_mf_tail_screen_20261117_v2", "unet_noise_norm_screen_20261117_v1")


def main():
    if platform.node().casefold() != "desktop-rfferm3" or ROOT.resolve() != Path("D:/code/Xin/ISRJ_222").resolve():
        raise RuntimeError("Run snapshot on server B in the dedicated second-paper directory")
    summaries, data_cache = [], {}
    for run in RUNS:
        base, evidence = ROOT / "outputs" / run, ROOT / "evidence/outputs" / run
        cfg = read(base / "CONFIG.json")
        drift = [name for name, digest in cfg["source_sha256"].items() if sha(ROOT / name) != digest]
        data_matches = {}
        for role in ("train", "dev"):
            path = ROOT / f"data/paper2_1103/{role}.npz"
            if role not in data_cache:
                data_cache[role] = sha(path)
            expected = cfg["training_data_sha256" if role == "train" else "dev_data_sha256"]
            data_matches[role] = data_cache[role] == expected
        status = read(base / "STATUS.json")
        for name in ("CONFIG.json", "STATUS.json", "ADMISSION.json", "NUMERICAL_PREFLIGHT.json",
                     "PIPELINE_STATUS.json", "DEV_SCREEN_SUMMARY.json"):
            if (base / name).exists():
                atomic_copy(base / name, evidence / name)
        for failure in base.glob("CONTROLLER_FAILURE_*.json"):
            atomic_copy(failure, evidence / failure.name)
        factor = "tail" if "tail" in run else "norm"
        seeds = []
        for seed in (17, 29, 41):
            folder = base / factor / f"seed_{seed}"
            recovery = sorted((folder / "recovery").glob("epoch_*.json"))
            saved_epoch = None
            digest, matches, metrics = None, None, None
            bad_recovery = []
            recovery_epochs = []
            bad_identity = []
            for receipt in recovery:
                rec = read(receipt)
                recovery_epochs.append(rec["epoch"])
                actual = sha(receipt.with_suffix(".pt"))
                if actual != rec["checkpoint_sha256"]:
                    bad_recovery.append(rec["epoch"])
                identity = rec.get("identity", {})
                if (identity.get("run") != run or identity.get("seed") != seed
                        or identity.get("config_sha256") != sha(base / "CONFIG.json")
                        or identity.get("data_sha256") != cfg["training_data_sha256"]
                        or identity.get("source_sha256") != cfg["source_sha256"]):
                    bad_identity.append(rec["epoch"])
            if recovery:
                last = recovery[-1]
                rec = read(last)
                saved_epoch = rec["epoch"]
                digest = sha(last.with_suffix(".pt"))
                matches = digest == rec["checkpoint_sha256"]
                metric = folder / f"training_epoch_{saved_epoch:03d}.json"
                if metric.exists():
                    metrics = read(metric)
                    atomic_copy(metric, evidence / factor / f"seed_{seed}" / "LATEST_TRAINING_EPOCH.json")
            milestones = []
            for receipt in sorted(folder.glob("epoch_*.json")):
                rec = read(receipt)
                actual = sha(receipt.with_suffix(".pt"))
                milestones.append({"epoch": rec["epoch"], "sha256": actual,
                                   "sidecar_matches": actual == rec["checkpoint_sha256"]})
            complete = folder / "TRAINING_COMPLETE.json"
            if complete.exists():
                atomic_copy(complete, evidence / factor / f"seed_{seed}" / complete.name)
            expected_epochs = set(range(saved_epoch + 1)) if saved_epoch is not None else set()
            expected_milestones = {e for e in expected_epochs if e > 0 and e % 32 == 0}
            seeds.append({"seed": seed, "training_complete": complete.exists(), "saved_epoch": saved_epoch,
                          "recovery_points": len(recovery), "checkpoint_sha256": digest,
                          "sidecar_matches": matches, "full_recovery_sha_audit_failures": bad_recovery,
                          "recovery_identity_failures": bad_identity,
                          "missing_recovery_epochs": sorted(expected_epochs - set(recovery_epochs)),
                          "missing_milestones": sorted(expected_milestones - {m["epoch"] for m in milestones}),
                          "milestones": milestones, "latest_epoch_metrics": metrics})
        summaries.append({"run": run, "physical_gpu": int(cfg["execution"]["cuda_visible_devices"]),
                          "config_sha256": sha(base / "CONFIG.json"), "source_files": len(cfg["source_sha256"]),
                          "source_drift": drift, "data_sha_matches": data_matches, "status": status,
                          "seeds": seeds, "dev_summary_exists": (base / "DEV_SCREEN_SUMMARY.json").exists(),
                          "pipeline_status_exists": (base / "PIPELINE_STATUS.json").exists()})
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], text=True)
    gpu = [{"index": int(row[0]), "uuid": row[1].strip(), "utilization_percent": float(row[2]),
            "memory_used_mib": float(row[3]), "memory_total_mib": float(row[4])}
           for row in csv.reader(io.StringIO(output))]
    result = {"checked_utc": now(), "host": platform.node(), "gpu": gpu, "experiments": summaries}
    atomic_json(ROOT / "evidence/outputs/PARALLEL_GPU_STATUS_20261117.json", result)
    import json
    print(json.dumps({"checked_utc": result["checked_utc"], "gpu": gpu,
                      "experiments": [{"run": r["run"], "stage": r["status"]["stage"],
                          "seeds": [{k: s[k] for k in ("seed", "saved_epoch", "recovery_points", "training_complete", "sidecar_matches", "full_recovery_sha_audit_failures")} for s in r["seeds"]],
                          "source_drift": r["source_drift"], "data_sha_matches": r["data_sha_matches"],
                          "dev_summary_exists": r["dev_summary_exists"]} for r in summaries]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
