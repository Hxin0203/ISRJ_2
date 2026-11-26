"""Resource-aware launcher for additional ReCoVP-Hybrid seeds."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from tools import server_gpu_admission
from tools.durable_io import ProcessLock
from tools.recovp_hybrid_seed_schedule import assigned_gpu, validate_completion
from tools.vp_1103_recovp_hybrid_extended import ALLOWED_SEEDS, SERVER_ROOT
from paper2.io import ROOT


def _receipt_path(seed: int) -> Path:
    return ROOT / "evidence" / "outputs" / f"recovp_hybrid_seed{seed}" / "TRAINING_COMPLETE.json"


def _dispatch_lock_path(seed: int) -> Path:
    return ROOT / "evidence" / "outputs" / f"recovp_hybrid_seed{seed}" / "DISPATCH.lock"


def choose_gpu(requested_gpu: int, poll_seconds: float) -> dict:
    while True:
        try:
            samples = server_gpu_admission.query_gpus()
            selected = next(row for row in samples if int(row["physical_gpu_index"]) == requested_gpu)
            if selected["utilization_percent"] >= server_gpu_admission.UTILIZATION_LIMIT:
                raise server_gpu_admission.NoGpuAvailable(
                    f"assigned GPU-{requested_gpu} is at/above 85 percent: {selected}")
            return {**selected, "admission_policy": server_gpu_admission.ADMISSION_POLICY}
        except server_gpu_admission.NoGpuAvailable:
            if poll_seconds <= 0:
                raise
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=None,
                        help="must match the seed's fixed two-GPU lane")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    server_gpu_admission.require_server_identity()
    assigned = assigned_gpu(args.seed)
    if args.gpu_index is not None and args.gpu_index != assigned:
        raise RuntimeError(f"seed {args.seed} is assigned to GPU-{assigned}, not GPU-{args.gpu_index}")

    lock = ProcessLock(_dispatch_lock_path(args.seed))
    while True:
        try:
            lock.acquire()
            break
        except RuntimeError:
            completed = validate_completion(_receipt_path(args.seed), args.seed)
            if completed is not None:
                print({"stage": "ALREADY_COMPLETE", **completed}, flush=True)
                return
            if args.poll_seconds <= 0:
                raise
            time.sleep(args.poll_seconds)

    try:
        completed = validate_completion(_receipt_path(args.seed), args.seed)
        if completed is not None:
            print({"stage": "ALREADY_COMPLETE", **completed}, flush=True)
            return
        selected = choose_gpu(assigned, args.poll_seconds)
        gpu = int(selected["physical_gpu_index"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [sys.executable, "-m", "tools.vp_1103_recovp_hybrid_extended",
                   "--seed", str(args.seed), "--gpu-index", str(gpu)]
        print({"dispatch": command, "selected_gpu": selected, "assigned_gpu": assigned}, flush=True)
        raise SystemExit(subprocess.run(command, cwd=str(SERVER_ROOT), env=env, check=False).returncode)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
