"""Select an eligible server-B GPU and launch one ReCoVP-Hybrid seed.

Admission is based only on current utilization below 85 percent.  The worker
rechecks the selected GPU, so a race with another job fails closed without
stopping or migrating any existing process.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from tools import server_gpu_admission
from tools.vp_1103_recovp_hybrid_multiseed import ALLOWED_SEEDS, SERVER_HOST, SERVER_ROOT


def choose_with_wait(requested: int | None, poll_seconds: float) -> dict:
    while True:
        try:
            samples = server_gpu_admission.query_gpus()
            if requested is None:
                return server_gpu_admission.select_gpu(samples)
            selected = next(row for row in samples if row["physical_gpu_index"] == requested)
            if selected["utilization_percent"] < server_gpu_admission.UTILIZATION_LIMIT:
                return {**selected, "admission_policy": server_gpu_admission.ADMISSION_POLICY}
            raise server_gpu_admission.NoGpuAvailable(
                f"requested GPU-{requested} is at or above 85 percent: {selected}")
        except server_gpu_admission.NoGpuAvailable:
            if poll_seconds <= 0:
                raise
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--gpu-index", type=int, choices=(0, 1), default=None,
                        help="optional physical GPU; by default choose the least-utilized eligible GPU")
    parser.add_argument("--poll-seconds", type=float, default=60.0,
                        help="wait interval when no eligible GPU exists; <=0 fails immediately")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    server_gpu_admission.require_server_identity()
    selected = choose_with_wait(args.gpu_index, args.poll_seconds)
    gpu = int(selected["physical_gpu_index"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, "-m", "tools.vp_1103_recovp_hybrid_multiseed",
               "--seed", str(args.seed), "--gpu-index", str(gpu)]
    if args.preflight_only:
        command.append("--preflight-only")
    print({"dispatch": command, "selected_gpu": selected}, flush=True)
    completed = subprocess.run(command, cwd=str(SERVER_ROOT), env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
