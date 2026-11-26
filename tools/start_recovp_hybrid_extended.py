"""Resource-aware launcher for additional ReCoVP-Hybrid seeds."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from tools import server_gpu_admission
from tools.vp_1103_recovp_hybrid_extended import ALLOWED_SEEDS, SERVER_ROOT


def choose_gpu(poll_seconds: float) -> dict:
    while True:
        try:
            return server_gpu_admission.choose_gpu()
        except server_gpu_admission.NoGpuAvailable:
            if poll_seconds <= 0:
                raise
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    server_gpu_admission.require_server_identity()
    selected = choose_gpu(args.poll_seconds)
    gpu = int(selected["physical_gpu_index"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, "-m", "tools.vp_1103_recovp_hybrid_extended",
               "--seed", str(args.seed), "--gpu-index", str(gpu)]
    print({"dispatch": command, "selected_gpu": selected}, flush=True)
    raise SystemExit(subprocess.run(command, cwd=str(SERVER_ROOT), env=env, check=False).returncode)


if __name__ == "__main__":
    main()
