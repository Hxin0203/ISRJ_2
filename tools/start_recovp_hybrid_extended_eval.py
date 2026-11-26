"""Select an eligible server-B GPU and run fixed DEV256 for an added seed."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from tools import server_gpu_admission
from tools.vp_1103_recovp_hybrid_extended import ALLOWED_SEEDS, SERVER_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--batch-parents", type=int, default=2)
    args = parser.parse_args()
    server_gpu_admission.require_server_identity()
    while True:
        try:
            selected = server_gpu_admission.choose_gpu()
            break
        except server_gpu_admission.NoGpuAvailable:
            if args.poll_seconds <= 0:
                raise
            time.sleep(args.poll_seconds)
    gpu = int(selected["physical_gpu_index"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [sys.executable, "-m", "tools.evaluate_recovp_hybrid_extended",
               "--seed", str(args.seed), "--gpu-index", str(gpu),
               "--batch-parents", str(args.batch_parents)]
    print({"dispatch": command, "selected_gpu": selected}, flush=True)
    raise SystemExit(subprocess.run(command, cwd=str(SERVER_ROOT), env=env, check=False).returncode)


if __name__ == "__main__":
    main()
