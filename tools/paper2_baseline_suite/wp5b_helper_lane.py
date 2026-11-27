"""WP5b helper lane (scheduling only; evaluation code unchanged).

Takes lane gpuA's pending jobs from the END of its list, so gpuA (front to back) and this helper
meet in the middle. Stops at the first job that already has partial cell files (gpuA is on it).
Each job runs the unchanged ``wp5b_suite eval``; results are deterministic and written
atomically, so a rare duplicate at the meeting point is harmless.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
sys.path.insert(0, str(ROOT))
from tools.paper2_baseline_suite import wp5b_suite as W  # noqa: E402
from tools.server_gpu_admission import NoGpuAvailable  # noqa: E402

lane = [j for j in W.jobs() if j[0] not in W.CPU_METHODS][0::2]
for method, seed in reversed(lane):
    tag = f"seed_{seed}" if seed is not None else "single"
    folder = W.OUT / "results" / method / tag
    if (folder / "DONE.json").is_file():
        continue
    if folder.is_dir() and any(folder.glob("cell_*.npz")):
        print("met gpuA at", method, seed, flush=True)
        break
    while True:
        try:
            gpu = W._pick_gpu(1)
        except NoGpuAvailable:
            time.sleep(60); continue
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUDA_VISIBLE_DEVICES": str(gpu["physical_gpu_index"])}
        args = [sys.executable, "-m", "tools.paper2_baseline_suite.wp5b_suite", "eval", "--method", method]
        if seed is not None:
            args += ["--seed", str(seed)]
        code = subprocess.call(args, cwd=str(ROOT), env=env)
        if code == 3:
            time.sleep(60); continue
        print("helper", method, seed, "exit", code, "gpu", gpu["physical_gpu_index"], flush=True)
        break
print("helper finished", flush=True)
