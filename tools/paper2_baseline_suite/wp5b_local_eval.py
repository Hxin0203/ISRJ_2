"""Run pending WP5b jobs on the LOCAL GPU (evaluation only; user authorization 2026-11-27).

The server-B GPU lanes stalled: utilization from the resumed seed-79/97 training jumped back to
~100% between GPU selection and the admission re-check, so almost no job was admitted for ~15 min.
This wrapper runs the UNCHANGED ``wp5b_suite.evaluate`` (byte-identical to server B) with only
paths and the server identity check redirected to the local byte-copy mirror; the admission record
in DONE.json states the local host. Results are uploaded to the server results tree afterwards.
Usage: python -m tools.paper2_baseline_suite.wp5b_local_eval --worker K   (K = 0, 1, 2)
       python -m tools.paper2_baseline_suite.wp5b_local_eval --method M [--seed S]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL))
MIRROR = LOCAL / "outputs/wp5b_local"
SERVER_DATA = "D:/code/Xin/ISRJ_222/data/paper2_baseline_suite_20261126/wp5b"
JOBS = [("CorrectionNet-Tail", 17), ("CorrectionNet-Tail", 41), ("UNet-Tail", 29), ("UNet-NoiseNorm", 17),
        ("UNet-NoiseNorm", 41), ("UNet-Freq", 29), ("UNet-Dilation", 17), ("UNet-Dilation", 29),
        ("UNet-Dilation", 41), ("RSDR-Net", None)]


def local_manifest() -> Path:
    src = LOCAL / "data/paper2_baseline_suite_20261126/wp5b/DATA_MANIFEST.json"
    m = json.loads(src.read_text(encoding="utf-8"))
    for c in m["cells"].values():
        c["path"] = str(LOCAL / "data/paper2_baseline_suite_20261126/wp5b" / Path(c["path"]).relative_to(Path(SERVER_DATA)))
    d = MIRROR / "manifest"; d.mkdir(parents=True, exist_ok=True)
    (d / "DATA_MANIFEST.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return d


def run_one(method, seed):
    import torch
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite import wp5b_suite as W
    W.ROOT = LOCAL
    W.SUITE = LOCAL / "outputs/paper2_baseline_suite_20261126"
    W.OUT = MIRROR / "WP5b"
    W.DATA = local_manifest()
    E.ROOT, E.OUTPUTS = LOCAL, LOCAL / "outputs"

    def require_local():
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one local CUDA device required")
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        return {"host": platform.node(), "gpu_name": torch.cuda.get_device_name(0), "nvidia_smi": util,
                "admission_policy": "local workstation GPU, user authorization 2026-11-27 (server GPUs saturated)"}
    E.require_server = require_local
    W.evaluate(method, seed)


def worker(k):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for method, seed in JOBS[k::3]:
        args = [sys.executable, "-m", "tools.paper2_baseline_suite.wp5b_local_eval", "--method", method]
        if seed is not None:
            args += ["--seed", str(seed)]
        code = subprocess.call(args, cwd=str(LOCAL), env=env)
        print("job", method, seed, "exit", code, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=int)
    p.add_argument("--method")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    worker(a.worker) if a.worker is not None else run_one(a.method, a.seed)
