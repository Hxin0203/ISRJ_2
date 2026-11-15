"""Engineering-only cross-host native parity and batching checks."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from paper2.io import ROOT, now, sha
from paper2.scan_1103 import build_batch, measure
from paper2.evaluation_1103_v4 import states_from_components
from tools.durable_io import atomic_json, atomic_write
from tools.rsdr_adapter import RSDRNet


def main():
    dest = ROOT / "outputs/rsdr_unified_20261115"
    conditions = [dict(jam_type="repeat", snr_db=-20, jsr_db=35),
                  dict(jam_type="direct", snr_db=0, jsr_db=10)]
    groups = [build_batch("engineering", "main", c, [0]) for c in conditions]
    x = np.concatenate([states_from_components(*g[:3]) for g in groups])
    model = RSDRNet(batch_pulses=2)
    start = time.monotonic()
    native_chunk, details = model.restore(x, diagnostics=True)
    seconds2 = time.monotonic() - start
    model.batch_pulses = 128
    start = time.monotonic()
    batched, diag = model.restore(x, diagnostics=True)
    seconds128 = time.monotonic() - start
    isolated = np.concatenate([model.restore(x[i:i+1]) for i in range(len(x))])
    atol, rtol = 2e-4, 2e-5
    np.testing.assert_allclose(batched, native_chunk, atol=atol, rtol=rtol)
    np.testing.assert_allclose(batched, isolated, atol=atol, rtol=rtol)
    # Only new engineering observations are sent through stdin. No remote files
    # are written, no historical DEV/TEST/Gate arrays are read, no training runs.
    historical = Path("D:/Project Code/ISRJ/RGSSR-ISRJ/engineering/isrrj_width_20261110_r2/runtime_width.py")
    source = historical.read_text(encoding="utf8").split('if __name__ == "__main__":')[0]
    raw = x[0, [0, 2, 1, 3]]
    raw = np.stack((raw.real, raw.imag), axis=2).astype(np.float32)
    payload = base64.b64encode(raw.tobytes()).decode()
    remote = (
        "import os;os.environ['PYTHONDONTWRITEBYTECODE']='1'\n"
        "import base64,json,numpy as np\n"
        f"exec({source!r})\n"
        f"raw=np.frombuffer(base64.b64decode({payload!r}),dtype=np.float32).reshape(4,8,2,1800)\n"
        "pipeline=load('cuda:1')\n"
        "f,rho=infer(raw,pipeline,'cuda:1')[:2]\n"
        "print('NATIVE_PARITY_JSON='+json.dumps({'f':base64.b64encode(f.tobytes()).decode(),'rho':rho.tolist(),'torch':torch.__version__ if 'torch' in globals() else pipeline[1].__version__}))\n"
    )
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "rsdr-b",
                              "D:/code/Xin/ISRJ_Last/RSDR32_lanes/lane1/.venv/Scripts/python.exe -B -"],
                             input=remote, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=180)
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-5000:])
    line = next(l for l in process.stdout.splitlines() if l.startswith("NATIVE_PARITY_JSON="))
    ref = json.loads(line.split("=", 1)[1])
    f = np.frombuffer(base64.b64decode(ref["f"]), dtype=np.float32).reshape(4, 8, 2, 1800)[[0, 2, 1, 3]]
    f = f[:, :, 0] + 1j * f[:, :, 1]
    np.testing.assert_allclose(native_chunk[0], f, atol=atol, rtol=rtol)
    np.testing.assert_allclose(details["rho"][0], np.asarray(ref["rho"])[[0,2,1,3]], atol=2e-5, rtol=2e-5)
    metrics_differences = {}
    for i, (s,j,e,meta) in enumerate(groups):
        obs = states_from_components(s,j,e)
        old = measure("main",s,j,e,obs,native_chunk[i:i+1],meta)
        new = measure("main",s,j,e,obs,batched[i:i+1],meta)
        for k in old:
            diff = float(np.max(abs(old[k]-new[k])))
            metrics_differences[k] = max(metrics_differences.get(k, 0), diff)
        for k in ("pd", "pd_no_jammer", "frame_pfa_noise", "frame_pfa_jammer", "cell_pfa_noise", "cell_pfa_jammer"):
            np.testing.assert_array_equal(old[k], new[k])
    report = dict(status="PASS", checked_utc=now(), role="ENGINEERING_ONLY",
                  engineering_conditions=conditions, historical_reference_source_sha256=sha(historical),
                  cross_host_native_max_abs_waveform_difference=float(abs(native_chunk[0]-f).max()),
                  batch_2_vs_128_max_abs_waveform_difference=float(abs(native_chunk-batched).max()),
                  batch_independence_max_abs_difference=float(abs(batched-isolated).max()),
                  absolute_tolerance=atol, relative_tolerance=rtol,
                  metric_max_absolute_differences=metrics_differences,
                  batch2_seconds=seconds2, batch128_seconds=seconds128,
                  native_torch_version=ref["torch"], local_torch_version=torch.__version__,
                  no_historical_arrays_read=True, no_remote_files_written=True)
    atomic_write(dest / "engineering_parity.npz", lambda stream:np.savez_compressed(stream, observations=x,
                 restored_chunk2=native_chunk, restored_batch128=batched, native_remote=f))
    atomic_json(ROOT / "evidence/outputs/RSDR_UNIFIED_20261115/ADAPTER_PARITY.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
