"""WP6 Fig. 5: restore 3 pre-selected DEV parents and export light plotting arrays.

Parents (rule in wp6_figures.py / FIG6_STATS.json): DEV indices 100, 145, 77 (P10/P50/P90 of
the Hybrid - UNet-Tail damage difference). Hybrid seed 67 (median DEV E_U seed), UNet-Tail
seed 17, frozen RSDR-Net, Max-TF. Exported per parent and jammer type: complex waveforms of
one jammed pulse, pulse-compression magnitude of that pulse, and the range-Doppler power of
the J+S+E frame. Usage on server B: ``python -m ... launch`` (selects a GPU below 85%).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/code/Xin/ISRJ_222")
OUT = ROOT / "outputs/paper2_baseline_suite_20261126/WP6"
DEV = ROOT / "data/paper2_1103/dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
PICKS = {"P10": 100, "P50": 145, "P90": 77}


def work():
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite import rsdr_suite as R
    from tools.paper2_baseline_suite import classical_suite as C
    from paper2.evaluation_1103_v4 import states_from_components
    from paper2.radar_eval import doppler_process, pulse_compress
    admission = E.require_server()
    E.configure_precision()
    if hashlib.sha256(DEV.read_bytes()).hexdigest() != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    idx = list(PICKS.values())
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k][idx] for k in ("s", "j", "epsilon"))
        pid = z["parent_id"][idx]
    st = states_from_components(s, j, e)
    hyb, hb = E.load("ReCoVP-Hybrid", 67)
    r = E.restore(hyb, "hybrid", st)
    unet, ub = E.load("UNet-Tail", 17)
    net, rb = R.load()
    outs = {"Observation": st, "Reference S+E": None, "Stage 1": r["stage1"], "Final": r["main"],
            "UNet-Tail": E.restore(unet, E.FAMILIES["UNet-Tail"][1], st)["main"],
            "RSDR-Net": R.restore(net, st), "Max-TF": C.restore("max_tf_isrj1", st, j)}
    jammed = np.abs(j).max(axis=-1) > 0                      # [3 parents, 3 types, 8 pulses]
    pulse = np.argmax(jammed, axis=-1)                       # first jammed pulse per parent/type
    data = {"parent_id": pid, "pulse": pulse, "target": s[np.arange(3)[:, None], np.arange(3)[None], pulse],
            "jammer": j[np.arange(3)[:, None], np.arange(3)[None], pulse]}
    for name, u in outs.items():
        x = st[:, 1] if name == "Reference S+E" else u[:, 3]   # [3, 3, 8, T]
        if name == "Observation":
            x = st[:, 3]
        key = name.replace(" ", "_").replace("+", "p")
        pc, _ = pulse_compress(x)
        data[f"{key}__wave"] = x[np.arange(3)[:, None], np.arange(3)[None], pulse].astype(np.complex64)
        data[f"{key}__pc_db"] = (20 * np.log10(np.abs(pc[np.arange(3)[:, None], np.arange(3)[None], pulse]) + 1e-12)).astype(np.float32)
        rd = np.stack([np.stack([np.abs(doppler_process(pc[p, t])[0]) ** 2 for t in range(3)]) for p in range(3)])
        data[f"{key}__rd_db"] = (10 * np.log10(rd + 1e-24)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "FIG5_INPUT.npz", **data)
    (OUT / "FIG5_INPUT.json").write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "picks": PICKS, "dev_sha256": DEV_SHA, "hybrid_seed": 67, "unet_tail_seed": 17,
        "bindings": {"hybrid": hb, "unet_tail": ub, "rsdr": rb}, "gpu_admission": admission,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1, default=str), encoding="utf-8")
    print("saved", (OUT / "FIG5_INPUT.npz").stat().st_size, flush=True)


def launch():
    from tools.paper2_baseline_suite.wp5b_suite import _pick_gpu
    from tools.server_gpu_admission import NoGpuAvailable
    while True:
        try:
            gpu = _pick_gpu(1)
        except NoGpuAvailable:
            time.sleep(60); continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu["physical_gpu_index"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        code = subprocess.call([sys.executable, "-m", "tools.paper2_baseline_suite.wp6_fig5_infer", "work"], cwd=str(ROOT), env=env)
        if code == 0:
            return
        if code != 3:
            raise SystemExit(f"fig5 worker failed with exit code {code}")
        time.sleep(60)


if __name__ == "__main__":
    if sys.argv[1:] == ["work"]:
        from tools.server_gpu_admission import NoGpuAvailable
        try:
            work()
        except NoGpuAvailable as exc:
            print("gpu busy", exc, flush=True); sys.exit(3)
    else:
        launch()
