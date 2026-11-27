"""WP6 Fig. 5 export on the LOCAL GPU (evaluation only; user authorization 2026-11-27).

Same picks, seeds and export as wp6_fig5_infer.py (server version, queued behind saturated
GPUs); paths are local byte copies SHA-checked against server-B values.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL))
from tools.paper2_baseline_suite import wp7_local_timing as T  # noqa: E402  (SHA-checked loaders)

DEV = LOCAL / "data/paper2_1103/dev.npz"
DEV_SHA = "0e71bf9ddb43a0a89f62ce3e09e741d9ea20c19596ab75aebc787cd1e7796dc1"
PICKS = {"P10": 100, "P50": 145, "P90": 77}
OUT = LOCAL / "evidence/outputs/paper2_baseline_suite_20261126/WP6"


def main():
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    from tools.paper2_baseline_suite import classical_suite as C
    from tools.rsdr_adapter import RSDRNet
    from paper2.evaluation_1103_v4 import states_from_components, restore_model
    from paper2.radar_eval import doppler_process, pulse_compress
    E.configure_precision()
    if T.sha(DEV) != DEV_SHA:
        raise RuntimeError("DEV identity changed")
    idx = list(PICKS.values())
    with np.load(DEV, allow_pickle=False) as z:
        s, j, e = (z[k][idx] for k in ("s", "j", "epsilon"))
        pid = z["parent_id"][idx]
    st = states_from_components(s, j, e)
    hyb, hb = T.load_net("ReCoVP-Hybrid")
    r = E.restore(hyb, "hybrid", st)
    unet, ub = T.load_net("UNet-Tail")
    from tools.paper2_baseline_suite import rsdr_suite as R
    net = RSDRNet(device="cuda:0", batch_pulses=256, verify_vendor=True)
    rs = R.restore(net, st)                                                  # [3, 4, 3, 8, T]
    outs = {"Observation": st, "Reference S+E": st, "Stage 1": r["stage1"], "Final": r["main"],
            "UNet-Tail": restore_model(unet, st, torch.device("cuda")), "RSDR-Net": rs,
            "Max-TF": C.restore("max_tf_isrj1", st, j)}
    a3 = np.arange(3)
    jammed = np.abs(j).max(axis=-1) > 0
    pulse = np.argmax(jammed, axis=-1)
    data = {"parent_id": pid, "pulse": pulse, "target": s[a3[:, None], a3[None], pulse],
            "jammer": j[a3[:, None], a3[None], pulse], "noise": e[a3[:, None], a3[None], pulse]}
    for name, u in outs.items():
        x = u[:, 1] if name == "Reference S+E" else u[:, 3]
        key = name.replace(" ", "_").replace("+", "p")
        pc, _ = pulse_compress(x)
        data[f"{key}__wave"] = x[a3[:, None], a3[None], pulse].astype(np.complex64)
        data[f"{key}__pc_db"] = (20 * np.log10(np.abs(pc[a3[:, None], a3[None], pulse]) + 1e-12)).astype(np.float32)
        rd = np.stack([np.stack([np.abs(doppler_process(pc[p, t])[0]) ** 2 for t in range(3)]) for p in range(3)])
        data[f"{key}__rd_db"] = (10 * np.log10(rd + 1e-24)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "FIG5_INPUT.npz", **data)
    (OUT / "FIG5_INPUT.json").write_text(json.dumps({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "local workstation GPU", "picks": PICKS, "dev_sha256": DEV_SHA, "hybrid_seed": 67, "unet_tail_seed": 17,
        "bindings": {"hybrid": hb, "unet_tail": ub}, "script_sha256": T.sha(Path(__file__))}, indent=1), encoding="utf-8")
    print("saved", (OUT / "FIG5_INPUT.npz").stat().st_size, flush=True)


if __name__ == "__main__":
    main()
