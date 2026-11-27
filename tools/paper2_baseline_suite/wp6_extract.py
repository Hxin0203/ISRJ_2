"""WP6 (Fig. 6) light extract: per-parent DEV stage metrics from WP8 PARENT_VALUES (5 seeds).

Writes one small npz (no waveforms) for local plotting; also lists E1 per-parent arrays
needed for the Fig. 5 parent-selection rule. Read-only on all inputs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SUITE = Path("D:/code/Xin/ISRJ_222/outputs/paper2_baseline_suite_20261126")
OUT = SUITE / "WP6"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ("u_nmse", "damage", "amplitude", "phase_deg", "noise_nmse", "frame_pfa_jammer", "pd")
data, src = {}, {}
for seed in (17, 29, 41, 53, 67):
    p = SUITE / "WP8" / f"seed_{seed}" / "PARENT_VALUES.npz"
    src[str(seed)] = hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p, allow_pickle=False) as z:
        if seed == 17:
            data["parent_id"] = z["parent_id"]
        assert np.array_equal(z["parent_id"], data["parent_id"])
        for stage in ("stage1_K16_learned", "final_K16_learned"):
            for k in KEYS:
                data[f"s{seed}__{stage}__{k}"] = z[f"{stage}__{k}"]
np.savez(OUT / "FIG6_INPUT.npz", **data)
(OUT / "FIG6_INPUT.json").write_text(json.dumps({"source": "WP8 PARENT_VALUES.npz (DEV256, K=16 learned)",
                                                 "source_sha256": src, "keys": KEYS}, indent=1), encoding="utf-8")
print("saved", (OUT / "FIG6_INPUT.npz").stat().st_size, "bytes")
e1 = SUITE / "E1"
for f in sorted(e1.rglob("*"))[:40]:
    if f.is_file():
        print("E1:", f.relative_to(e1), f.stat().st_size)
