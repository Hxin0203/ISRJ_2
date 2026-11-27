"""Read-only: sizes/SHA of WP12b rerun inputs on server B, and WP5b progress."""
import hashlib
import json
from pathlib import Path

R = Path("D:/code/Xin/ISRJ_222")
EQ = R / "data/unet_ablation_3seed_256_20261115/equal_pfa"
for p in (EQ / "calibration/test.npz", EQ / "evaluation/test.npz"):
    print(p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:16], p)
for s in (17, 29, 41, 53, 67):
    d = R / f"outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_{s}"
    print(s, [(f.name, f.stat().st_size) for f in sorted(d.glob("*"))])
    cp = R / f"outputs/recovp_hybrid_seed{s}/seed_{s}"
    print("   ckpt", [(f.name, f.stat().st_size) for f in sorted(cp.glob("epoch_256*"))])
w = R / "outputs/paper2_baseline_suite_20261126/WP5b/results"
print("WP5b DONE:", sum(1 for _ in w.glob("*/*/DONE.json")), "of", 41 + 5)
