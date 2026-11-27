"""Read-only: list the weight files WP7 timing needs (path, bytes, SHA256) on server B."""
import hashlib
from pathlib import Path

OUT = Path("D:/code/Xin/ISRJ_222/outputs")
files = [OUT / "recovp_hybrid_seed67/seed_67/epoch_256.pt", OUT / "recovp_hybrid_seed67/seed_67/epoch_256.json",
         OUT / "unet_mf_tail_screen_20261117_v2/tail/seed_17/epoch_256.pt",
         OUT / "correction_mf_tail_20261117_v1/arm/seed_17/epoch_256.pt",
         Path("D:/code/Xin/RSDR_Net/cvednet_baseline_20261125_r1/sealed_e320/cvednet_e320.pt")]
files += sorted((OUT / "rsdr_unified_20261115/assets").glob("*"))
for cp in list(files):
    for side in (cp.with_suffix(".json"), cp.parent / "TRAINING_COMPLETE.json"):
        if cp.suffix == ".pt" and side.is_file() and side not in files:
            files.append(side)
for f in files:
    if f.is_file():
        print(f"{f.stat().st_size:>10} {hashlib.sha256(f.read_bytes()).hexdigest()} {f}")
    else:
        print("MISSING", f)
