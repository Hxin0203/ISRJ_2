"""Read-only probe for WP5b planning on server B (no writes, no inference)."""
import json, subprocess
from pathlib import Path

ROOT = Path("D:/code/Xin/ISRJ_222")
print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
e3 = ROOT / "outputs/paper2_baseline_suite_20261126/E3"
rows = sorted(e3.glob("*/seed_*/ROW.json"))
print("E3 ROW files:", len(rows))
if rows:
    r = json.loads(rows[0].read_text(encoding="utf-8"))
    print(rows[0])
    print("keys:", list(r.keys()))
    for k, v in r.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(k, "first:", json.dumps(v[0])[:400])
print("families:", sorted({p.parent.parent.name for p in rows}))
e5 = ROOT / "outputs/paper2_baseline_suite_20261126/E5/results"
print("E5 methods:", sorted(p.name for p in e5.glob("*")) if e5.is_dir() else None)
for name in ("RSDR-Net", "CVEDNet", "Max-TF", "GMM-OMP"):
    d = e3 / name
    print(name, "E3:", sorted(x.name for x in d.glob("*")) if d.is_dir() else "absent")
ext = ROOT / "outputs/paper2_baseline_suite_20261126"
print("suite dirs:", sorted(p.name for p in ext.glob("*")))
