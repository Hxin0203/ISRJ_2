"""Read-only status: WP5b lanes/results, GPU load, and WP8 per-parent array layout (for Fig. 6)."""
import json
import subprocess
from pathlib import Path

import numpy as np

SUITE = Path("D:/code/Xin/ISRJ_222/outputs/paper2_baseline_suite_20261126")
W = SUITE / "WP5b"
print(subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
for lane in ("cpu", "gpuA", "gpuB"):
    p = W / f"QUEUE_{lane}.json"
    if p.is_file():
        q = json.loads(p.read_text(encoding="utf-8"))
        print(lane, "done", len(q["done"]), "failed", q["failed"], "finished" if "finished_utc" in q else "")
    else:
        print(lane, "no status yet")
res = W / "results"
if res.is_dir():
    for d in sorted(res.glob("*/*")):
        cells = sorted(x.stem for x in d.glob("cell_*.npz"))
        print(f"  {d.parent.name}/{d.name}: {len(cells)} cells", "DONE" if (d / "DONE.json").is_file() else "")
for log in sorted(W.glob("queue_*.stderr.log"))[-3:]:
    txt = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    print(log.name, "|", " / ".join(txt[-3:])[:400])
for log in sorted(W.glob("queue_*.stdout.log"))[-3:]:
    txt = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    print(log.name, "|", " / ".join(txt[-2:])[:300])
wp8 = SUITE / "WP8"
print("WP8 entries:", sorted(p.name for p in wp8.glob("*"))[:20])
for f in sorted(wp8.rglob("*.npz"))[:3]:
    with np.load(f, allow_pickle=False) as z:
        print(f.relative_to(wp8), {k: z[k].shape for k in z.files})
