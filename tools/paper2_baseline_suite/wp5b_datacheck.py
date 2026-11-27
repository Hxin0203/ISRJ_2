"""Read-only check of the generated WP5b cells (pairing, timing, coverage, JSR)."""
import json
from pathlib import Path

import numpy as np

DATA = Path("D:/code/Xin/ISRJ_222/data/paper2_baseline_suite_20261126/wp5b")
m = json.loads((DATA / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
print("paired parents:", m["paired_parents"], "overlaps:", {Path(k).name: v for k, v in m["overlap_counts"].items()})
ref = None
for name, c in m["cells"].items():
    with np.load(c["path"], allow_pickle=False) as z:
        s, j, e = z["s"], z["j"], z["epsilon"]
    if ref is None:
        ref = (s, e)
    same = bool(np.array_equal(s, ref[0]) and np.array_equal(e, ref[1]))
    out = []
    for jt, lab in enumerate(("direct", "repeat", "cyclic")):
        jj, ss = j[:, jt], s[:, jt]
        jammed = np.abs(jj).max(axis=-1) > 0
        sel = jammed[..., None] & (np.abs(ss) > 0)
        cover = float(((np.abs(jj) > 0) & sel).sum() / sel.sum())
        out.append(f"{lab}: eta={1 - cover:.3f}")
    print(f"{name:9s} w={c['width']:2d} T={c['period']:3d} jsr={c['actual_jsr_db_mean']:.2f} "
          f"same_s_e_as_first={same} | " + "  ".join(out))
