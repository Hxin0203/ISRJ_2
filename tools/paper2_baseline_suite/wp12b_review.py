"""Planner review of WP12b ATTRIBUTION.json (local, read-only): size of oracle-bound violations,
label counts, noise-tail and peak-location flags. Usage: python wp12b_review.py <ATTRIBUTION.json>"""
import json
import sys
from collections import Counter

import numpy as np

doc = json.load(open(sys.argv[1], encoding="utf-8"))
print("protocol", doc.get("protocol_sha256", "")[:12], "script", doc.get("script_sha256", "")[:12])
for role in ("evaluation", "calibration"):
    recs = doc[role]["records"]
    o = np.array([r["oracle_pooled"] for r in recs]); s = np.array([r["stage1_pooled"] for r in recs])
    v = s < o - 1e-6
    print(f"\n== {role}: {len(recs)} frames, labels {dict(Counter(r['label'] for r in recs))}")
    print(f"oracle pooled median {np.nanmedian(o):.3f}  stage1 pooled median {np.nanmedian(s):.3f}")
    if v.any():
        gap = (o - s)[v]; rel = gap / o[v]
        print(f"violations {v.sum()}: abs gap median {np.median(gap):.3g} max {gap.max():.3g}; "
              f"relative gap median {np.median(rel):.3g} max {rel.max():.3g}")
        for r in [r for r, x in zip(recs, v) if x][:6]:
            print(f"  seed {r['seed']} p{r['parent_index']} v{r['variant']} oracle {r['oracle_pooled']:.4f} "
                  f"stage1 {r['stage1_pooled']:.4f} removed {sum(r['removed_columns_per_pulse'])} "
                  f"cond {r['gram_condition_max']:.3g}")
    print("noise_exceeds_threshold", sum(r["noise_exceeds_threshold"] for r in recs),
          " peak_within_pm1", sum(r["peak_within_plusminus_1"] for r in recs),
          " solver_failed", sum(r["label"] == "solver_failed" for r in recs))
    print("bootstrap:", {k: (round(x["proportion"], 3) if x["proportion"] is not None else None)
                         for k, x in doc[role]["bootstrap95_by_parent"].items()})
