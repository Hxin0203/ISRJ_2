"""Diagnostic input and ideal interference-removal references on the fixed master."""
import os
import time

import numpy as np
import pandas as pd

from tools.hybrid_condition_benchmark import load_components, score, OUT, REPORT, PRIMARY_SNR, PRIMARY_JSR, EVAL_PARENTS, cell_name, regime, write_json, now


def main():
    from paper2.evaluation_1103_v4 import states_from_components
    b, manifest = load_components()
    rows = []
    started = time.perf_counter()
    for snr in PRIMARY_SNR:
        e = b["epsilon"] * np.float32(10 ** (-snr / 20))
        for jsr in PRIMARY_JSR:
            j = b["j"] * np.float32(10 ** (jsr / 20))
            for label in ["Observed", "Ideal J removal"]:
                folder = OUT / "references" / label
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{cell_name(snr, jsr)}.npz"
                if not path.exists():
                    parts = []
                    for lo in range(0, EVAL_PARENTS, 8):
                        hi = min(lo + 8, EVAL_PARENTS)
                        s0, j0, e0 = b["s"][lo:hi], j[lo:hi], e[lo:hi]
                        observed = states_from_components(s0, j0, e0)
                        restored = observed if label == "Observed" else states_from_components(s0, np.zeros_like(j0), e0)
                        parts.append(score(s0, j0, e0, restored, b["doppler_hz"][lo:hi], {}))
                    merged = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
                    tmp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(tmp, parent_id=b["parent_id"], **merged)
                    os.replace(tmp, path)
                with np.load(path, allow_pickle=False) as z:
                    rows.append({"method": label, "snr_db": snr, "jsr_db": jsr, "regime": regime(snr),
                                 **{key: float(z[key].mean()) for key in z.files if key != "parent_id"}})
            print("references", snr, jsr, round(time.perf_counter() - started, 1), flush=True)
    pd.DataFrame(rows).to_csv(REPORT / "REFERENCE_CONDITION_METRICS.csv", index=False)
    write_json(OUT / "references/DONE.json", {"finished_utc": now(), "master_sha256": manifest["component_sha256"],
               "conditions": len(PRIMARY_SNR) * len(PRIMARY_JSR), "parents_per_cell": EVAL_PARENTS,
               "threshold_policy": "Original nominal cell-CFAR only, no new threshold calibration",
               "ideal_reference": "Truth-known diagnostic S+E, retains noise; unavailable to deployed processing"})


if __name__ == "__main__":
    main()
