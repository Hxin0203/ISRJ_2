"""WP12b planner diagnostic (server B, CPU only, read-only inputs, no training).

For EVAL frames that violated "stage1 jammer residual >= oracle residual", recompute per pulse:
  in_span  = ||(I-P_D) c|| / ||c||   with c = stage-1 correction, D = exported dictionary
  oracle   = min_c ||J - D c||^2 / ||J||^2   (lstsq, complex128)
  stage1   = ||J - c||^2 / ||J||^2
in_span must be ~0 if the oracle compares against the dictionary stage 1 actually used.
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/code/Xin/ISRJ_222")
FRAMES = [(17, 479, 0), (17, 100, 2), (17, 1883, 2), (17, 75, 0), (17, 888, 1)]   # last one is a non-violating control


def main():
    from tools.paper2_baseline_suite import equal_pfa_suite as E
    torch.set_num_threads(4)
    cp = ROOT / "outputs/recovp_hybrid_seed17/seed_17/epoch_256.pt"
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    net = E.build("hybrid"); net.load_state_dict(payload["model"], strict=True); net.eval()
    out = []
    with np.load(E.EVAL, allow_pickle=False) as z:
        for seed, pi, vi in FRAMES:
            J = z["j"][pi, vi].astype(np.complex128); N = z["epsilon"][pi, vi].astype(np.complex128)
            y = (J + N).astype(np.complex64)
            iq = torch.from_numpy(np.stack((y.real, y.imag), 1).astype(np.float32))
            with torch.inference_mode():
                f = net.factor(iq)
            D = f["dictionary"].numpy().astype(np.complex128)                  # [8, T, K]
            corr = f["correction"].numpy().astype(np.complex128)              # physical units
            u1 = f["stage1_restored"].numpy().astype(np.complex128)
            rows = []
            for p in range(8):
                A, c, b = D[p], corr[p], J[p]
                q, _ = np.linalg.qr(A)
                outside = c - q @ (q.conj().T @ c)
                coef = np.linalg.lstsq(A, b, rcond=1e-12)[0]
                eb = float(np.vdot(b, b).real)
                rows.append({"pulse": p, "jam_energy": eb,
                             "corr_outside_span_rel": float(np.linalg.norm(outside) / max(np.linalg.norm(c), 1e-300)),
                             "oracle_rel": float(np.linalg.norm(b - A @ coef) ** 2 / eb) if eb > 0 else None,
                             "stage1_rel_corr": float(np.linalg.norm(b - c) ** 2 / eb) if eb > 0 else None,
                             "stage1_rel_u1": float(np.linalg.norm(u1[p] - N[p]) ** 2 / eb) if eb > 0 else None,
                             "u1_matches_y_minus_corr": float(np.max(np.abs(u1[p] - (y[p] - c))))})
            out.append({"seed": seed, "parent_index": pi, "variant": vi, "pulses": rows})
            jam = [r for r in rows if r["jam_energy"] > 0]
            print(f"seed {seed} p{pi} v{vi}: jammed pulses {len(jam)} | outside-span max "
                  f"{max(r['corr_outside_span_rel'] for r in rows):.2e} | oracle "
                  f"{[round(r['oracle_rel'], 4) for r in jam]} | stage1 {[round(r['stage1_rel_corr'], 4) for r in jam]}",
                  flush=True)
    (ROOT / "outputs/paper2_baseline_suite_20261126/WP12b/SPANCHECK.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
