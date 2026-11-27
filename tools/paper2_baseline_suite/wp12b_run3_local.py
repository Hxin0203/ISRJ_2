"""WP12b run_3 on the LOCAL GPU (evaluation only; user authorization 2026-11-27).

Reuses the registered implementation tools/hybrid_wp12b.py (SHA 1744afce..., Stage-1 residual
u1 - epsilon) for selection, oracle, detection chain and bootstrap. Amendments (protocol v2,
written before this run): (1) per-pulse mean ratios use only jammed pulses (||J_p||^2 > 0);
the previous floor 1e-300 turned unjammed-pulse corrections into ~1e296; (2) per-frame span
self-check: the Stage-1 correction must lie in span(D) (outside-span ratio recorded; frames
above 1e-6 are flagged). Comparison overlap uses UNet-Tail (3 seeds, E3) and RSDR-Net (E7).
Inputs are byte copies of server-B files, SHA-checked against values read on server B.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

LOCAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL))
import tools.hybrid_wp12b as W  # noqa: E402  (registered implementation; functions reused unchanged)

OUT = LOCAL / "outputs/wp12b_local/run_3"
EVID = LOCAL / "evidence/outputs/HYBRID_NEXTROUND_20261127/WP12b/run_3"
EQ = LOCAL / "data/unet_ablation_3seed_256_20261115/equal_pfa"
CAL, EVAL = EQ / "calibration/test.npz", EQ / "evaluation/test.npz"
DATA_SHA = {"cal": "5ae29b5af490c514", "eval": "3da7c3b28afb35be"}
E3 = LOCAL / "outputs/paper2_baseline_suite_20261126/E3"
RSDR_SCORES = LOCAL / "outputs/paper2_baseline_suite_20261126/E7/equal_pfa/rsdr_net/SCORES.npz"
SEEDS = W.SEEDS


def load_model(seed, eq):
    cp = LOCAL / f"outputs/recovp_hybrid_seed{seed}/seed_{seed}/epoch_256.pt"
    side = json.loads(cp.with_suffix(".json").read_text(encoding="utf-8"))
    digest = W.sha(cp)
    if digest != side.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint SHA mismatch: {cp}")
    p = torch.load(cp, map_location="cpu", weights_only=False)
    if p.get("seed") != seed or p.get("epoch") != 256:
        raise RuntimeError(f"checkpoint identity mismatch: {cp}")
    for name, d in p["identity"]["source_sha256"].items():
        if W.sha(LOCAL / name) != d:
            raise RuntimeError(f"source SHA mismatch: {name}")
    net = eq.build("hybrid"); net.load_state_dict(p["model"], strict=True); net.cuda().eval()
    return net, {"path": str(cp.relative_to(LOCAL)), "sha256": digest}
def classify(oracle_a, stage1_a, ok):
    if not np.isfinite(oracle_a) or not ok: return "solver_failed"
    if oracle_a > .5: return "coverage_limited"
    if .1 < oracle_a <= .5 and stage1_a <= 1.5 * oracle_a: return "representation_residual"
    if stage1_a > max(.1, 2 * oracle_a): return "shrinkage_limited"
    if oracle_a <= .1 and stage1_a <= .1: return "near_oracle"
    return "other"


def diagnose(seed, role, path, selected, ids, net, threshold):
    records = []
    with np.load(path, allow_pickle=False) as z:
        j_all, e_all = z["j"], z["epsilon"]
        for pi, vi in selected:
            J = j_all[pi, vi].astype(np.complex64); N = e_all[pi, vi].astype(np.complex64)
            wave = J + N
            iq = torch.from_numpy(np.stack([wave.real, wave.imag], 1).astype(np.float32)).cuda()
            with torch.inference_mode():
                f = net.factor(iq)
            D = f["dictionary"].detach().cpu().numpy().reshape(8, 1800, -1).astype(np.complex128)
            u1 = f["stage1_restored"].detach().cpu().numpy().reshape(8, 1800).astype(np.complex128)
            corr = f["correction"].detach().cpu().numpy().reshape(8, 1800).astype(np.complex128)
            o_res, s_res, jam_e, removed, conds, ok, outside = [], [], [], [], [], [], []
            for p in range(8):
                A, b = D[p], J[p].astype(np.complex128)
                norms = np.linalg.norm(A, axis=0); keep = np.isfinite(norms) & (norms >= 1e-8)
                removed.append(int((~keep).sum())); Ak = A[:, keep]
                coef = np.linalg.lstsq(Ak, b, rcond=W.RCOND)[0] if Ak.shape[1] else np.zeros(0, np.complex128)
                rr = b - Ak @ coef; e = float(np.vdot(rr, rr).real)
                ok.append(bool(np.isfinite(coef).all() and np.isfinite(e))); o_res.append(e)
                conds.append(float(np.linalg.cond(Ak.conj().T @ Ak)) if Ak.shape[1] else float("inf"))
                jam_e.append(float(np.vdot(b, b).real))
                sr = u1[p] - N[p].astype(np.complex128); s_res.append(float(np.vdot(sr, sr).real))
                q, _ = np.linalg.qr(A); c = corr[p]
                outside.append(float(np.linalg.norm(c - q @ (q.conj().T @ c)) / max(np.linalg.norm(c), 1e-300)))
            jam_e = np.asarray(jam_e); jammed = jam_e > 0; tj = float(jam_e.sum())
            oa, sa = float(np.sum(o_res)) / tj, float(np.sum(s_res)) / tj
            ob = float(np.mean(np.asarray(o_res)[jammed] / jam_e[jammed]))
            sb = float(np.mean(np.asarray(s_res)[jammed] / jam_e[jammed]))
            unj_corr = float(np.sum(np.asarray(s_res)[~jammed])) / tj if (~jammed).any() else 0.0
            stat, ncell, _ = W.noise_stat(N)
            _, jcell = W.residual_peak(J, J * 0)
            records.append({"seed": seed, "role": role, "parent_index": int(pi), "parent_id": int(ids[pi]),
                "variant": int(vi), "label": classify(oa, sa, all(ok)), "label_mean_pulse": classify(ob, sb, all(ok)),
                "oracle_pooled": oa, "stage1_pooled": sa, "oracle_mean_jammed_pulse": ob, "stage1_mean_jammed_pulse": sb,
                "unjammed_pulse_correction_over_jam_energy": unj_corr, "jammed_pulses": int(jammed.sum()),
                "stage1_below_oracle_tolerance": bool(sa < oa - 1e-6),
                "corr_outside_span_max": float(max(outside)), "span_check_pass": bool(max(outside) < 1e-6),
                "removed_columns": int(sum(removed)), "gram_condition_max": float(max(conds)),
                "rho_max_mean": float(f["rho"].detach().cpu().numpy().max(-1).mean()),
                "precision_min_mean": float(f["precision"].detach().cpu().numpy().min(-1).mean()),
                "alpha_mean": float(f["alpha"].detach().cpu().numpy().mean()),
                "noise_frame_statistic": stat, "hybrid_1pct_threshold": threshold,
                "noise_exceeds_threshold": bool(stat > threshold),
                "peak_within_plusminus_1_of_jammer_only_peak": bool(abs(ncell[0] - jcell[0]) <= 1 and abs(ncell[1] - jcell[1]) <= 1)})
    return records
def extremes(scores, ids):
    k = int(np.ceil(.01 * scores.size)); ix = np.argsort(-scores.ravel(), kind="stable")[:k]
    pi, vi = np.unravel_index(ix, scores.shape)
    return {(int(ids[p]), int(v)) for p, v in zip(pi, vi)}


def summarize(recs):
    out = {"frames": len(recs), "labels": {}, "labels_mean_pulse": {}}
    for key, dst in (("label", "labels"), ("label_mean_pulse", "labels_mean_pulse")):
        for r in recs:
            out[dst][r[key]] = out[dst].get(r[key], 0) + 1
    out["bootstrap95_by_parent"] = W.bootstrap(recs)
    out["bootstrap95_by_parent_mean_pulse"] = W.bootstrap([dict(r, label=r["label_mean_pulse"]) for r in recs])
    out["oracle_bound_violations"] = sum(r["stage1_below_oracle_tolerance"] for r in recs)
    out["span_check_failures"] = sum(not r["span_check_pass"] for r in recs)
    out["noise_tail_frames"] = sum(r["noise_exceeds_threshold"] for r in recs)
    out["medians"] = {k: float(np.median([r[k] for r in recs])) for k in
                      ("oracle_pooled", "stage1_pooled", "rho_max_mean", "precision_min_mean", "alpha_mean",
                       "unjammed_pulse_correction_over_jam_energy")}
    out["by_label_medians"] = {lab: {k: float(np.median([r[k] for r in recs if r["label"] == lab]))
                                     for k in ("oracle_pooled", "stage1_pooled", "rho_max_mean", "precision_min_mean",
                                               "alpha_mean", "jammed_pulses")}
                               for lab in out["labels"]}
    return out


def main():
    from tools.paper2_baseline_suite import equal_pfa_suite as eq
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("exactly one local CUDA device expected")
    eq.configure_precision(); torch.set_num_threads(4)
    for key, p in (("cal", CAL), ("eval", EVAL)):
        if W.sha(p)[:16] != DATA_SHA[key]:
            raise RuntimeError(f"{key} data identity differs from server B")
    OUT.mkdir(parents=True, exist_ok=True); EVID.mkdir(parents=True, exist_ok=True)
    proto = EVID / "PROTOCOL_v2.json"
    if not proto.is_file():
        proto.write_text(json.dumps({"version": 2, "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_protocol_sha256": "2ddddefef75d5f4fa72cdfa98ba1a66dcf057e01c67f7fc40bd1448e9a443b91",
            "base_script_sha256": W.sha(Path(W.__file__)), "script_sha256": W.sha(Path(__file__)),
            "amendments": ["per-pulse mean ratios over jammed pulses only (||J_p||^2 > 0); unjammed-pulse correction reported separately",
                           "per-frame span self-check: Stage-1 correction outside span(D) must be < 1e-6",
                           "overlap: Hybrid 5-seed union of per-seed top-1% vs UNet-Tail 3-seed union (E3) and RSDR-Net (E7)",
                           "labels from pooled (a) ratios as registered; mean-pulse (b) labels reported alongside"],
            "host": "local workstation GPU (user authorization 2026-11-27; server GPUs saturated by resumed training)",
            "training": False, "confirmation_access": False}, indent=1), encoding="utf-8")
    ev_rows, cal_rows, meta = [], [], []
    for seed in SEEDS:
        net, binding = load_model(seed, eq)
        row = json.loads((E3 / "ReCoVP-Hybrid" / f"seed_{seed}" / "ROW.json").read_text(encoding="utf-8"))
        thr = next(x["threshold_ratio"] for x in row["curve"] if x["target_frame_pfa"] == .01)
        with np.load(E3 / "ReCoVP-Hybrid" / f"seed_{seed}" / "SCORES.npz", allow_pickle=False) as z:
            ev_scores, ev_ids = z["eval_resid_peak_noise_units"].copy(), z["eval_parent_id"].copy()
        with np.load(EVAL, allow_pickle=False) as z:
            if not np.array_equal(z["parent_id"], ev_ids):
                raise RuntimeError("E3 SCORES parent order differs from EVAL")
        cal_ev, cal_ids = eq.run_set(net, "hybrid", CAL, full=True)
        cal_scores = cal_ev["main"]["resid_peak_noise_units"]
        for role, scores, ids, path, acc in (("eval", ev_scores, ev_ids, EVAL, ev_rows), ("calibration", cal_scores, cal_ids, CAL, cal_rows)):
            k = int(np.ceil(.01 * scores.size)); ix = np.argsort(-scores.ravel(), kind="stable")[:k]
            pi, vi = np.unravel_index(ix, scores.shape)
            acc.extend(diagnose(seed, role, path, list(zip(pi.tolist(), vi.tolist())), ids, net, thr))
            meta.append({"seed": seed, "role": role, "selected": k, "checkpoint": binding, "threshold": thr})
            print(f"done {role} seed={seed} selected={k}", flush=True)
        del net; torch.cuda.empty_cache()
    hyb = set()
    for s in SEEDS:
        with np.load(E3 / "ReCoVP-Hybrid" / f"seed_{s}" / "SCORES.npz") as z:
            hyb |= extremes(z["eval_resid_peak_noise_units"], z["eval_parent_id"])
    overlap = {}
    for name, files in (("UNet-Tail", [E3 / "UNet-Tail" / f"seed_{s}" / "SCORES.npz" for s in (17, 29, 41)]), ("RSDR-Net", [RSDR_SCORES])):
        keys = set()
        for f in files:
            with np.load(f) as z:
                keys |= extremes(z["eval_resid_peak_noise_units"], z["eval_parent_id"])
        overlap[name] = {"hybrid_union": len(hyb), "method_union": len(keys), "intersection": len(hyb & keys),
                         "jaccard": len(hyb & keys) / max(len(hyb | keys), 1), "hybrid_coverage": len(hyb & keys) / max(len(hyb), 1)}
    doc = {"protocol_v2_sha256": W.sha(proto), "script_sha256": W.sha(Path(__file__)),
           "base_script_sha256": W.sha(Path(W.__file__)), "gpu": torch.cuda.get_device_name(0), "selection": meta,
           "evaluation": {"summary": summarize(ev_rows), "records": ev_rows},
           "calibration": {"summary": summarize(cal_rows), "records": cal_rows}, "overlap_eval": overlap}
    (EVID / "ATTRIBUTION_run3.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({"eval": doc["evaluation"]["summary"], "cal": doc["calibration"]["summary"], "overlap": overlap},
                     indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
