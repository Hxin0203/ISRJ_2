"""Separate TRAIN16 dictionary coverage, gate bias and paired operator drift."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, provenance, read, sha,
                           stratified, worst_cases, write_cases)
from vp_1103_p2_smoothvp import SmoothVPNet, operator_apply, noise_gain
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_evaluate import iq_tensor, failure_details
from vp_1103_p0 import evaluate
from tools.durable_io import atomic_write, process_alive

OUT = WORK / "evidence/outputs/p2_smoothvp_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"


def run(coordinates):
    torch.set_num_threads(2)
    start = now()
    training, binding = load("train")
    cp = PRIVATE / coordinates / "extended_overfit.pt"
    saved = torch.load(cp, map_location="cpu", weights_only=False)
    ids = np.sort(np.asarray(saved["parent_ids"], np.int64))
    small = subset(training, np.searchsorted(training["parent_id"], ids))
    model = SmoothVPNet(coordinates).cpu().eval()
    model.load_state_dict(saved["model"])
    metrics = evaluate(model, small)
    count = len(ids)
    extra = {name: np.empty((count, 3)) for name in (
        "blind_j_relative_residual", "learned_b_oracle_beta_relative_residual",
        "leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy",
        "noise_gain", "log10_precision_median", "rho_max", "mean_max_delay_probability")}
    max_error = 0.
    with torch.inference_mode():
        for parent in range(count):
            for family in range(3):
                s, j, e = [small[key][parent, family].astype(np.complex128) for key in ("s", "j", "epsilon")]
                y0, y1 = j + e, s + j + e
                p0, p1 = [model.factor(iq_tensor(y, "cpu")) for y in (y0, y1)]
                st, jt, x0, x1 = [torch.from_numpy(v) for v in (s, j, y0, y1)]
                ej = jt.abs().square().sum()
                es = st.abs().square().sum()
                b = p0["b"]
                normal = p0["gram"] + 1e-6 * torch.eye(8, dtype=torch.complex128)[None]
                beta = torch.linalg.solve(normal, b.mH @ jt[..., None])
                oracle = (b @ beta).squeeze(-1)
                c0, c1, c10 = operator_apply(p0, x0), operator_apply(p1, x1), operator_apply(p1, x0)
                leak, drift = -operator_apply(p1, st), c0 - c10
                error = torch.linalg.vector_norm((x1 - c1) - (x0 - c0) - st - leak - drift)
                max_error = max(max_error, float(error / torch.linalg.vector_norm(x1)))
                _, _, f2, features, matched = model.encode(iq_tensor(y0, "cpu"))
                logits = model.spatial_head(features).chunk(2, dim=1)[0][..., :1201]
                probability = torch.softmax(logits + model.delay_anchor_bias +
                                            12 * torch.log1p(matched[:, None, :1201]), dim=-1)
                values = {
                    "blind_j_relative_residual": (jt - c0).abs().square().sum() / ej,
                    "learned_b_oracle_beta_relative_residual": (jt - oracle).abs().square().sum() / ej,
                    "leak_over_target_energy": leak.abs().square().sum() / es,
                    "drift_over_target_energy": drift.abs().square().sum() / es,
                    "cross_term_over_target_energy": 2 * (leak.conj() * drift).real.sum() / es,
                    "noise_gain": noise_gain(p0).sum(),
                    "log10_precision_median": torch.log10(p0["precision"]).median(),
                    "rho_max": p0["rho"].max(),
                    "mean_max_delay_probability": probability.max(-1).values.mean()}
                for key, value in values.items():
                    extra[key][parent, family] = float(value)
    metrics.update(extra)
    jnr = frame_jnr(small)
    cases = worst_cases(metrics, ids, jnr)
    diagnosis = failure_details(model, small, cases, PRIVATE / coordinates)
    raw = PRIVATE / coordinates / "fit_diagnosis.npz"
    atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=ids, **metrics))
    summary = {**provenance(__file__, Path(__file__).with_name("vp_1103_p2_smoothvp.py")),
               "stage": "COMPLETE", "coordinates": coordinates, "started_utc": start, "ended_utc": now(),
               "data": {"train": binding}, "parent_count": count, "checkpoint_sha256": sha(cp),
               "metrics": stratified(metrics, jnr), "private_metrics_sha256": sha(raw),
               "identity_max_relative_error": max_error, "failure_atoms": diagnosis, "worst_cases": cases,
               "scope": "TRAIN16 engineering diagnosis. Oracle beta uses J solely to test the learned dictionary."}
    if max_error > 1e-12:
        raise ValueError("Dictionary diagnosis operator identity failed")
    atomic_json(OUT / f"FIT_DIAGNOSIS_{coordinates}.json", summary)
    g = summary["metrics"]["global"]
    lines = [f"# {coordinates} TRAIN16 fit diagnosis", "",
             "Oracle amplitudes are fit on J to test the already learned dictionary. The normal model sees y only.", "",
             "| Measurement | Mean | Parent bootstrap 95% interval |",
             "|---|---:|---:|"]
    for key in extra:
        low, high = g[key]["parent_bootstrap_ci95"]
        lines.append(f"| {key} | {g[key]['mean']:.6g} | {low:.6g}–{high:.6g} |")
    lines += ["", "The dictionary-only optimization report separately tests additional attainable mask and delay fitting.",
              f"Algebraic paired operator identity maximum error is {max_error:.3g}.", "",
              "## Worst training parents", ""]
    write_cases(lines, cases)
    atomic_write(OUT / f"FIT_DIAGNOSIS_{coordinates}.md",
                 lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coordinates", choices=("RAW_IQ", "DECHIRP"))
    parser.add_argument("--wait-for-checkpoint", action="store_true")
    args = parser.parse_args()
    while args.wait_for_checkpoint and not (PRIVATE / args.coordinates / "extended_overfit.pt").exists():
        state = read(OUT / "STATUS.json")
        if state["stage"] == "FAILED" or not process_alive(state["pid"]):
            raise RuntimeError("CPU checkpoint producer stopped")
        time.sleep(20)
    run(args.coordinates)
