"""Fixed TRAIN16 CPU v2 preflight; never starts a formal GPU experiment."""
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from vp_1103_common import (WORK, atomic_json, distribution, frame_jnr, now,
                           parent_mean, provenance, read, sha, stratified, worst_cases, write_cases)
from vp_1103_p2_preflight import load, subset, health
from vp_1103_p2_smoothvp_v2 import SmoothVPNetV2
from vp_1103_p2_smoothvp import operator_apply, noise_gain, real_parameters
from vp_1103_p2_evaluate import iq_tensor, failure_details
from vp_1103_p0 import evaluate
from paper2.train_1103 import make_batch
from paper2.train_256_8seed import seed_rng, atomic_torch_save
from tools.correction_tail_study import objective as objective_module
from tools.durable_io import ProcessLock, atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_v2_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_v2_preflight"
STAGES = ((768, 1e-3, 20261125), (3072, 3e-4, 20261126))


def mechanism_metrics(model, small):
    size = len(small["parent_id"])
    metrics = {key: np.empty((size, 3)) for key in (
        "blind_j_relative_residual", "learned_b_oracle_beta_relative_residual",
        "leak_over_target_energy", "drift_over_target_energy", "cross_term_over_target_energy",
        "removed_noise_gain_frobenius_squared", "matched_delay_change_samples",
        "raw_channel_delay_change_samples", "J_energy_outside_strong_atom_regions",
        "effective_rank_J_weighted", "anchor_fallback_count")}
    max_error, permutation_error = 0., 0.
    with torch.inference_mode():
        for parent in range(size):
            for family in range(3):
                s, j, e = [small[key][parent, family].astype(np.complex128) for key in ("s", "j", "epsilon")]
                y0, y1 = j + e, s + j + e
                x = iq_tensor(y0, "cpu")
                p0, p1 = model.factor(x), model.factor(iq_tensor(y1, "cpu"))
                # Fixed independent permutations are generated without using data.
                perm = torch.tensor(np.random.default_rng(20261127 + parent * 3 + family).permutation(8))
                switched = model.factor(x, anchors_override=p0["anchors"][:, perm])
                permutation_error = max(permutation_error,
                                        float((p0["correction"] - switched["correction"]).abs().max()))
                st, jt, x0, x1 = [torch.from_numpy(v) for v in (s, j, y0, y1)]
                c0 = operator_apply(p0, x0)
                c1 = operator_apply(p1, x1)
                leak = -operator_apply(p1, st)
                drift = c0 - operator_apply(p1, x0)
                lhs = x1 - c1 - (x0 - c0) - st
                max_error = max(max_error, float(torch.linalg.vector_norm(lhs - leak - drift) /
                                                 torch.linalg.vector_norm(x1)))
                normal = p0["gram"] + 1e-6 * torch.eye(8, dtype=torch.complex128)[None]
                beta = torch.linalg.solve(normal, p0["b"].mH @ jt[..., None])
                oracle = (p0["b"] @ beta)[..., 0]
                es, ej = st.abs().square().sum(), jt.abs().square().sum()
                weights = jt.abs().square().sum(-1) / ej
                costs = []
                for pulse in range(8):
                    distances = (p0["delay"][pulse, :, None] - p1["delay"][pulse, None]).abs().numpy()
                    a, b = linear_sum_assignment(distances)
                    costs.append(float(distances[a, b].mean()))
                density = p0["b"].abs().square()
                support = (density / density.amax(1, keepdim=True).clamp_min(1e-24) >= .1).any(-1)
                rank = 64. / p0["gram"].abs().square().sum((-2, -1))
                values = {
                    "blind_j_relative_residual": (jt - c0).abs().square().sum() / ej,
                    "learned_b_oracle_beta_relative_residual": (jt - oracle).abs().square().sum() / ej,
                    "leak_over_target_energy": leak.abs().square().sum() / es,
                    "drift_over_target_energy": drift.abs().square().sum() / es,
                    "cross_term_over_target_energy": 2 * (leak.conj() * drift).real.sum() / es,
                    "removed_noise_gain_frobenius_squared": noise_gain(p0).sum(),
                    "matched_delay_change_samples": (torch.tensor(costs) * weights).sum(),
                    "raw_channel_delay_change_samples": ((p0["delay"] - p1["delay"]).abs().mean(-1) * weights).sum(),
                    "J_energy_outside_strong_atom_regions": (jt.abs().square() * ~support).sum() / ej,
                    "effective_rank_J_weighted": (rank * weights).sum(),
                    "anchor_fallback_count": p0["anchor_fallback_count"].float().mean()}
                for key, value in values.items():
                    metrics[key][parent, family] = float(value)
    if max_error > 1e-12:
        raise ValueError("Paired VP algebra identity failed")
    return metrics, max_error, permutation_error


def gate(metrics, old, jnr, health_result, permutation_error):
    changes = {
        key: metrics[key] - old[key] for key in
        ("learned_b_oracle_beta_relative_residual", "frame_pfa_jammer", "blind_j_relative_residual")}
    paired = stratified(changes, jnr)
    global_fit = float(metrics["learned_b_oracle_beta_relative_residual"].mean())
    families = {name: float(metrics["learned_b_oracle_beta_relative_residual"][:, i].mean())
                for i, name in enumerate(("direct", "repeat", "cyclic"))}
    significant = all(paired[f"{name}/all"]["learned_b_oracle_beta_relative_residual"]
                      ["parent_bootstrap_ci95"][1] < 0 for name in ("repeat", "cyclic"))
    g1 = global_fit <= .25 and families["repeat"] <= .35 and families["cyclic"] <= .35 and significant
    g2 = (metrics["blind_j_relative_residual"].mean() <= .35 and
          paired["global"]["frame_pfa_jammer"]["parent_bootstrap_ci95"][1] < 0)
    g3 = (metrics["drift_over_target_energy"].mean() <= .05 and
          metrics["leak_over_target_energy"].mean() <= .01 and
          metrics["no_jammer_pooled_nmse"].mean() <= 1e-4)
    numerical = (56728 <= health_result["effective_real_parameters"] <= 62700 and
                 health_result["all_parameter_tensors_have_nonzero_gradient"] and
                 health_result["clean_pooled_nmse"] <= 1e-4 and permutation_error < 1e-5)
    return {"G-v2-1": "PASS" if g1 else "FAIL", "G-v2-2": "PASS" if g2 else "FAIL",
            "G-v2-3": "PASS" if g3 else "FAIL", "engineering": "PASS" if numerical else "FAIL",
            "frame_pfa_target_le_10_percent_met": bool(metrics["frame_pfa_jammer"].mean() <= .10),
            "paired_v2_minus_v1": paired}


def coordinate(coordinates, small, old, update):
    folder = PRIVATE / coordinates
    folder.mkdir(parents=True, exist_ok=True)
    seed_rng(17)
    model = SmoothVPNetV2(coordinates).cpu()
    update(coordinates=coordinates, phase="GRADIENT_HEALTH")
    initial = health(model, small, np.arange(16))
    with torch.no_grad():
        x = make_batch(small, np.array([0]), "cpu")["y"]
        p = model.factor(x)
        switched = model.factor(x, anchors_override=p["anchors"][:, torch.arange(7, -1, -1)])
        initial_permutation = float((p["correction"] - switched["correction"]).abs().max())
    trajectory = []
    objective_module.ARM = "tail"
    total = 0
    for steps, lr, seed in STAGES:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        schedule = np.concatenate([rng.permutation(48) for _ in range(steps // 48)])
        for record in schedule:
            optimizer.zero_grad(set_to_none=True)
            batch = make_batch(small, np.array([record]), "cpu")
            correction = model(batch["y"]).reshape(-1, 4, 2, 1800)
            terms = objective_module.study_objective(correction, batch, 1)
            if not all(torch.isfinite(v).all() for v in terms.values()):
                raise FloatingPointError("v2 CPU objective nonfinite")
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += 1
            trajectory.append({key: float(value.detach()) for key, value in terms.items()})
            if total % 32 == 0:
                atomic_json(folder / "TRAJECTORY.json", trajectory)
                update(coordinates=coordinates, phase="TRAIN16_CPU_OVERFIT", step=total, budget=3840,
                       latest_loss=trajectory[-1]["total"])
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": total,
                          "coordinates": coordinates, "parent_ids": small["parent_id"].tolist(),
                          "scope": "CPU TRAIN16 preflight; formal GPU training remains paused"},
                         folder / f"step_{total:04d}.pt")
    update(coordinates=coordinates, phase="POSTFIT_REVIEW", step=total)
    metrics = evaluate(model.eval(), small)
    extra, algebra_error, permutation_error = mechanism_metrics(model, small)
    metrics.update(extra)
    ids, jnr = small["parent_id"], frame_jnr(small)
    raw = folder / "metrics.npz"
    atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=ids, **metrics))
    ordering = np.argsort(ids)
    sorted_small = subset(small, ordering)
    sorted_values = {key: value[ordering] for key, value in metrics.items()}
    cases = worst_cases(sorted_values, ids[ordering], jnr[ordering])
    atoms = failure_details(model, sorted_small, cases, folder)
    return {"metrics": stratified(metrics, jnr), "gate": gate(metrics, old, jnr, initial, permutation_error),
            "initial_health": initial, "initial_permutation_error": initial_permutation,
            "postfit_permutation_error": permutation_error, "algebra_error_max": algebra_error,
            "private_metrics_sha256": sha(raw), "checkpoint_sha256": sha(folder / "step_3840.pt"),
            "worst_cases": cases, "failure_atoms": atoms}


def finish(summary):
    p0dir = WORK / "evidence/outputs/p0_dechirp_tail_2x2"
    p0, state = read(p0dir / "SUMMARY.json"), read(p0dir / "STATUS.json")
    ready = (state["stage"] == "COMPLETE" and p0["coordinates"] in ("RAW_IQ", "DECHIRP") and
             p0.get("six_cell_status") == "COMPLETE")
    if ready:
        for arm, rows in p0["new_arms"].items():
            for seed, row in rows.items():
                cp = WORK / "outputs/p0_dechirp_tail_2x2" / arm / seed / "epoch_256.pt"
                if sha(cp) != row["checkpoint_sha256"]:
                    raise ValueError("P0 final checkpoint binding mismatch")
    summary["G-v2-4"] = "PASS" if ready else "PENDING"
    summary["p0_coordinates"] = p0["coordinates"]
    summary["p0_summary_sha256"] = sha(p0dir / "SUMMARY.json") if ready else None
    selected = p0["coordinates"] if ready else "RAW_IQ"
    row = summary["coordinates"].get(selected)
    numerical = row and all(row["gate"][key] == "PASS" for key in
                             ("G-v2-1", "G-v2-2", "G-v2-3", "engineering"))
    summary["route"] = ("AWAIT_HUMAN_CONFIRMATION" if numerical and ready else
                        "NO_GO_REVIEW" if row and not numerical else "WAITING_RECEIPTS")
    summary["formal_gpu_training_paused"] = True
    atomic_json(OUT / "SUMMARY.json", summary)
    lines = ["# SmoothVP-Net v2 CPU 预检", "",
             f"路由 **{summary['route']}**。P0 坐标 **{p0['coordinates']}**。正式 GPU 训练保持暂停。",
             "", "| 坐标 | 参数 | Oracle-β 残差 | repeat | cyclic | 盲残差 | 虚警 | 漂移比 | 泄漏比 | 无扰 NMSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary["coordinates"].items():
        m = result["metrics"]
        values = [m["global"]["learned_b_oracle_beta_relative_residual"]["mean"],
                  m["repeat/all"]["learned_b_oracle_beta_relative_residual"]["mean"],
                  m["cyclic/all"]["learned_b_oracle_beta_relative_residual"]["mean"]]
        values += [m["global"][key]["mean"] for key in (
            "blind_j_relative_residual", "frame_pfa_jammer", "drift_over_target_energy",
            "leak_over_target_energy", "no_jammer_pooled_nmse")]
        lines.append(f"| {name} | {result['initial_health']['effective_real_parameters']} | " +
                     " | ".join(f"{value:.6g}" for value in values) + " |")
    lines += ["", "| 坐标 | G-v2-1 | G-v2-2 | G-v2-3 | 数值与置换 |", "|---|---|---|---|---|"]
    for name, row in summary["coordinates"].items():
        g = row["gate"]
        lines.append(f"| {name} | {g['G-v2-1']} | {g['G-v2-2']} | {g['G-v2-3']} | {g['engineering']} |")
    lines += ["", f"G-v2-4 **{summary['G-v2-4']}**。",
              "虚警≤10%作为任务书所述目标单列记录，G-v2-2 的显著改善采用同 parent 配对区间。",
              "所有模型均使用固定末步，保留 768 和 3840 步权重及完整轨迹。此处为 TRAIN16 诊断，范围不延伸到 DEV 泛化。", ""]
    for name, row in summary["coordinates"].items():
        lines += [f"## {name} 最差 parent", ""]
        write_cases(lines, row["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


def run():
    torch.set_num_threads(2)
    started = now()
    first = read(WORK / "evidence/outputs/p2_head_substitution_audit/SUMMARY.json")
    if first["stage2_route"] != "GO_V2_CPU_PREFLIGHT":
        raise RuntimeError("Controlled substitution did not support stage2")
    data, binding = load("train")
    original = torch.load(WORK / "outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt",
                          map_location="cpu", weights_only=False)
    indices = np.searchsorted(data["parent_id"], original["parent_ids"])
    small = subset(data, indices)
    status = {**provenance(__file__, *[WORK / "tools" / name for name in (
        "vp_1103_p2_smoothvp_v2.py", "vp_1103_p2_smoothvp.py", "vp_1103_p2_preflight.py",
        "vp_1103_p2_evaluate.py", "vp_1103_p0.py")]),
              "stage": "RUNNING", "started_utc": started, "ended_utc": None, "pid": os.getpid(),
              "device": "cpu", "seed": 17, "micro_batch_records": 1, "budget_per_coordinate": 3840,
              "data": {"train": binding}, "parent_ids": small["parent_id"].tolist(),
              "formal_gpu_training_paused": True}

    def update(**fields):
        status.update(fields, updated_utc=now())
        atomic_json(OUT / "STATUS.json", status)

    update()
    summary = {"coordinates": {}, "seed": 17, "parent_ids": small["parent_id"].tolist(),
               "substitution_summary_sha256": sha(WORK / "evidence/outputs/p2_head_substitution_audit/SUMMARY.json"),
               "optimization": {"stages": STAGES, "loss": "endpoint + .1 pc + damage + .05 tail",
                                "clip_grad_norm": 5., "additional_losses": []},
               "delay_probability": "Not applicable: v2 uses explicit peak anchors and shared bounded refinements."}
    for coordinates in ("RAW_IQ", "DECHIRP"):
        old_path = WORK / "outputs/p2_smoothvp_preflight" / coordinates / "fit_diagnosis.npz"
        with np.load(old_path, allow_pickle=False) as z:
            order = np.searchsorted(z["parent_id"], small["parent_id"])
            old = {key: z[key][order] for key in (
                "learned_b_oracle_beta_relative_residual", "frame_pfa_jammer", "blind_j_relative_residual")}
        summary["coordinates"][coordinates] = coordinate(coordinates, small, old, update)
        finish(summary)
    finish(summary)
    update(stage="COMPLETE", ended_utc=now(), summary_sha256=sha(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "PROCESS.lock"):
        try:
            run()
        except Exception as exc:
            path = OUT / "STATUS.json"
            prior = read(path) if path.exists() else {}
            atomic_json(OUT / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(path, {**prior, "stage": "FAILED", "ended_utc": now(), "error": repr(exc)})
            raise
