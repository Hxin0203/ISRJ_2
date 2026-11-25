"""TRAIN16 zero-update delay/window substitutions before v2 model admission."""
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, provenance, read, sha,
                           stratified, worst_cases, write_cases, distribution)
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_p2_smoothvp import SmoothVPNet, noise_gain
from vp_1103_replay_truth import recover_selected, replay_components
from paper2.evaluation_1103_v4 import group_metrics, radar_metrics, states_from_components
from tools.durable_io import ProcessLock, atomic_write

OUT = WORK / "evidence/outputs/p2_head_substitution_audit"
PRIVATE = WORK / "outputs/p2_head_substitution_audit"
METHODS = ("S0_blind", "S1_oracle_beta", "S2_blind", "S2_oracle_beta",
           "S3_blind", "S3_oracle_beta", "S4_blind", "S4_oracle_beta",
           "S4_physical_delay_oracle", "S4_range1200_oracle", "S4_analytic_full_phase_oracle",
           "S4_all_delays_full_phase_oracle", "S4_smooth9_oracle")


def dictionary(model, delays, masks, full_phase=False):
    if full_phase:
        n = torch.arange(1800, dtype=torch.float64)
        templates = torch.exp(1j * np.pi / 1200 * (n[None] - delays[:, None] - 300)**2)
        b = templates * masks
        b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min(1e-12)
        return b.T
    return model.dictionary(delays[None], masks[None])[0]


def replacement(parts, components, mode):
    old_delay = parts["delay"][0].double()
    old_mask = parts["mask"][0].double()
    if not components:
        return old_delay, torch.zeros_like(old_mask)
    selected = components[:8]
    tau = np.array([row["phase_delay"] for row in selected])
    truth_mask = np.stack([row["mask"] for row in selected])
    slots, members = linear_sum_assignment(abs(old_delay.numpy()[:, None] - tau[None]))
    delay = old_delay.clone()
    masks = torch.zeros_like(old_mask)
    delay[slots] = torch.from_numpy(tau[members])
    masks[slots] = torch.from_numpy(truth_mask[members])
    if mode == "S2":
        # A true zero-support atom is used for unused oracle delay slots.
        keep = torch.zeros(8, dtype=torch.float64)
        keep[slots] = 1.
        return delay, old_mask * keep[:, None]
    if mode == "S3":
        nearest = abs(old_delay.numpy()[:, None] - tau[None]).argmin(axis=1)
        return old_delay, torch.from_numpy(truth_mask[nearest])
    if mode == "physical":
        delay[slots] = torch.tensor([selected[k]["physical_delay"] for k in members], dtype=torch.float64)
    return delay, masks


def fit(b, target, parts, observed, oracle):
    size = b.shape[-1]
    if size == 0:
        return torch.zeros(1800, dtype=torch.complex128), 0.
    gram = b.mH @ b
    if oracle:
        precision = torch.full((size,), 1e-6, dtype=torch.float64)
    else:
        scale = parts["scale"][0, 0, 0].double()
        normalized = observed / scale
        noise = normalized.abs().square().median() / np.log(2.)
        rho = (b.mH @ normalized).abs().square() / noise.clamp_min(1e-12)
        exponent = parts["log_precision"][0].double() + 2 * (np.log(9.) - torch.log1p(rho))
        precision = 1e-4 + torch.exp(30 * torch.tanh(exponent / 30))
    right = torch.linalg.solve(gram + torch.diag(precision).to(torch.complex128), b.mH)
    correction = b @ (right @ target)
    gain = float(torch.trace(gram @ (right @ right.mH)).real)
    return correction, gain


def score(small, restored, noise_cost, residual):
    s, j, e = [small[key].astype(np.complex128).reshape(-1, 8, 1800) for key in ("s", "j", "epsilon")]
    observed = states_from_components(s, j, e)
    output = restored.reshape(-1, 4, 8, 1800).astype(np.complex64)
    metrics = group_metrics(s, j, e, output)
    metrics.update(radar_metrics(s, observed, output, np.repeat(small["doppler_hz"], 3)))
    result = {key: value.reshape(len(small["s"]), 3) for key, value in metrics.items()}
    result["j_relative_residual"] = residual.sum(-1) / np.sum(abs(small["j"])**2, axis=(-2, -1))
    result["removed_noise_gain"] = noise_cost.sum(-1)
    return result


def run():
    torch.set_num_threads(2)
    started = now()
    status = {**provenance(__file__, WORK / "tools/vp_1103_replay_truth.py",
                           WORK / "tools/vp_1103_p2_smoothvp.py"),
              "stage": "RUNNING", "started_utc": started, "ended_utc": None,
              "pid": os.getpid(), "training_updates": 0, "device": "cpu"}
    atomic_json(OUT / "STATUS.json", status)
    data, binding = load("train")
    cp = WORK / "outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt"
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    selected = np.searchsorted(data["parent_id"], np.sort(payload["parent_ids"]))
    small = subset(data, selected)
    status.update(data={"train": binding}, parent_ids=small["parent_id"].tolist(), checkpoint_sha256=sha(cp))
    atomic_json(OUT / "STATUS.json", status)
    geometry = recover_selected(small, PRIVATE, binding)
    model = SmoothVPNet("RAW_IQ").cpu().eval()
    model.load_state_dict(payload["model"])
    shape = (16, 3, 4, 8, 1800)
    outputs = {name: np.zeros(shape, np.complex64) for name in METHODS}
    costs = {name: np.zeros((16, 3, 8)) for name in METHODS}
    residual = {name: np.zeros((16, 3, 8)) for name in METHODS}
    structure = {key: np.zeros((16, 3)) for key in
                 ("true_delay_count_max", "fraction_jammed_pulses_over8", "energy_beyond_top8",
                  "energy_phase_delay_beyond1200", "mean_abs_frequency_equivalent_shift")}
    synthesis_error = []
    with torch.inference_mode():
        for parent in range(16):
            for family in range(3):
                s, j, e = [small[k][parent, family].astype(np.complex128) for k in ("s", "j", "epsilon")]
                states = np.stack((e, s + e, j + e, s + j + e))
                parts = [model.factor(iq_tensor(y, "cpu")) for y in states]
                count, over8, energies, discarded, outside, freq_shift = [], [], [], [], [], []
                for pulse in range(8):
                    components = replay_components(float(small["target_delay_samples"][parent]),
                        float(small["doppler_hz"][parent]), family, geometry[parent, family, pulse])
                    if components:
                        canonical = sum(row["wave"] for row in components)
                        beta = np.vdot(canonical, j[pulse]) / np.vdot(canonical, canonical)
                        error = np.linalg.norm(j[pulse] - beta * canonical) / np.linalg.norm(j[pulse])
                        synthesis_error.append(float(error))
                        if error > 2e-7:
                            raise ValueError(f"Geometry-only replay reconstruction failed {error}")
                        count.append(len(components))
                        over8.append(len(components) > 8)
                        total = sum(row["energy"] for row in components)
                        energies.append(total)
                        discarded.append(sum(row["energy"] for row in components[8:]))
                        outside.append(sum(row["energy"] for row in components if row["phase_delay"] > 1200))
                        freq_shift.append(abs(components[0]["frequency_shift_hz"] / 50_000.))
                    for state in range(4):
                        p = {key: value[pulse:pulse + 1] for key, value in parts[state].items()}
                        obs = torch.from_numpy(states[state, pulse])
                        truth = torch.from_numpy(j[pulse])
                        for name in METHODS:
                            oracle = name != "S0_blind" and "blind" not in name
                            if name in ("S0_blind", "S1_oracle_beta"):
                                b = p["b"][0]
                            else:
                                mode = "S2" if name.startswith("S2") else "S3" if name.startswith("S3") else "S4"
                                if name == "S4_physical_delay_oracle":
                                    mode = "physical"
                                delay, masks = replacement(p, components, mode)
                                if name == "S4_range1200_oracle":
                                    delay = delay.clamp(max=1200.)
                                if name == "S4_all_delays_full_phase_oracle":
                                    delay = torch.tensor([row["phase_delay"] for row in components], dtype=torch.float64)
                                    masks = (torch.tensor(np.array([row["mask"] for row in components]))
                                             if components else torch.empty((0, 1800), dtype=torch.float64))
                                if name == "S4_smooth9_oracle":
                                    masks = torch.nn.functional.avg_pool1d(masks[None], 9, stride=1,
                                                                           padding=4, count_include_pad=False)[0]
                                b = dictionary(model, delay, masks, full_phase="full_phase" in name)
                            if name == "S0_blind":
                                correction = p["correction"][0]
                                cost = float(noise_gain(p)[0])
                            else:
                                correction, cost = fit(b, truth if oracle else obs, p, obs, oracle)
                            outputs[name][parent, family, state, pulse] = (obs - correction).numpy()
                            if state == 2:
                                costs[name][parent, family, pulse] = cost
                                residual[name][parent, family, pulse] = float((truth - correction).abs().square().sum())
                if count:
                    structure["true_delay_count_max"][parent, family] = max(count)
                    structure["fraction_jammed_pulses_over8"][parent, family] = np.mean(over8)
                    structure["energy_beyond_top8"][parent, family] = sum(discarded) / sum(energies)
                    structure["energy_phase_delay_beyond1200"][parent, family] = sum(outside) / sum(energies)
                    structure["mean_abs_frequency_equivalent_shift"][parent, family] = np.mean(freq_shift)
            atomic_json(OUT / "STATUS.json", {**status, "parents_scored": parent + 1, "updated_utc": now()})
    ids, jnr = small["parent_id"], frame_jnr(small)
    rows = {}
    raw_metrics = {}
    for name in METHODS:
        metrics = score(small, outputs[name], costs[name], residual[name])
        path = PRIVATE / f"{name}.npz"
        atomic_write(path, lambda stream: np.savez_compressed(stream, parent_id=ids, **metrics))
        raw_metrics[name] = metrics
        rows[name] = {"metrics": stratified(metrics, jnr), "private_metrics_sha256": sha(path),
                      "worst_cases": worst_cases(metrics, ids, jnr)}
    primary = rows["S4_oracle_beta"]["metrics"]
    adequate = all(primary[f"{family}/all"]["j_relative_residual"]["mean"] < .1
                   for family in ("direct", "repeat", "cyclic"))
    deltas = {name: stratified({"j_relative_residual_change":
                              raw_metrics[name]["j_relative_residual"] - raw_metrics["S1_oracle_beta"]["j_relative_residual"]}, jnr)
              for name in ("S2_oracle_beta", "S3_oracle_beta")}
    attribution = any(deltas[name]["global"]["j_relative_residual_change"]["parent_bootstrap_ci95"][1] < 0
                      for name in deltas)
    decision = "GO_V2_CPU_PREFLIGHT" if adequate and attribution else "STOP_REPRESENTATION_OR_ATTRIBUTION"
    summary = {"methods": rows, "structure": stratified(structure, jnr), "paired_substitution_changes": deltas,
               "stage2_route": decision, "stage2_expression_adequate": adequate,
               "stage2_localization_or_window_attribution": attribution,
               "geometry_reconstruction_relative_error_max": max(synthesis_error),
               "true_delay_count_max": float(structure["true_delay_count_max"].max()),
               "parents": 16, "training_updates": 0,
               "oracle_scope": "Truth geometry and support are used only in substitution controls. Oracle beta reads J. "
                               "Blind variants recompute rho on the replaced B while retaining learned log precision.",
               "primary_delay": "Phase-equivalent chirp delay = target_delay + replay_shift - (target_fd+jammer_fd)/50000.",
               "clean_pulses": "Oracle replacements have zero columns when no replay was generated. S0/S1 retain learned B."}
    atomic_json(OUT / "SUMMARY.json", summary)
    lines = ["# TRAIN16 字典头受控替换审计", "",
             "全部结果来自固定 16 个 TRAIN parent 的 RAW_IQ CPU 预检检查点。训练更新数为零。",
             "主规格采用计入目标与干扰频偏的等效 LFM 延迟。物理延迟、1200 点限制、有限模板边界与平滑窗均另有对照。",
             "Oracle 替换获知真实重放支撑，空重放脉冲使用零列。其收益包含该 oracle 信息。",
             "", "| 方法 | J 残差比 | direct | repeat | cyclic | 虚警 | Pd |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        m = row["metrics"]
        values = [m[key]["j_relative_residual"]["mean"] for key in ("global", "direct/all", "repeat/all", "cyclic/all")]
        values += [m["global"][key]["mean"] for key in ("frame_pfa_jammer", "pd")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4%}" for value in values) + " |")
    lines += ["", f"阶段 2 路由 **{decision}**。独立有效重放延迟的最大个数为 {int(summary['true_delay_count_max'])}。",
              "", "## 解释范围", "",
              "S2 同时修正学得的延迟值及初版 0–1200 点取期望范围。其改善不足以单独证明多峰均值塌缩是唯一原因。",
              "S4_range1200 与完整范围 S4 的差异量化原延迟范围限制。",
              "S4_physical_delay 与等效延迟对照量化频偏补偿，full_phase 对照量化有限参考支撑与 FFT 移位边界。",
              "all_delays_full_phase 对照继续放开 K=8 限制，用于核验真实离散重放分解。",
              "", "## S4 最差 parent", ""]
    write_cases(lines, rows["S4_oracle_beta"]["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "STATUS.json", {**status, "stage": "COMPLETE", "ended_utc": now(),
                                    "parents_scored": 16, "summary_sha256": sha(OUT / "SUMMARY.json")})


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "PROCESS.lock"):
        try:
            run()
        except Exception as exc:
            previous = read(OUT / "STATUS.json")
            atomic_json(OUT / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(OUT / "STATUS.json", {**previous, "stage": "FAILED", "ended_utc": now(), "error": repr(exc)})
            raise
