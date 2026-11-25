"""Produce the DEV projection decision and wait for the six-cell comparison."""
import argparse
import time

import numpy as np

from vp_1103_common import (DATA, FAMILIES, SEEDS, WORK, atomic_json, distribution, frame_jnr,
                           now, parent_mean, parent_roster, provenance, read, sha,
                           strata, stratified, worst_cases, write_cases)
from vp_1103_references import HISTORY, reference_metrics
from tools.durable_io import ProcessLock, atomic_write, process_alive

EVIDENCE = WORK / "evidence" / "outputs"
OUT = EVIDENCE / "vp_1103_decision_gate"
NAMES = {"p0": "p0_dechirp_tail_2x2", "a": "p1a_oracle_vp_audit",
         "b": "p1b_mismatch_drift_audit", "c": "p1c_structure_ambiguity_audit"}
P0_ARMS = ("P0-NoDechirp-Base", "P0-NoDechirp-Tail")
THRESHOLDS = {
    "oracle_global_pfa_decrease": .04, "oracle_high_jnr_pfa_decrease": .04,
    "oracle_damage_max": .0488, "proxy_drift_paired_upper_ci_max": 0.,
    "proxy_pfa_paired_upper_ci_max": .02, "adaptive_stress_gain_ceiling": 100.,
    "lag_min_gain_repeat_and_cyclic": .10, "lag_min_observed_rho": .30,
    "hankel_heavy_overlap": .60, "profile_min_conflict_rate": .15,
}


def private_metrics(stage, name, binding):
    path = WORK / "outputs" / NAMES[stage] / f"{name}.npz"
    if sha(path) != binding:
        raise ValueError(f"metric archive binding failed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verdict(condition):
    return "PASS" if condition else "FAIL"


def gate_one(a, unet, jnr):
    primary = a["primary_method"]
    baseline = (a["methods"][primary]["metrics"])
    oracle = private_metrics("a", primary, a["methods"][primary]["private_metrics_sha256"])
    differences = {key: oracle[key] - unet[key] for key in ("frame_pfa_jammer", "damage", "pd")}
    paired = stratified(differences, jnr)
    m = paired["global"]["frame_pfa_jammer"]
    hi = paired["jnr/>=30"]["frame_pfa_jammer"]
    goal = baseline["global"]["damage"]["mean"] < THRESHOLDS["oracle_damage_max"]
    comparisons = {
        "global_pfa": verdict(m["mean"] <= -THRESHOLDS["oracle_global_pfa_decrease"] and
                              m["parent_bootstrap_ci95"][1] < 0),
        "high_jnr_pfa": verdict(hi["mean"] <= -THRESHOLDS["oracle_high_jnr_pfa_decrease"] and
                                hi["parent_bootstrap_ci95"][1] < 0),
        "damage": verdict(goal and paired["global"]["damage"]["parent_bootstrap_ci95"][1] < 0),
    }
    return {"status": verdict(all(result == "PASS" for result in comparisons.values())),
            "checks": comparisons, "oracle_minus_unet_tail": paired, "primary_method": primary}


def gate_two(b, jnr):
    score = "relative_target_leakage_l2"
    local = b["perturbations"]
    error_means, gains = {}, {}
    checks = {}
    for axis in ("delay", "doppler"):
        for variant in ("tangent", "neighbors"):
            key = f"{axis}/{variant}"
            regimes = {}
            order = []
            for delta in (-1., -.5, .5, 1.):
                treatment = f"{axis}_{variant}_{delta:+g}"
                control = f"{axis}_zero_{delta:+g}"
                left = private_metrics("b", treatment, local[treatment]["private_metrics_sha256"])
                right = private_metrics("b", control, local[control]["private_metrics_sha256"])
                regimes[f"{delta:+g}"] = distribution((left[score] - right[score]).mean(axis=1))
            row0 = local[f"{axis}_{variant}_+0"]["metrics"]["global"][score]["mean"]
            for sign in (-1., 1.):
                by_scale = [local[f"{axis}_{variant}_{sign * step:+g}"]["metrics"]["global"][score]["mean"]
                            for step in (.5, 1.)]
                values = [abs(value - row0) for value in by_scale]
                order.append(float(np.log2(values[1] / values[0])) if min(values) > 1e-12 else None)
            error_means[key] = regimes
            gains[key] = order
            checks[key] = verdict(all(row["parent_bootstrap_ci95"][1] < 0 for row in regimes.values())
                                  and all(value is not None and value >= 1.5 for value in order))
    comparison = b["blind_proxy"]["isolated_minus_raw"]["global"]
    drift = comparison["drift_over_target_energy"]
    pfa = comparison["frame_pfa_jammer"]
    noise = b["noise_stress"]
    low = [row for row in noise if row["chi"] <= .01]
    checks["observed_isolation"] = verdict(drift["mean"] < 0 and
                                           drift["parent_bootstrap_ci95"][1] <= THRESHOLDS["proxy_drift_paired_upper_ci_max"]
                                           and pfa["parent_bootstrap_ci95"][1] <= THRESHOLDS["proxy_pfa_paired_upper_ci_max"])
    checks["adaptive_near_singularity"] = verdict(max(row["adaptive"] for row in low) <
                                                  THRESHOLDS["adaptive_stress_gain_ceiling"] and
                                                  max(row["adaptive"] for row in low) < max(row["fixed_1e-6"] for row in low))
    axes_pass = any(checks[f"delay/{mode}"] == checks[f"doppler/{mode}"] == "PASS"
                    for mode in ("tangent", "neighbors"))
    return {"status": verdict(axes_pass and checks["observed_isolation"] == "PASS" and
                              checks["adaptive_near_singularity"] == "PASS"),
            "checks": checks, "basis_minus_zero_on_relative_l2_leakage": error_means,
            "empirical_log2_leakage_increment_ratio": gains, "isolated_minus_raw": comparison,
            "stress_peak_gain": {"adaptive": max(row["adaptive"] for row in low),
                                 "fixed_1e-6": max(row["fixed_1e-6"] for row in low)},
            "note": "A 1.5 minimum log2 ratio allows sampling error around quadratic leakage scaling. "
                    "Finite perturbation and hard simulator edges limit an asymptotic interpretation."}


def gate_three_and_four(c):
    corr = c["correlations"]["global"]["peak_Y_active_windows"]
    repeat = c["stats"]["repeat/all"]["lag_J_minus_S_active_windows"]
    cyclic = c["stats"]["cyclic/all"]["lag_J_minus_S_active_windows"]
    separation = all(row["mean"] >= THRESHOLDS["lag_min_gain_repeat_and_cyclic"] and
                     row["parent_bootstrap_ci95"][0] > 0 for row in (repeat, cyclic))
    association = (corr["rho"] is not None and corr["rho"] >= THRESHOLDS["lag_min_observed_rho"] and
                   corr["parent_bootstrap_ci95"][0] > 0)
    profile = c["profile_likelihood"]["high_risk_parent_conflict_rate"]
    overlap = c["hankel_overlap"]["global"]
    return ({"status": verdict(separation and association),
             "repeat_separation": repeat, "cyclic_separation": cyclic, "observed_unet_pfa_association": corr},
            {"status": verdict(profile["mean"] is not None and profile["mean"] > THRESHOLDS["profile_min_conflict_rate"]),
             "high_risk_multimodal": profile, "hankel_overlap": overlap,
             "hankel_rule": "ABANDON" if overlap["overlap"] >= THRESHOLDS["hankel_heavy_overlap"] else "KEEP_AS_EVIDENCE"})


def phase_status(stage):
    path = EVIDENCE / NAMES[stage] / "STATUS.json"
    return read(path) if path.is_file() else {"stage": "PENDING"}


def enrich_p0(final, historical, hist_binding, jnr):
    summary_path = EVIDENCE / NAMES["p0"] / "SUMMARY.json"
    report_path = EVIDENCE / NAMES["p0"] / "REPORT.md"
    summary = read(summary_path)
    summary["aligned_historical_four_cell"] = historical
    summary["historical_reevaluation_binding"] = hist_binding
    if final:
        summary["six_cell_status"] = "COMPLETE"
        rotated = ("correction_original", "correction_tail")
        comparisons = {}
        for arm, old in zip(P0_ARMS, rotated, strict=True):
            difference = {}
            for seed in SEEDS:
                row = summary["new_arms"][arm][str(seed)]
                private = WORK / "outputs" / NAMES["p0"] / arm / str(seed) / "dev_parent_metrics.npz"
                if sha(private) != row["parent_metrics_sha256"]:
                    raise ValueError(f"P0 parent metrics changed: {arm}/{seed}")
                with np.load(private, allow_pickle=False) as archive:
                    difference[str(seed)] = {key: archive[key] - values
                                             for key, values in historical[old]["raw"][seed].items()}
            seed_avg = {key: np.mean([metrics[key] for metrics in difference.values()], axis=0)
                        for key in difference[str(SEEDS[0])]}
            comparisons[arm] = {"per_seed": {seed: stratified(values, jnr) for seed, values in difference.items()},
                                "seed_averaged": stratified(seed_avg, jnr)}
            if arm == P0_ARMS[0]:
                base_delta = seed_avg
            else:
                comparisons["dechirp_by_tail_interaction"] = stratified(
                    {key: value - base_delta[key] for key, value in seed_avg.items()}, jnr)
        summary["paired_raw_iq_minus_dechirp"] = comparisons
    summary["aligned_historical_four_cell"] = {
        arm: {k: v for k, v in item.items() if k != "raw"} for arm, item in historical.items()}
    atomic_json(summary_path, summary)
    marker = "## 冻结四格的同批分层对照"
    previous = report_path.read_text(encoding="utf8").split(marker, 1)[0].rstrip()
    lines = [previous, "", marker, "",
             "四组的历史检查点和 DEV 父场景均经 SHA-256 核验。重新评分得到逐类型、逐 JNR 与跨 seed 的 parent 聚类区间，见 `SUMMARY.json`。",
             "", "| 历史方法 | 虚警均值 | parent 95% 区间 | Pd 均值 | 目标损伤 |",
             "|---|---:|---:|---:|---:|"]
    for arm, item in historical.items():
        m = item["seed_averaged"]["global"]
        fa = m["frame_pfa_jammer"]
        lo, hi = fa["parent_bootstrap_ci95"]
        lines.append(f"| {arm} | {fa['mean']:.3%} | {lo:.3%}–{hi:.3%} | {m['pd']['mean']:.3%} | {m['damage']['mean']:.6g} |")
    if final:
        lines += ["", "新四格的逐 seed 配对坐标差和坐标×尾部损失交互量见 `SUMMARY.json`。"]
    atomic_write(report_path, lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


def run():
    states = {stage: phase_status(stage) for stage in NAMES}
    completed = {stage: state["stage"] == "COMPLETE" for stage, state in states.items()}
    status = {stage: state["stage"] for stage, state in states.items()}
    path = DATA / "dev.npz"
    with np.load(path, allow_pickle=False) as archive:
        ids, jnr = parent_roster(archive, 256), frame_jnr(archive)
    historical, bindings = {}, {}
    for arm in HISTORY:
        rows, bindings[arm] = reference_metrics(arm, ids)
        mean = {key: np.mean([row[key] for row in rows.values()], axis=0) for key in rows[SEEDS[0]]}
        historical[arm] = {"raw": rows, "per_seed": {str(seed): stratified(values, jnr)
                                                    for seed, values in rows.items()},
                           "seed_averaged": stratified(mean, jnr),
                           "worst_cases": worst_cases(mean, ids, jnr),
                           "seed_average_raw": mean}
    unet = historical["unet_tail"]["seed_average_raw"]
    audit = {stage: read(EVIDENCE / NAMES[stage] / "SUMMARY.json")
             for stage in ("a", "b", "c") if completed[stage]}
    gates = {str(index): {"status": "PENDING"} for index in range(1, 5)}
    if "a" in audit:
        gates["1"] = gate_one(audit["a"], unet, jnr)
    if "b" in audit:
        gates["2"] = gate_two(audit["b"], jnr)
    if "c" in audit:
        gates["3"], gates["4"] = gate_three_and_four(audit["c"])
    coordinate = read(EVIDENCE / NAMES["p0"] / "SUMMARY.json")["coordinates"]
    gates_to_train = [gates[index]["status"] for index in ("1", "2")]
    if "FAIL" in gates_to_train:
        route = "NO_GO"
    elif all(completed.values()) and gates_to_train == ["PASS", "PASS"]:
        route = "AWAIT_USER_CONFIRMATION"
    else:
        route = "WAITING_FOR_RECEIPTS"
    if completed["p0"]:
        for value in historical.values():
            del value["seed_average_raw"]
        enrich_p0(True, historical, bindings, jnr)
    lines = ["# ISRJ_2 · P0/P1 决策闸门", "",
             f"**P2 路由 {route}。** P2 训练状态为未启动。", "",
             "| 阶段 | 状态 |", "|---|---|"]
    lines.extend(f"| {NAMES[key]} | {value} |" for key, value in status.items())
    lines += ["", f"P0 坐标结论 **{coordinate}**。逐种子、逐类型和逐 JNR 的历史对照随本报告保存。",
              "", "| 闸门 | 判定 |", "|---|---|",
              f"| 1 · 已知几何上限 | {gates['1']['status']} |",
              f"| 2 · 结构失配与漂移 | {gates['2']['status']} |",
              f"| 3 · 长程重复特征 | {gates['3']['status']} |",
              f"| 4 · 条件 Flow 支线 | {gates['4']['status']} |", ""]
    if "a" in audit:
        a = gates["1"]["oracle_minus_unet_tail"]
        lines += ["## 门槛 1", ""]
        for label, key in [("整体", "global"), ("JNR ≥30 dB", "jnr/>=30"), ("JNR ≥35 dB", "jnr/>=35")]:
            fa = a[key]["frame_pfa_jammer"]
            lo, hi = [v * 100 for v in fa["parent_bootstrap_ci95"]]
            lines.append(f"- {label}虚警差 {fa['mean'] * 100:+.3f} 个百分点，95% 区间 [{lo:+.3f}, {hi:+.3f}]，{fa['parents']} 个 parent。")
        lines += [f"- 整体目标损伤差 {a['global']['damage']['mean']:+.6g}。",
                  "- 通过要求包括整体和 ≥30 dB 虚警各下降至少 4 个百分点，且配对区间上界低于零。",
                  "- Oracle 已掌握几何和受扰脉冲指示。该结果用于判定有无求解余量。", ""]
    if "b" in audit:
        metrics = audit["b"]["blind_proxy"]
        raw, isolated = [metrics[key]["metrics"]["global"] for key in ("raw", "isolated")]
        lines += ["## 门槛 2", "",
                  f"- 估计输入由原始观测改为保护正交补时，整体虚警由 {raw['frame_pfa_jammer']['mean']:.3%} 变为 {isolated['frame_pfa_jammer']['mean']:.3%}，漂移能量/目标能量由 {raw['drift_over_target_energy']['mean']:.6g} 变为 {isolated['drift_over_target_energy']['mean']:.6g}。",
                  f"- 近共线压力测试的峰值噪声增益，自适应为 {gates['2']['stress_peak_gain']['adaptive']:.4g}，固定弱正则为 {gates['2']['stress_peak_gain']['fixed_1e-6']:.4g}。",
                  "- 局部容错要求同一保护基在时延与频偏轴的 ±0.5、±1 扰动均降低泄漏，并达到预定的泄漏增量阶数。配对区间和各分项判定见 JSON。", ""]
    if "c" in audit:
        g3, g4 = gates["3"], gates["4"]
        rho = g3["observed_unet_pfa_association"]["rho"]
        conflict = g4["high_risk_multimodal"]
        lines += ["## 门槛 3 与 4", "",
                  f"- repeat 与 cyclic 的 J−S 相关峰均差分别为 {g3['repeat_separation']['mean']:.4f} 和 {g3['cyclic_separation']['mean']:.4f}。观测峰与 U-Net+Tail 虚警的 Spearman ρ 为 {rho:.4f}。",
                  f"- 强干扰可辨识子样本的波形歧义比例为 {conflict['mean']:.2%}，共 {conflict['parents']} 个 parent。",
                  f"- Hankel 重叠为 {g4['hankel_overlap']['overlap']:.3f}，判定为 {g4['hankel_rule']}。", ""]
    if route == "NO_GO":
        lines += ["## 决策", "", "暂停端到端原型。保留闭式求解作为已知几何对照，下一次评审优先看几何估计、伪峰误保护和算子漂移的证据。", ""]
    decision_lines = {
        "a": ("## 门槛 1", f"判定 **{gates['1']['status']}**。配对差和区间见总闸门报告。"),
        "b": ("## 门槛 2", f"判定 **{gates['2']['status']}**。分项判定见总闸门报告。"),
        "c": ("## 门槛 3 与 4",
              f"长程特征 **{gates['3']['status']}**，条件 Flow **{gates['4']['status']}**。")
    }
    for stage, (heading, decision) in decision_lines.items():
        if stage in audit:
            path = EVIDENCE / NAMES[stage] / "REPORT.md"
            current = path.read_text(encoding="utf8")
            begin = current.index(heading + "\n\n") + len(heading) + 2
            end = current.find("\n", begin)
            revised = current[:begin] + decision + current[end:]
            if current != revised:
                atomic_write(path, lambda stream: stream.write(revised.encode("utf8")))
    result = {
        "route": route, "p2_training_started": False, "phase_status": status, "p0_coordinates": coordinate,
        "thresholds": THRESHOLDS, "gates": gates, "historical_binding": bindings,
        "historical_four_cell": {arm: {key: value for key, value in row.items()
                                       if key not in ("raw", "seed_average_raw")} for arm, row in historical.items()},
        "audit_summary_sha256": {stage: sha(EVIDENCE / NAMES[stage] / "SUMMARY.json") for stage in audit},
    }
    atomic_json(OUT / "SUMMARY.json", result)
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    return result, states


def monitor(wait):
    started = now()
    source = provenance(__file__, WORK / "tools/vp_1103_references.py",
                        WORK / "tools/vp_1103_reference_readonly.py")
    last_signature = None
    while True:
        states = {stage: phase_status(stage) for stage in NAMES}
        signature = tuple(state["stage"] for state in states.values())
        if signature != last_signature:
            result, states = run()
            last_signature = tuple(state["stage"] for state in states.values())
        stage = "COMPLETE" if all(state["stage"] == "COMPLETE" for state in states.values()) else "PROVISIONAL"
        failed = any(state["stage"] == "FAILED" for state in states.values())
        orphans = [name for name, state in states.items()
                   if state.get("pid") and state["stage"] not in ("COMPLETE", "FAILED")
                   and not process_alive(state["pid"])]
        if failed or orphans:
            stage = "BLOCKED"
        atomic_json(OUT / "STATUS.json", {
            **source, "started_utc": started, "updated_utc": now(),
            "ended_utc": now() if stage != "PROVISIONAL" else None,
            "stage": stage, "route": result["route"], "phase_status": result["phase_status"],
            "unowned_processes": orphans,
            "dev_sha256": states.get("a", {}).get("data", {}).get("dev", {}).get("sha256"),
            "p0_progress": {key: states["p0"].get(key) for key in
                            ("arm", "seed", "last_committed_epoch", "checkpoint_sha256", "updated_utc")},
            "seeds": list(SEEDS), "p0_micro_batch": 8, "p1_micro_batch": None,
        })
        if not wait or stage != "PROVISIONAL":
            return
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    with ProcessLock(WORK / "outputs" / "vp_1103_decision_gate" / "PROCESS.lock"):
        monitor(parser.parse_args().wait)
