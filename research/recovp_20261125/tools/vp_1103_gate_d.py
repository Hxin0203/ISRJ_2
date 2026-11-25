"""Evaluate the Replay-First gates from sealed DEV receipts."""
import argparse
import time

import numpy as np

from vp_1103_common import (DATA, DEV_HASH, WORK, atomic_json, distribution, frame_jnr,
                           now, parent_mean, parent_roster, provenance, read, sha, validate_role)
from vp_1103_p1d import NAMES
from vp_1103_references import reference_metrics
from tools.durable_io import ProcessLock, atomic_write, process_alive

EVIDENCE = WORK / "evidence" / "outputs"
OUT = EVIDENCE / "vp_1103_gate_d"
THRESHOLDS = {"d1_pfa_max": .06, "d1_high_jnr_pfa_gain_min": .04, "d1_damage_max": .025,
              "d1_nojam_nmse_max": 5e-4, "d2_drift_max_exclusive": .5,
              "d3_pfa_delta_upper_max": .005, "d3_low_chi_exclusive": .1,
              "d3_low_chi_min_parents": 16}
LAST_STATUS = {}


def verdict(condition):
    return "PASS" if condition else "FAIL"


def status(folder):
    path = EVIDENCE / folder / "STATUS.json"
    for attempt in range(12):
        try:
            current = read(path)
            LAST_STATUS[folder] = current
            return current
        except FileNotFoundError:
            return {"stage": "PENDING"}
        except PermissionError:
            if attempt == 11:
                if folder in LAST_STATUS:
                    return LAST_STATUS[folder]
                raise
            time.sleep(.15)


def method(stage, name, row):
    target = WORK / "outputs" / NAMES[stage] / f"{name}.npz"
    if sha(target) != row["private_metrics_sha256"]:
        raise ValueError(f"Archive hash mismatch for {stage}/{name}")
    with np.load(target, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if any(value.shape != (256, 3) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"Invalid parent-by-family metric array in {stage}/{name}")
    return values


def d1(audit, unet, jnr):
    cases = {}
    for name, row in audit["methods"].items():
        metrics = method("d1", name, row)
        pfa = metrics["frame_pfa_jammer"]
        damage = metrics["damage"]
        delta_pfa = pfa - unet["frame_pfa_jammer"]
        paired = distribution(delta_pfa.mean(axis=1))
        high = distribution(parent_mean(delta_pfa, jnr >= 30))
        paired_damage = distribution((damage - unet["damage"]).mean(axis=1))
        levels = {key: row["metrics"]["global"][key]["mean"]
                  for key in ("frame_pfa_jammer", "damage", "no_jammer_pooled_nmse",
                              "removed_noise_gain_frobenius_squared", "pd")}
        checks = {
            "global_pfa": (pfa.mean() <= THRESHOLDS["d1_pfa_max"] and paired["parent_bootstrap_ci95"][1] < 0.),
            "high_jnr_pfa": high["mean"] <= -THRESHOLDS["d1_high_jnr_pfa_gain_min"],
            "damage": (damage.mean() <= THRESHOLDS["d1_damage_max"] and paired_damage["parent_bootstrap_ci95"][1] < 0.),
            "no_jammer_keep": levels["no_jammer_pooled_nmse"] <= THRESHOLDS["d1_nojam_nmse_max"],
        }
        passed = row["gate_eligible"] and all(checks.values())
        cases[name] = {"status": verdict(passed), "eligible": row["gate_eligible"],
                       "checks": {key: verdict(value) for key, value in checks.items()},
                       "levels": levels, "paired_pfa": paired, "paired_pfa_ge30": high,
                       "paired_damage": paired_damage, "k": row["k"], "precision": row["rule"],
                       "center": row["center"]}
    passed = [name for name, row in cases.items() if row["status"] == "PASS"]
    nearest = sorted((name for name, row in cases.items() if row["eligible"]),
                     key=lambda name: (-sum(x == "PASS" for x in cases[name]["checks"].values()),
                                       cases[name]["levels"]["frame_pfa_jammer"], name))[:5]
    return {"status": verdict(bool(passed)), "passing_configurations": passed,
            "strongest_eligible_descriptive": nearest, "cases": cases,
            "scope": "Conditional replay-anchor geometry. No target protection or truth jam switching. "
                     "Reported intervals are pointwise DEV screening intervals across a configuration scan."}


def d2_and_d3(audit):
    rows = audit["methods"]
    baseline = read(EVIDENCE / "p1b_mismatch_drift_audit" / "SUMMARY.json")["blind_proxy"]
    ref_drift = baseline["isolated"]["metrics"]["global"]["drift_over_target_energy"]["mean"]
    ref_pfa = baseline["raw"]["metrics"]["global"]["frame_pfa_jammer"]["mean"]
    reproduced = rows["target_first_hard"]["metrics"]["global"]
    old_pfa = baseline["isolated"]["metrics"]["global"]["frame_pfa_jammer"]["mean"]
    reproduced_pfa = reproduced["frame_pfa_jammer"]["mean"]
    if abs(reproduced_pfa - old_pfa) > 1e-12 or abs(reproduced["drift_over_target_energy"]["mean"] - ref_drift) > 1e-4:
        raise ValueError("Historical hard proxy identity did not reproduce")
    primary = audit["primary"]
    smooth = method("d2", primary, rows[primary])
    guard = method("d2", "replay_guard", rows["replay_guard"])
    drift = float(smooth["drift_over_target_energy"].mean())
    pfa = float(smooth["frame_pfa_jammer"].mean())
    criteria = {"drift": drift < THRESHOLDS["d2_drift_max_exclusive"] and drift < .1 * ref_drift,
                "pfa": pfa < ref_pfa}
    low = smooth["min_blind_chi_background"] < THRESHOLDS["d3_low_chi_exclusive"]
    if not np.array_equal(smooth["min_blind_chi_background"], guard["min_blind_chi_background"]):
        raise ValueError("Guard and smooth χ geometry differ")
    pfa_diff = distribution((guard["frame_pfa_jammer"] - smooth["frame_pfa_jammer"]).mean(axis=1))
    damage_diff = distribution(parent_mean(guard["damage"] - smooth["damage"], low))
    pfa_safe = pfa_diff["parent_bootstrap_ci95"][1] <= THRESHOLDS["d3_pfa_delta_upper_max"]
    enough = damage_diff["parents"] >= THRESHOLDS["d3_low_chi_min_parents"]
    improved = enough and damage_diff["parent_bootstrap_ci95"][1] < 0.
    controlled = {key: {metric: row["metrics"]["global"][metric]["mean"] for metric in
                        ("drift_over_target_energy", "frame_pfa_jammer", "damage", "no_jammer_pooled_nmse")}
                  for key, row in rows.items() if row.get("conditional_geometry")}
    return ({"status": verdict(all(criteria.values())),
             "checks": {key: verdict(value) for key, value in criteria.items()},
             "historical_raw_pfa": ref_pfa, "historical_isolated_pfa": old_pfa,
             "historical_isolated_drift": ref_drift, "smooth_pfa": pfa, "smooth_drift": drift,
             "controlled_fallback": controlled},
            {"status": verdict(pfa_safe and improved),
             "guard_minus_smooth_pfa": pfa_diff, "guard_minus_smooth_damage_low_chi": damage_diff,
             "pfa_noninferiority": verdict(pfa_safe), "sufficient_low_chi": enough,
             "include_m4": bool(pfa_safe and improved)})


def append_verdict(stage, verdict_text):
    target = EVIDENCE / NAMES[stage] / "REPORT.md"
    content = target.read_text(encoding="utf8")
    revised = content.replace("待总控脚本判定。", verdict_text)
    if revised != content:
        atomic_write(target, lambda stream: stream.write(revised.encode("utf8")))


def append_failure_diagnostics(stage, audit, names, ids):
    target = EVIDENCE / NAMES[stage] / "REPORT.md"
    marker = "## 闸门相关最差 parent 的量化分解"
    original = target.read_text(encoding="utf8").split(marker, 1)[0].rstrip()
    lines = [original, "", marker, "",
             "虚警与损伤各列出五个最差 parent。分解量在同一 parent 的三种干扰内平均。", ""]
    for name in names:
        row = audit["methods"][name]
        private = method(stage, name, row)
        lines += [f"### {name}", ""]
        lines += ["| 排序 | parent | 损伤 | 泄漏比 | 漂移比 | 残峰/噪声 | 噪声增益 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for metric, cases in row["worst_cases"].items():
            for case in cases:
                index = np.searchsorted(ids, case["parent_id"])
                fields = ("damage", "leak_over_target_energy", "drift_over_target_energy",
                          "mf_residual_peak_over_noise_rms", "removed_noise_gain_frobenius_squared")
                values = " | ".join(f"{private[key][index].mean():.6g}" for key in fields)
                lines.append(f"| {metric} | {case['parent_id']} | {values} |")
        lines.append("")
    revised = "\n".join(lines) + "\n"
    atomic_write(target, lambda stream: stream.write(revised.encode("utf8")))


def evaluate(states):
    path, data = validate_role("dev")
    with np.load(path, allow_pickle=False) as archive:
        ids, jnr = parent_roster(archive, 256), frame_jnr(archive)
    frozen, ref = reference_metrics("unet_tail", ids)
    baseline = {key: np.mean([rows[key] for rows in frozen.values()], axis=0)
                for key in frozen[17]}
    summaries, digest = {}, {}
    for stage, folder in NAMES.items():
        if states[stage]["stage"] == "COMPLETE":
            source = EVIDENCE / folder / "SUMMARY.json"
            digest[stage] = sha(source)
            summaries[stage] = read(source)
            if digest[stage] != states[stage]["summary_sha256"] or summaries[stage]["dev_sha256"] != DEV_HASH:
                raise ValueError(f"{stage} receipt identity mismatch")
    gates = {name: {"status": "PENDING"} for name in ("D1", "D2", "D3")}
    if "d1" in summaries:
        gates["D1"] = d1(summaries["d1"], baseline, jnr)
        append_verdict("d1", f"判定 **{gates['D1']['status']}**。逐配置条件和配对区间见 Gate-D。")
        strongest = gates["D1"]["passing_configurations"] or gates["D1"]["strongest_eligible_descriptive"]
        append_failure_diagnostics("d1", summaries["d1"], strongest[:1], ids)
    if "d2" in summaries:
        gates["D2"], gates["D3"] = d2_and_d3(summaries["d2"])
        append_verdict("d2", f"平滑重放 **{gates['D2']['status']}**，软保护附加项 **{gates['D3']['status']}**。")
        append_failure_diagnostics("d2", summaries["d2"], [summaries["d2"]["primary"], "replay_guard"], ids)
    auxiliary = summaries.get("d3", {}).get("auxiliary_feature", "PENDING")
    coordinate = read(EVIDENCE / "p0_dechirp_tail_2x2/SUMMARY.json")["coordinates"]
    p0_path = EVIDENCE / "p0_dechirp_tail_2x2/SUMMARY.json"
    p0_ready = (states["p0"]["stage"] == "COMPLETE" and
                coordinate in ("RAW_IQ", "DECHIRP") and read(p0_path).get("six_cell_status") == "COMPLETE")
    bridge_complete = len(summaries) == len(NAMES)
    if any(gates[name]["status"] == "FAIL" for name in ("D1", "D2")):
        route = "NO_GO"
    elif not bridge_complete or not p0_ready:
        route = "WAITING_FOR_RECEIPTS"
    elif gates["D3"]["include_m4"]:
        route = "AWAIT_USER_CONFIRMATION"
    else:
        route = "AWAIT_USER_CONFIRMATION_M3_ONLY"
    phase_status = {key: item["stage"] for key, item in states.items()}
    lines = ["# ISRJ_2 · Gate-D 权衡与路由", "",
             f"**P2-v2 路由 {route}。** P2-v2 尚未启动。",
             f"P0 坐标 **{coordinate}**。P0 六格已对齐 **{'是' if p0_ready else '否'}**。", "",
             "| 闸门 | 判定 |", "|---|---|"]
    for key, item in gates.items():
        lines.append(f"| {key} | {item['status']} |")
    lines += ["", f"含噪 Hankel 辅助筛查 **{auxiliary}**。", "",
              "本轮继续关闭条件 Flow/SDE、独立长程滞后相关主线和目标优先硬阻塞原型。目标优先硬代理保留为对照。", ""]
    if "d1" in summaries:
        row = gates["D1"]
        lines += ["## Gate D1 · 已通过或最接近门槛的配置", "",
                  "| 配置 | 全局虚警 | ≥30 dB 相对 U-Net+Tail 差值 | 损伤 | 无扰 NMSE | 判定 |",
                  "|---|---:|---:|---:|---:|---|"]
        for name in row["passing_configurations"] or row["strongest_eligible_descriptive"]:
            cell = row["cases"][name]
            m = cell["levels"]
            lines.append(f"| {name} | {m['frame_pfa_jammer']:.3%} | {cell['paired_pfa_ge30']['mean']*100:+.3f} pp | "
                         f"{m['damage']:.6g} | {m['no_jammer_pooled_nmse']:.6g} | {cell['status']} |")
        lines += ["", "各配置配对区间为 DEV 点态区间。零失配名义条目为参考，满足条件的条目仍依赖提供的模拟几何锚点。", ""]
        lines += ["| 平滑邻域维度 | 虚警 | ≥35 dB 虚警 | 损伤 | 保持 NMSE | 噪声增益 |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for name, cell in summaries["d1"]["methods"].items():
            if cell["axis"] == "neighborhood" and cell["rule"] == "smooth":
                m = cell["metrics"]["global"]
                fa35 = cell["metrics"]["jnr/>=35"]["frame_pfa_jammer"]["mean"]
                lines.append(f"| {cell['k']} | {m['frame_pfa_jammer']['mean']:.3%} | {fa35:.3%} | "
                             f"{m['damage']['mean']:.6g} | {m['no_jammer_pooled_nmse']['mean']:.6g} | "
                             f"{m['removed_noise_gain_frobenius_squared']['mean']:.6g} |")
        lines.append("")
    if "d2" in summaries:
        row, guard = gates["D2"], gates["D3"]
        lines += ["## Gate D2 与 D3 · 带噪代理", "",
                  f"- 平滑重放优先的漂移能量比为 {row['smooth_drift']:.6g}，历史隔离硬代理为 {row['historical_isolated_drift']:.6g}。",
                  f"- 平滑重放优先的全局虚警为 {row['smooth_pfa']:.3%}，历史原始硬代理为 {row['historical_raw_pfa']:.3%}。"]
        for name, metrics in row["controlled_fallback"].items():
            lines.append(f"- {name} 的漂移能量比为 {metrics['drift_over_target_energy']:.6g}，虚警为 {metrics['frame_pfa_jammer']:.3%}。")
        lo, hi = guard["guard_minus_smooth_pfa"]["parent_bootstrap_ci95"]
        lines += [f"- 软切空间叠加后的虚警差 95% 区间为 [{lo*100:+.3f}, {hi*100:+.3f}] pp。"
                  f" χ <0.1 的配对 parent 有 {guard['guard_minus_smooth_damage_low_chi']['parents']} 个。", ""]
    if "d3" in summaries:
        hankel = summaries["d3"]["comparisons"]["jammed_vs_unjammed"]
        jammer_vs_target = summaries["d3"]["comparisons"]["jammer_vs_single_target"]["global"]["roc"]
        mixed_vs_target = summaries["d3"]["comparisons"]["jammed_target_vs_single_target"]["global"]["roc"]
        lines += ["## 含噪 Hankel", "",
                  f"整体 AUC 为 {hankel['global']['roc']['auc']:.3f}，≤0 dB 为 {hankel['jnr/<=0']['roc']['auc']:.3f}，"
                  f"≥35 dB 为 {hankel['jnr/>=35']['roc']['auc']:.3f}。",
                  f"以高谱比指示干扰方向时，J+E 对 S+E 的 AUC 为 {jammer_vs_target['reverse_direction_auc']:.3f}，"
                  f"S+J+E 对 S+E 的 AUC 为 {mixed_vs_target['reverse_direction_auc']:.3f}。", ""]
    result = {"route": route, "p2_training_started": False, "p0_coordinates": coordinate,
              "p0_six_cell_verified": p0_ready, "bridge_complete": bridge_complete,
              "phase_status": phase_status, "gates": gates, "hankel": auxiliary,
              "thresholds": THRESHOLDS, "source_audit_sha256": digest, "unet_tail_binding": ref, "dev": data}
    atomic_json(OUT / "SUMMARY.json", result)
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    return result


def monitor(wait):
    started = now()
    source = provenance(__file__, WORK / "tools/vp_1103_p1d.py", WORK / "tools/vp_1103_p1d_replay.py",
                        WORK / "tools/vp_1103_p1d3.py")
    signature = None
    while True:
        try:
            states = {key: status(folder) for key, folder in {**NAMES, "p0": "p0_dechirp_tail_2x2"}.items()}
        except PermissionError:
            if not wait:
                raise
            time.sleep(1)
            continue
        old_gate = status("vp_1103_decision_gate")["stage"]
        new = (*[state["stage"] for state in states.values()], old_gate)
        if new != signature:
            result = evaluate(states)
            signature = new
        stopped = [key for key, state in states.items() if
                   state["stage"] == "FAILED" or (state["stage"] in ("RUNNING", "TRAINING") and
                                                 not process_alive(state.get("pid")))]
        stage = "COMPLETE" if result["bridge_complete"] and result["p0_six_cell_verified"] else "PROVISIONAL"
        if stopped:
            stage = "BLOCKED"
        atomic_json(OUT / "STATUS.json", {**source, "started_utc": started, "updated_utc": now(),
                    "ended_utc": now() if stage != "PROVISIONAL" else None, "stage": stage,
                    "route": result["route"], "phase_status": result["phase_status"],
                    "blocked": stopped, "dev_sha256": DEV_HASH, "micro_batch": None, "training_seeds": [],
                    "p0_progress": {key: states["p0"].get(key) for key in
                                    ("arm", "seed", "last_committed_epoch", "updated_utc")}})
        if not wait or stage != "PROVISIONAL":
            return
        time.sleep(45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    with ProcessLock(WORK / "outputs" / "vp_1103_gate_d" / "PROCESS.lock"):
        monitor(parser.parse_args().wait)
