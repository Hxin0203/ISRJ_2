"""Read-only TRAIN16 diagnostics and readable review of completed v2 receipts."""
import numpy as np

from vp_1103_common import (WORK, FAMILIES, atomic_json, frame_jnr, now,
                           provenance, read, sha, stratified)
from vp_1103_p2_preflight import load, subset
from tools.durable_io import atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_v2_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_v2_preflight"
FIELDS = ("frame_pfa_jammer", "pd", "damage", "blind_j_relative_residual",
          "learned_b_oracle_beta_relative_residual", "drift_over_target_energy",
          "leak_over_target_energy", "no_jammer_pooled_nmse")


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def run():
    started = now()
    state = read(OUT / "STATUS.json")
    if state["stage"] != "COMPLETE" or not (OUT / "FINAL_SEAL.json").exists():
        raise ValueError("Both CPU coordinates and the final P0 binding must be complete")
    summary = read(OUT / "SUMMARY.json")
    seal = read(OUT / "FINAL_SEAL.json")
    if seal["final_summary_sha256"] != sha(OUT / "SUMMARY.json"):
        raise ValueError("Final summary seal mismatch")
    anchors = read(OUT / "ANCHOR_RECALL.json")
    if not anchors["complete_both_coordinates"]:
        raise ValueError("Both post-fit anchor audits are required")
    data, data_receipt = load("train")
    ids = np.asarray(summary["parent_ids"])
    small = subset(data, np.searchsorted(data["parent_id"], ids))
    jnr = frame_jnr(small)
    result = {}
    lines = ["# v2 固定末步定位审阅", "",
             "全部结果来自同一批 16 个 TRAIN parent、seed 17、第 3,840 次更新。"
             "两个坐标均保留固定预算，正式 P2 GPU 训练保持暂停。", "",
             "## 分层对照", "",
             "表中残差为能量比；虚警为 parent 内三类干扰平均后再跨 parent 平均。"
             "95% 区间按 parent 聚类重采样，包含零事件的区间不能解释为零风险。", ""]
    for coord, row in summary["coordinates"].items():
        raw_path = PRIVATE / coord / "metrics.npz"
        anchor_path = PRIVATE / coord / "anchor_recall.npz"
        old_path = WORK / "outputs/p2_smoothvp_preflight" / coord / "fit_diagnosis.npz"
        if sha(raw_path) != row["private_metrics_sha256"]:
            raise ValueError("Post-fit metrics binding mismatch")
        if sha(anchor_path) != anchors["coordinates"][coord]["private_metrics_sha256"]:
            raise ValueError("Anchor metrics binding mismatch")
        raw, ar, old = [archive(path) for path in (raw_path, anchor_path, old_path)]
        if not np.array_equal(raw["parent_id"], ids):
            raise ValueError("Post-fit parent order changed")
        old_order = np.searchsorted(old["parent_id"], ids)
        anchor_order = np.searchsorted(ar["parent_id"], ids)
        if not np.array_equal(old["parent_id"][old_order], ids) or not np.array_equal(ar["parent_id"][anchor_order], ids):
            raise ValueError("Review rosters differ")
        paired = stratified({key: raw[key] - old[key][old_order] for key in FIELDS}, jnr)
        baseline = stratified({key: old[key][old_order] for key in FIELDS}, jnr)
        cases = {}
        for criterion, worst in row["worst_cases"].items():
            cases[criterion] = []
            for item in worst:
                index = int(np.flatnonzero(ids == item["parent_id"])[0])
                detail = {"parent_id": item["parent_id"], "score": item["score"], "families": {}}
                for family, name in enumerate(FAMILIES):
                    detail["families"][name] = {
                        **{key: float(raw[key][index, family]) for key in FIELDS},
                        "jnr_db": float(jnr[index, family]),
                        "anchor_energy_recall_2_samples": float(ar["anchor_component_energy_recall_2_samples"]
                                                                [anchor_order[index], family]),
                        "anchor_nearest_error_samples": float(ar["anchor_weighted_nearest_error_samples"]
                                                                [anchor_order[index], family]),
                        "outside_strong_support": float(raw["J_energy_outside_strong_atom_regions"][index, family])}
                cases[criterion].append(detail)
        trace_path = PRIVATE / coord / "TRAJECTORY.json"
        trace = read(trace_path)
        total_loss = np.array([record["total"] for record in trace])
        if len(total_loss) != 3840 or not np.isfinite(total_loss).all():
            raise ValueError("Fixed CPU optimization budget or trajectory is incomplete")
        pass_means = total_loss.reshape(80, 48).mean(1)
        convergence = {"passes": 80, "updates_per_pass": 48,
                       "stage1_last_pass_loss": float(pass_means[15]),
                       "stage2_first_pass_loss": float(pass_means[16]),
                       "last_10_pass_loss": pass_means[-10:].tolist(),
                       "last_10_pass_slope": float(np.polyfit(np.arange(10), pass_means[-10:], 1)[0]),
                       "trajectory_sha256": sha(trace_path),
                       "scope": "Mean pre-update loss of all 48 records in each pass, not checkpoint selection"}
        result[coord] = {"paired_v2_minus_v1": paired, "baseline_v1": baseline,
                         "trajectory_statistics": convergence,
                         "failure_cases": cases, "baseline_metrics_sha256": sha(old_path),
                         "metrics_sha256": sha(raw_path), "anchor_metrics_sha256": sha(anchor_path)}
        lines += [f"### {coord}", "",
                  "| 分层 | parent | v1 虚警 | v2 虚警 | v2 Pd | Oracle-β 残差 | 盲残差 | 漂移比 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for stratum in ("global", "direct/all", "repeat/all", "cyclic/all", "jnr/<=0", "jnr/5-25", "jnr/>=30", "jnr/>=35",
                        "direct/>=30", "repeat/>=30", "cyclic/>=30", "direct/>=35", "repeat/>=35", "cyclic/>=35"):
            values = row["metrics"][stratum]
            lines.append(f"| {stratum} | {values['pd']['parents']} | "
                         f"{baseline[stratum]['frame_pfa_jammer']['mean']:.3%} | "
                         f"{values['frame_pfa_jammer']['mean']:.3%} | {values['pd']['mean']:.3%} | "
                         f"{values['learned_b_oracle_beta_relative_residual']['mean']:.3%} | "
                         f"{values['blind_j_relative_residual']['mean']:.3%} | "
                         f"{values['drift_over_target_energy']['mean']:.5g} |")
        lines += ["", "| 全局 v2−v1 | 均值差 | 配对 parent 95% 区间 |", "|---|---:|---:|"]
        for key in FIELDS:
            d = paired["global"][key]
            lo, hi = d["parent_bootstrap_ci95"]
            lines.append(f"| {key} | {d['mean']:.6g} | [{lo:.6g}, {hi:.6g}] |")
        lines += ["", f"置换最大误差 {row['postfit_permutation_error']:.3g}；"
                  f"算子分解相对误差 {row['algebra_error_max']:.3g}。", "",
                  "最后十个完整遍历的更新前平均损失：" + ", ".join(f"{v:.5g}" for v in pass_means[-10:]) + "。",
                  "完整遍历均包含全部 48 个 parent×干扰类型记录。该轨迹用于判断固定预算内的优化状态，"
                  "没有据此选取其他检查点；本轮结果不能证伪所有训练预算下的该参数化。", ""]
        for criterion, cases_by_parent in cases.items():
            nonzero = int(np.count_nonzero(raw[criterion].mean(1)))
            lines += [f"### {coord} Top-5：{criterion}", "",
                      f"16 个 parent 中此指标非零者为 {nonzero} 个。排序可能包含零值，占位项不代表失败事件。",
                      "每个 parent 选择该指标最大的干扰类型列出定位证据；完整三类明细见 HEAD_REVIEW.json。", "",
                      "| parent | 类型/JNR | 指标 | 锚点召回 | 锚点误差/点 | 支撑外能量 | Oracle-β 残差 | 盲残差 | 漂移比 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for detail in cases_by_parent:
                family, v = max(detail["families"].items(), key=lambda pair: pair[1][criterion])
                lines.append(f"| {detail['parent_id']} | {family}/{v['jnr_db']:.2f} dB | {v[criterion]:.6g} | "
                             f"{v['anchor_energy_recall_2_samples']:.2%} | {v['anchor_nearest_error_samples']:.3g} | "
                             f"{v['outside_strong_support']:.2%} | {v['learned_b_oracle_beta_relative_residual']:.2%} | "
                             f"{v['blind_j_relative_residual']:.2%} | {v['drift_over_target_energy']:.4g} |")
            lines.append("")
    lines += ["## 解释边界", "",
              "- Top-8 候选的集合相同、仅顺序变化时，已验证共享头与求解合成的置换不变性；集合成员变化仍可能产生漂移。",
              "- Oracle-β 在固定已学字典上用 J 重拟合，仅用于诊断字典覆盖，盲推理与训练前向均不读取 J 或几何真值。",
              "- 候选召回按独立重放分量能量加权；重叠分量可能相干叠加，因此该召回不等于可直接扣除的总 J 能量比例。",
              "- 极低 JNR 的定位失败参与全局闸门，未事后排除。高 JNR 的改善另行分层呈现。",
              "- repeat ≥35 dB 在这批 TRAIN16 中只有 1 个 parent，相应区间不能刻画跨场景不确定性；零事件的 bootstrap 区间也不构成零风险保证。",
              "- 算子分解在给定观测条件下用 complex128 固定算子核验；部署输出仍为 float32。",
              "- 本轮只有一个初始化种子和 16 个训练 parent，CPU 拟合收益不代表 DEV 泛化或独立确认。", ""]
    payload = {**provenance(__file__), "started_utc": started, "ended_utc": now(),
               "training_updates": 0, "formal_gpu_training_paused": True,
               "summary_sha256": sha(OUT / "SUMMARY.json"), "anchor_recall_sha256": sha(OUT / "ANCHOR_RECALL.json"),
               "data": data_receipt, "parent_ids": ids.tolist(), "coordinates": result}
    atomic_json(OUT / "HEAD_REVIEW.json", payload)
    atomic_write(OUT / "HEAD_REVIEW.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    report_path = OUT / "REPORT.md"
    report = report_path.read_text(encoding="utf8").split("\n## CPU 定位复核补充\n")[0]
    report += "\n## CPU 定位复核补充\n\n" + (OUT / "ANCHOR_RECALL.md").read_text(encoding="utf8") + "\n" + "\n".join(lines)
    atomic_write(report_path, lambda stream: stream.write(report.encode("utf8")))
    atomic_json(OUT / "REVIEW_SEAL.json", {"ended_utc": now(), "summary_sha256": sha(OUT / "SUMMARY.json"),
               "report_sha256": sha(report_path), "head_review_sha256": sha(OUT / "HEAD_REVIEW.json"),
               "anchor_recall_sha256": sha(OUT / "ANCHOR_RECALL.json"), "formal_gpu_training_paused": True})


if __name__ == "__main__":
    run()
