"""CPU-only review of learned delays, support coverage and atom collapse."""
from pathlib import Path

import numpy as np
import torch

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, provenance, sha,
                           stratified, write_cases, read)
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_p2_smoothvp import SmoothVPNet
from tools.durable_io import atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_localization_review"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"


def statistics(parts):
    b = parts["b"]
    gram = parts["gram"]
    coherence = gram.abs().clone()
    coherence[:, torch.arange(8), torch.arange(8)] = 0
    rank = (gram.diagonal(dim1=-2, dim2=-1).real.sum(-1)**2 /
            gram.abs().square().sum((-2, -1)))
    return {"maximum_atom_coherence": coherence.amax((-2, -1)),
            "effective_rank_participation": rank}


def run():
    start = now()
    torch.set_num_threads(2)
    data, binding = load("train")
    result = {}
    status = {**provenance(__file__, WORK / "tools/vp_1103_p2_smoothvp.py"),
              "stage": "RUNNING", "started_utc": start, "ended_utc": None,
              "data": {"train": binding}, "device": "cpu", "training_updates": 0}
    atomic_json(OUT / "STATUS.json", status)
    for coordinates in ("RAW_IQ", "DECHIRP"):
        cp = PRIVATE / coordinates / "extended_overfit.pt"
        saved = torch.load(cp, map_location="cpu", weights_only=False)
        selected = np.searchsorted(data["parent_id"], np.sort(saved["parent_ids"]))
        small = subset(data, selected)
        count = len(selected)
        model = SmoothVPNet(coordinates).cpu().eval()
        model.load_state_dict(saved["model"])
        metrics = {key: np.zeros((count, 3)) for key in (
            "paired_delay_mean_absolute_samples", "paired_delay_max_absolute_samples",
            "paired_log10_precision_mean_absolute_change", "maximum_atom_coherence",
            "effective_rank_participation", "J_energy_outside_strong_atom_regions",
            "paired_operator_frobenius_change", "paired_delay_mean_absolute_J_weighted_samples",
            "paired_delay_mean_absolute_high35_samples", "maximum_atom_coherence_J_weighted",
            "effective_rank_J_weighted", "paired_log10_precision_J_weighted_change")}
        with torch.inference_mode():
            for parent in range(count):
                for family in range(3):
                    s, j, e = [small[k][parent, family].astype(np.complex128) for k in ("s", "j", "epsilon")]
                    p0, p1 = [model.factor(iq_tensor(y, "cpu")) for y in (j + e, s + j + e)]
                    delta = (p1["delay"] - p0["delay"]).abs()
                    pulse_energy = torch.from_numpy(np.sum(np.abs(j)**2, axis=-1))
                    weight = pulse_energy / pulse_energy.sum()
                    high = (small["snr_db"][parent] + small["jsr_db"][parent, family]) >= 35
                    metrics["paired_delay_mean_absolute_samples"][parent, family] = float(delta.mean())
                    metrics["paired_delay_max_absolute_samples"][parent, family] = float(delta.max())
                    metrics["paired_delay_mean_absolute_J_weighted_samples"][parent, family] = float(
                        (delta.mean(-1) * weight).sum())
                    metrics["paired_delay_mean_absolute_high35_samples"][parent, family] = (
                        float(delta[high].mean()) if high.any() else np.nan)
                    metrics["paired_log10_precision_mean_absolute_change"][parent, family] = float(
                        (p1["precision"].log10() - p0["precision"].log10()).abs().mean())
                    metrics["paired_log10_precision_J_weighted_change"][parent, family] = float(
                        ((p1["precision"].log10() - p0["precision"].log10()).abs().mean(-1) * weight).sum())
                    stats = statistics(p0)
                    for key, value in stats.items():
                        metrics[key][parent, family] = float(value.mean())
                    metrics["maximum_atom_coherence_J_weighted"][parent, family] = float(
                        (stats["maximum_atom_coherence"] * weight).sum())
                    metrics["effective_rank_J_weighted"][parent, family] = float(
                        (stats["effective_rank_participation"] * weight).sum())
                    density = p0["b"].abs().square()
                    relative = density / density.amax(dim=1, keepdim=True).clamp_min(1e-24)
                    strong = (relative >= .1).any(-1)
                    jp = torch.from_numpy(np.abs(j)**2)
                    metrics["J_energy_outside_strong_atom_regions"][parent, family] = float(
                        (jp * ~strong).sum() / jp.sum())
                    r0, r1 = p0["right"], p1["right"]
                    b0, b1 = p0["b"], p1["b"]
                    gain0 = torch.einsum("bij,bji->b", p0["gram"], r0 @ r0.mH).real
                    gain1 = torch.einsum("bij,bji->b", p1["gram"], r1 @ r1.mH).real
                    inner = torch.einsum("bij,bji->b", b1.mH @ b0, r0 @ r1.mH).real
                    difference = (gain0 + gain1 - 2 * inner).clamp_min(0).sum()
                    metrics["paired_operator_frobenius_change"][parent, family] = float(difference)
        raw = PRIVATE / coordinates / "localization_review.npz"
        atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=small["parent_id"], **metrics))
        old = read(WORK / f"evidence/outputs/p2_smoothvp_preflight/FIT_DIAGNOSIS_{coordinates}.json")
        result[coordinates] = {"metrics": stratified(metrics, frame_jnr(small)),
                               "checkpoint_sha256": sha(cp), "private_metrics_sha256": sha(raw),
                               "fit_diagnosis": old["metrics"], "worst_cases": old["worst_cases"],
                               "fit_diagnosis_sha256": sha(WORK / f"evidence/outputs/p2_smoothvp_preflight/FIT_DIAGNOSIS_{coordinates}.json")}
    summary = {"coordinates": result, "scope": "TRAIN16 post-fit diagnosis, zero new optimization.",
               "strong_region_threshold": .1, "support_note": "Descriptive support coverage, not a representability bound.",
               "interpretation": [
                   "Oracle coefficient fitting on the learned B leaves most interference energy.",
                   "Dictionary-only mask/delay fitting attains lower residuals, so encoder output accuracy remains a live issue.",
                   "Paired delay and precision changes quantify observation-dependent operator motion.",
                   "These results do not establish that increasing K or adding another branch is needed."]}
    atomic_json(OUT / "SUMMARY.json", summary)
    lines = ["# SmoothVP-Net 定位诊断复核", "",
             "本报告仅分析 16 个 TRAIN parent 的 CPU 预检模型。各类型和八脉冲始终按 parent 聚合，区间为配对聚类重采样结果。",
             "本次复核没有新增优化更新。正式 P2 按用户选择暂停，P0 原队列继续运行。", "",
             "| 指标 | RAW_IQ | DECHIRP |",
             "|---|---:|---:|"]
    labels = {
        "paired_delay_mean_absolute_J_weighted_samples": "按干扰能量加权的时延变化（采样点）",
        "paired_delay_mean_absolute_high35_samples": "≥35 dB 脉冲的平均时延变化（采样点）",
        "paired_log10_precision_J_weighted_change": "按干扰能量加权的精度变化（log10）",
        "maximum_atom_coherence_J_weighted": "按干扰能量加权的最大相干度",
        "effective_rank_J_weighted": "按干扰能量加权的参与率有效秩",
        "paired_delay_mean_absolute_samples": "全部八脉冲的平均时延变化（采样点）",
        "J_energy_outside_strong_atom_regions": "强原子支撑区域外的干扰能量比例",
        "paired_operator_frobenius_change": "配对算子变化的 Frobenius 范数平方"}
    for key, label in labels.items():
        cells = []
        for coordinate in result.values():
            row = coordinate["metrics"]["global"][key]
            lo, hi = row["parent_bootstrap_ci95"]
            cells.append(f"{row['mean']:.5g} [{lo:.5g}, {hi:.5g}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in (
        ("learned_b_oracle_beta_relative_residual", "固定已学字典、重拟合幅相后的 J 残差比"),
        ("leak_over_target_energy", "目标泄漏能量/目标能量"),
        ("drift_over_target_energy", "算子漂移能量/目标能量")):
        cells = []
        for coordinate in result.values():
            row = coordinate["fit_diagnosis"]["global"][key]
            lo, hi = row["parent_bootstrap_ci95"]
            cells.append(f"{row['mean']:.5g} [{lo:.5g}, {hi:.5g}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 复核判断", "",
              "1. 幅相求解层已正常工作。在固定已学字典上重新拟合真值 J 的复幅相后，残差仍大，主要短板包含字典定位与覆盖。",
              "2. 独立逐样本优化软窗和时延可取得更低残差，当前结果尚不能把问题归为 K=8 的表示容量上限。",
              "3. 同背景注入目标后，时延、精度和整个算子仍发生变化。连续参数化提供可微性，需要进一步验证其稳定性。",
              "全部八脉冲的时延统计包含无干扰脉冲；目标注入后这部分定位变化可能被高阻尼抑制。"
              "定位稳定性的主要判断应优先使用干扰能量加权与 ≥35 dB 脉冲统计。",
              "时延按固定输出通道配对，原子交换也会影响该值；应结合实际算子变化和漂移能量解释。",
              "4. 支撑诊断使用相对原子能量的 0.1 阈值，仅为后验可视化统计。它不是前向硬门控，也不是误差下界。", "",
              "## 后续最小审查建议", "",
              "先保持 K=8、求解器和统一损失，分别核验时延头与软窗头的定位精度。采用 CPU 受控替换逐项测试，"
              "将固定字典下的幅相误差、时延误差和支撑误差拆开，再决定新的初始化或训练顺序。",
              "当前证据支持优先审查字典生成头，尚不足以要求扩大 K、恢复二阶段保护或增加支线。", ""]
    for coord, row in result.items():
        lines += [f"## {coord} 最差 parent", ""]
        write_cases(lines, row["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "STATUS.json", {**status, "stage": "COMPLETE", "ended_utc": now(),
                                    "summary_sha256": sha(OUT / "SUMMARY.json")})


if __name__ == "__main__":
    run()
