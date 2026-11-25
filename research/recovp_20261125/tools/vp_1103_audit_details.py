"""Counterfactual false alarms, top-parent attribution, and perturbation plots."""
import argparse
import itertools
import time

import numpy as np

from vp_1103_common import (FAMILIES, WORK, atomic_json, frame_jnr, now, parent_roster,
                           provenance, read, sha, stratified, validate_role, worst_cases,
                           write_cases)
from vp_1103_gate import EVIDENCE, NAMES, phase_status, private_metrics
from vp_1103_p1 import Projection, complement, orthogonal_basis, save_metrics
from vp_1103_p1b import estimate_protection, estimate_replay
from vp_1103_references import reference_metrics
from paper2.radar_eval import ca_cfar, doppler_process, pulse_compress
from tools.durable_io import atomic_write


def detection_flags(y):
    compressed, _ = pulse_compress(y)
    values = []
    for block in compressed:
        rd, _ = doppler_process(block)
        found, _, valid, _, _ = ca_cfar(np.abs(rd)**2)
        values.append(float(np.any(found & valid)))
    return np.asarray(values).reshape(256, 3)


def contribution(dev, mode):
    s, j, e = [dev[name].reshape(-1, 8, 1800).astype(np.complex128)
               for name in ("s", "j", "epsilon")]
    y0 = j + e
    y1 = s + j + e
    residual = np.empty_like(s)
    leak = np.empty_like(s)
    drift = np.empty_like(s)

    def fit(y):
        a, _ = estimate_protection(y)
        source = complement(orthogonal_basis(a), y) if mode == "isolated" else y
        return Projection(a, estimate_replay(source), "adaptive")

    for group, pulse in itertools.product(range(len(y0)), range(8)):
        index = group, pulse
        h0, h1 = fit(y0[index]), fit(y1[index])
        residual[index] = y0[index] - h0(y0[index])
        leak[index] = -h1(s[index])
        drift[index] = h0(y0[index]) - h1(y0[index])
    values = {"background": detection_flags(residual),
              "background_plus_leakage": detection_flags(residual + leak),
              "background_plus_drift": detection_flags(residual + drift),
              "no_target_residual_after_refit": detection_flags(residual + leak + drift)}
    for label in tuple(values)[1:]:
        values[f"{label}_minus_background"] = values[label] - values["background"]
    return values


def add_report(out, title, contents):
    path = out / "REPORT.md"
    current = path.read_text(encoding="utf8").split(title, 1)[0].rstrip()
    lines = [current, "", title, "", *contents, ""]
    atomic_write(path, lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


def build_cases(cases, ids, metrics, keys):
    result = {}
    for category, rows in cases.items():
        enriched = []
        for row in rows:
            index = np.searchsorted(ids, row["parent_id"])
            evidence = {name: dict(zip(FAMILIES, map(float, metrics[name][index]))) for name in keys}
            enriched.append({**row, "mechanism_values": evidence})
        result[category] = enriched
    return result


def case_lines(rows, labels):
    result = []
    for key, group in rows.items():
        result.extend([f"#### {key}", ""])
        for row in group:
            details = []
            for name, values in row["mechanism_values"].items():
                details.append(labels[name] + " " + "/".join(f"{value:.3g}" for value in values.values()))
            result.append(f"- `{row['parent_id']}` " + "；".join(details))
        result.append("")
    return result


def plot_curves(local, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {"relative_target_leakage_l2": "Relative target leakage (L2)",
               "damage": "Target-increment damage",
               "frame_pfa_jammer": "Jammer-only frame false-alarm rate",
               "removed_noise_gain_frobenius_squared": r"Removed-component noise gain $\|H\|_F^2$"}
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 10.5), layout="constrained")
    for column, coordinate in enumerate(("delay", "doppler")):
        for row, (key, label) in enumerate(metrics.items()):
            ax = axes[row, column]
            for mode, marker in (("zero", "o"), ("tangent", "s"), ("neighbors", "^")):
                selected = sorted((entry for entry in local.values()
                                   if entry["axis"] == coordinate and entry["mode"] == mode),
                                  key=lambda entry: entry["delta"])
                x = [entry["delta"] * (50 if coordinate == "doppler" else 1) for entry in selected]
                values = [entry["metrics"]["global"][key] for entry in selected]
                y = [value["mean"] for value in values]
                lower, upper = np.asarray([value["parent_bootstrap_ci95"] for value in values]).T
                curve, = ax.plot(x, y, marker=marker, markersize=4, label=mode)
                ax.fill_between(x, lower, upper, color=curve.get_color(), alpha=.15)
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Delay error (samples)" if column == 0 else "Doppler error (kHz)")
            ax.axvline(0, color=".65", linewidth=.8, zorder=0)
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("DEV geometry perturbations · mean and 95% parent bootstrap interval")
    target = out / "PERTURBATION_CURVES.png"
    atomic_write(target, lambda stream: fig.savefig(stream, format="png", dpi=180))
    plt.close(fig)
    return sha(target)


def run(wait):
    path, receipt = validate_role("dev")
    with np.load(path, allow_pickle=False) as archive:
        dev = {key: archive[key] for key in archive.files}
    ids, jnr = parent_roster(dev, 256), frame_jnr(dev)
    start = now()
    b_private = WORK / "outputs" / NAMES["b"]
    followup = {}
    bindings = {}
    for mode in ("raw", "isolated"):
        metrics = contribution(dev, mode)
        followup[mode] = stratified(metrics, jnr)
        bindings[mode] = save_metrics(b_private, f"counterfactual_{mode}", metrics)
    while phase_status("b")["stage"] != "COMPLETE":
        if not wait or phase_status("b")["stage"] == "FAILED":
            raise RuntimeError("P1-B primary results unavailable")
        time.sleep(20)

    a_out = EVIDENCE / NAMES["a"]
    a = read(a_out / "SUMMARY.json")
    key = a["primary_method"]
    values = private_metrics("a", key, a["methods"][key]["private_metrics_sha256"])
    labels = {"mf_residual_peak_over_noise_rms": "残峰/噪声 RMS", "frame_pfa_noise": "噪声对照虚警",
              "min_chi_jammed_pulse": "最小 χ"}
    a["worst_case_attribution"] = build_cases(a["worst_cases"][key], ids, values, labels)
    atomic_json(a_out / "SUMMARY.json", a)
    add_report(a_out, "## 最差 parent 的机制核验",
               ["每行依次列出 direct、repeat、cyclic。噪声对照同样触发时，当前聚合证据无法把全部虚警归因给残留干扰。",
                "保护投影的损伤约在数值精度范围，排名仅保留审计信息。", "",
                *case_lines(a["worst_case_attribution"], labels)])

    out = EVIDENCE / NAMES["b"]
    summary = read(out / "SUMMARY.json")
    joint = ["用同背景的恢复观测作基线，分别加回泄漏、漂移和两项之和，再用同一 CFAR 打分。目标真值只用于评估时移除目标。",
             "", "| 估计输入 | 背景 | 加泄漏 | 加漂移 | 加泄漏与漂移 |",
             "|---|---:|---:|---:|---:|"]
    cases = {}
    labels = {"leak_over_target_energy": "泄漏能量/目标", "drift_over_target_energy": "漂移能量/目标",
              "ghost_protection_frame": "误保护", "protection_frame_pfa_delta_all": "保护引起的虚警差"}
    for mode, groups in followup.items():
        row = summary["blind_proxy"][mode]
        values = private_metrics("b", f"blind_{mode}", row["private_metrics_sha256"])
        if abs(values["frame_pfa_jammer"].mean() - groups["global"]["background"]["mean"]) > 1e-14:
            raise ValueError("counterfactual background differs from primary B audit")
        cases[mode] = build_cases(summary["worst_cases"][mode], ids, values, labels)
        keys = ("background", "background_plus_leakage", "background_plus_drift", "no_target_residual_after_refit")
        joint.append(f"| {mode} | " + " | ".join(f"{groups['global'][key]['mean']:.3%}" for key in keys) + " |")
    curves_hash = plot_curves(summary["perturbations"], out)
    summary.update(counterfactual_false_alarm=followup, counterfactual_metrics_sha256=bindings,
                   worst_case_attribution=cases, perturbation_curves_sha256=curves_hash)
    atomic_json(out / "SUMMARY.json", summary)
    for mode, selected in cases.items():
        joint += ["", f"### {mode} 的最差 parent 归因", "", *case_lines(selected, labels)]
    joint += ["", "![时延与频偏扰动曲线](PERTURBATION_CURVES.png)"]
    add_report(out, "## 泄漏与漂移的虚警分解", joint)

    c_out = EVIDENCE / NAMES["c"]
    c = read(c_out / "SUMMARY.json")
    refs, _ = reference_metrics("unet_tail", ids)
    metrics = {key: np.mean([row[key] for row in refs.values()], axis=0) for key in refs[17]}
    features = private_metrics("c", "features", c["private_metrics_sha256"])
    categories = ("peak_Y_active_windows", "hankel_Y", "min_chi_jammed_pulse")
    c["unet_tail_worst_cases"] = build_cases(worst_cases(metrics, ids, jnr), ids, features, categories)
    risk = c["profile_likelihood"]["high_risk_parent_conflict_rate"]
    if risk["mean"] == 0:
        risk["one_sided_binomial_upper_95"] = 1 - .05**(1 / risk["parents"])
    atomic_json(c_out / "SUMMARY.json", c)
    lines = ["指标对应既有 U-Net+Tail 的逐 parent 均值。排序仅用于定位极端样本，观测特征本身不用于归因。", ""]
    if risk.get("one_sided_binomial_upper_95") is not None:
        lines += [f"零命中的单侧精确二项 95% 上限为 {risk['one_sided_binomial_upper_95']:.2%}，限于本次有限网格和筛选子集。", ""]
    lines += case_lines(c["unet_tail_worst_cases"], {
        "peak_Y_active_windows": "观测峰", "hankel_Y": "观测 Hankel 比", "min_chi_jammed_pulse": "最小 χ"})
    add_report(c_out, "## 历史 U-Net+Tail 最差 parent", lines)
    atomic_json(out / "AUDIT_DETAILS_RECEIPT.json", {
        **provenance(__file__), "stage": "COMPLETE", "started_utc": start, "ended_utc": now(),
        "data": {"dev": receipt}, "counterfactual_metrics_sha256": bindings,
        "perturbation_curves_sha256": curves_hash})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-b", action="store_true")
    run(parser.parse_args().wait_for_b)
