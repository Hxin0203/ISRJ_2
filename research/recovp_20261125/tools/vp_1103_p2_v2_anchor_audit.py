"""Post-fit oracle recall of candidate anchors; no optimization or GPU use."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from vp_1103_common import WORK, atomic_json, frame_jnr, now, provenance, read, sha, stratified
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_p2_smoothvp_v2 import SmoothVPNetV2
from vp_1103_replay_truth import recover_selected, replay_components
from tools.durable_io import atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_v2_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_v2_preflight"


def run(coordinates_to_run=("RAW_IQ", "DECHIRP")):
    started = now()
    torch.set_num_threads(2)
    data, binding = load("train")
    existing = OUT / "ANCHOR_RECALL.json"
    rows = read(existing)["coordinates"] if existing.exists() else {}
    for name, row in rows.items():
        if sha(PRIVATE / name / "step_3840.pt") != row["checkpoint_sha256"]:
            raise ValueError("Previously scored checkpoint changed")
    for coordinates in coordinates_to_run:
        path = PRIVATE / coordinates / "step_3840.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        small = subset(data, np.searchsorted(data["parent_id"], np.sort(state["parent_ids"])))
        geometry = recover_selected(small, WORK / "outputs/p2_head_substitution_audit", binding)
        model = SmoothVPNetV2(coordinates).eval()
        model.load_state_dict(state["model"])
        metrics = {key: np.zeros((16, 3)) for key in (
            "anchor_component_energy_recall_2_samples", "refined_component_energy_recall_2_samples",
            "refined_component_energy_recall_0p5_samples", "anchor_weighted_nearest_error_samples",
            "refined_weighted_nearest_error_samples", "candidate_membership_change_after_target")}
        with torch.inference_mode():
            for parent in range(16):
                for family in range(3):
                    s, j, e = [small[k][parent, family].astype(np.complex128) for k in ("s", "j", "epsilon")]
                    parts = [model.factor(iq_tensor(y, "cpu")) for y in (j + e, s + j + e)]
                    p0, p1 = parts
                    sums = dict.fromkeys(metrics, 0.)
                    total_energy = 0.
                    for pulse in range(8):
                        truth = replay_components(float(small["target_delay_samples"][parent]),
                            float(small["doppler_hz"][parent]), family, geometry[parent, family, pulse])
                        if not truth:
                            continue
                        tau = np.array([item["phase_delay"] for item in truth])
                        energy = np.array([item["energy"] for item in truth])
                        anchor = p0["anchors"][pulse].numpy()
                        refined = p0["delay"][pulse].numpy()
                        ea = abs(tau[:, None] - anchor).min(axis=1)
                        er = abs(tau[:, None] - refined).min(axis=1)
                        sums["anchor_component_energy_recall_2_samples"] += float(energy @ (ea <= 2.))
                        sums["refined_component_energy_recall_2_samples"] += float(energy @ (er <= 2.))
                        sums["refined_component_energy_recall_0p5_samples"] += float(energy @ (er <= .5))
                        sums["anchor_weighted_nearest_error_samples"] += float(energy @ ea)
                        sums["refined_weighted_nearest_error_samples"] += float(energy @ er)
                        distances = abs(p0["anchors"][pulse, :, None] - p1["anchors"][pulse, None]).numpy()
                        left, right = linear_sum_assignment(distances)
                        sums["candidate_membership_change_after_target"] += float((distances[left, right] > 2.).mean() * energy.sum())
                        total_energy += float(energy.sum())
                    for key in metrics:
                        metrics[key][parent, family] = sums[key] / total_energy
        raw = PRIVATE / coordinates / "anchor_recall.npz"
        atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=small["parent_id"], **metrics))
        rows[coordinates] = {"metrics": stratified(metrics, frame_jnr(small)),
                             "private_metrics_sha256": sha(raw), "checkpoint_sha256": sha(path)}
    summary = {**provenance(__file__, WORK / "tools/vp_1103_replay_truth.py",
                            WORK / "tools/vp_1103_p2_smoothvp_v2.py"),
               "started_utc": started, "ended_utc": now(), "training_updates": 0, "coordinates": rows,
               "complete_both_coordinates": set(rows) == {"RAW_IQ", "DECHIRP"},
               "train_data": binding,
               "scope": "Oracle geometry is used only for post-fit TRAIN16 recall. Every true replay delay is scored.",
               "energy_weight": "Isolated unit-component energies; overlapping components can interfere."}
    atomic_json(OUT / "ANCHOR_RECALL.json", summary)
    lines = ["# v2 候选峰召回核验", "", "本附检仅做训练后定位评分，没有优化更新。",
             "能量按独立重放分量计权，所有真实延迟均参与，包括超过八个延迟的情形。", "",
             "| 坐标 | 锚点 ±2 点召回 | 微调后 ±2 点召回 | 微调后 ±0.5 点召回 | 锚点平均最近误差 | 微调后误差 | 目标注入后集合变化 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in rows.items():
        m = row["metrics"]["global"]
        fields = ("anchor_component_energy_recall_2_samples", "refined_component_energy_recall_2_samples",
                  "refined_component_energy_recall_0p5_samples", "anchor_weighted_nearest_error_samples",
                  "refined_weighted_nearest_error_samples", "candidate_membership_change_after_target")
        lines.append(f"| {name} | " + " | ".join(f"{m[key]['mean']:.5g}" for key in fields) + " |")
    atomic_write(OUT / "ANCHOR_RECALL.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", nargs="+", choices=("RAW_IQ", "DECHIRP"),
                        default=("RAW_IQ", "DECHIRP"))
    run(parser.parse_args().coordinates)
