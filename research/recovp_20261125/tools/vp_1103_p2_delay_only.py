"""Supplement S2 with strictly unchanged learned masks for all eight columns."""
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from vp_1103_common import WORK, atomic_json, frame_jnr, now, provenance, read, sha, stratified
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_smoothvp import SmoothVPNet
from vp_1103_p2_evaluate import iq_tensor
from vp_1103_replay_truth import recover_selected, replay_components
from vp_1103_p2_head_ablation import dictionary, fit, score, OUT, PRIVATE
from tools.durable_io import atomic_write


def run():
    started = now()
    torch.set_num_threads(2)
    data, binding = load("train")
    saved = torch.load(WORK / "outputs/p2_smoothvp_preflight/RAW_IQ/extended_overfit.pt",
                       map_location="cpu", weights_only=False)
    small = subset(data, np.searchsorted(data["parent_id"], np.sort(saved["parent_ids"])))
    geometry = recover_selected(small, PRIVATE, binding)
    model = SmoothVPNet("RAW_IQ").eval()
    model.load_state_dict(saved["model"])
    outputs = {key: np.zeros((16, 3, 4, 8, 1800), np.complex64)
               for key in ("S2_delay_only_blind", "S2_delay_only_oracle_beta")}
    costs = {key: np.zeros((16, 3, 8)) for key in outputs}
    residual = {key: np.zeros((16, 3, 8)) for key in outputs}
    with torch.inference_mode():
        for parent in range(16):
            for family in range(3):
                s, j, e = [small[k][parent, family].astype(np.complex128) for k in ("s", "j", "epsilon")]
                states = np.stack((e, s + e, j + e, s + j + e))
                batches = [model.factor(iq_tensor(y, "cpu")) for y in states]
                for pulse in range(8):
                    components = replay_components(float(small["target_delay_samples"][parent]),
                        float(small["doppler_hz"][parent]), family, geometry[parent, family, pulse])
                    for state in range(4):
                        parts = {key: value[pulse:pulse + 1] for key, value in batches[state].items()}
                        delay = parts["delay"][0].double().clone()
                        masks = parts["mask"][0].double()
                        if components:
                            tau = np.array([row["phase_delay"] for row in components[:8]])
                            matrix = abs(delay.numpy()[:, None] - tau[None])
                            slots, members = linear_sum_assignment(matrix)
                            nearest = matrix.argmin(axis=1)
                            delay = torch.from_numpy(tau[nearest])
                            delay[slots] = torch.from_numpy(tau[members])
                        b = dictionary(model, delay, masks)
                        observed, truth = torch.from_numpy(states[state, pulse]), torch.from_numpy(j[pulse])
                        for name in outputs:
                            oracle = "oracle" in name
                            c, gain = fit(b, truth if oracle else observed, parts, observed, oracle)
                            outputs[name][parent, family, state, pulse] = (observed - c).numpy()
                            if state == 2:
                                costs[name][parent, family, pulse] = gain
                                residual[name][parent, family, pulse] = float((truth - c).abs().square().sum())
    rows = {}
    jnr = frame_jnr(small)
    for name in outputs:
        metrics = score(small, outputs[name], costs[name], residual[name])
        path = PRIVATE / f"{name}.npz"
        atomic_write(path, lambda stream: np.savez_compressed(stream, parent_id=small["parent_id"], **metrics))
        rows[name] = {"metrics": stratified(metrics, jnr), "private_metrics_sha256": sha(path)}
    archive = OUT / "admission_snapshot"
    archive.mkdir()
    for file in ("SUMMARY.json", "STATUS.json", "REPORT.md"):
        shutil.copy2(OUT / file, archive / file)
    summary = read(OUT / "SUMMARY.json")
    summary["methods"].update(rows)
    receipt = {**provenance(__file__), "started_utc": started, "ended_utc": now(),
               "training_updates": 0, "admission_summary_sha256": sha(archive / "SUMMARY.json"),
               "admission_summary_path": (archive / "SUMMARY.json").relative_to(WORK).as_posix(),
               "interpretation": "Original S2 uses zero padding when true distinct delays are fewer than eight. "
                                 "This supplemental S2 changes delays only, retains every learned mask, "
                                 "and leaves truly unjammed pulse dictionaries unchanged."}
    summary["delay_only_refinement"] = receipt
    atomic_json(OUT / "SUMMARY.json", summary)
    status = read(OUT / "STATUS.json")
    atomic_json(OUT / "STATUS.json", {**status, "summary_sha256": sha(OUT / "SUMMARY.json"),
                                    "delay_only_refinement": receipt})
    text = (OUT / "REPORT.md").read_text(encoding="utf8")
    lines = [text.rstrip(), "", "## 严格只替换延迟的 S2 补充", "",
             "原 S2 对独立真值延迟不足八个的脉冲做了零列填充，因此同时改变了有效列数。"
             "本补充保留八个已学软窗，剩余列复用最近的真值延迟，且空重放脉冲保留原字典，单独考察延迟替换。",
             "", "| 方法 | 全局 J 残差 | direct | repeat | cyclic |", "|---|---:|---:|---:|---:|"]
    for name, row in rows.items():
        values = [row["metrics"][key]["j_relative_residual"]["mean"] for key in
                  ("global", "direct/all", "repeat/all", "cyclic/all")]
        lines.append(f"| {name} | " + " | ".join(f"{value:.3%}" for value in values) + " |")
    lines += ["", "原阶段 2 准入收据与源码保持可追溯。S4 表达力结论保持原数值。"]
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))


if __name__ == "__main__":
    run()
