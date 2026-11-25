"""Private TRAIN16 dictionary-only optimization to diagnose failed blind fitting.

This is explicitly supervised by J as an expression ceiling. It never supplies
geometric labels to the deployable model and is excluded from P2 comparisons.
"""
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from vp_1103_common import WORK, atomic_json, now, provenance, read, sha
from vp_1103_p2_preflight import load, representative
from vp_1103_p2_smoothvp import SmoothVPNet, MIN_PRECISION
from tools.durable_io import ProcessLock, atomic_write, process_alive

OUT = WORK / "evidence/outputs/p2_smoothvp_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"
STEPS = 512


def run():
    start = now()
    torch.set_num_threads(2)
    while True:
        state = read(OUT / "STATUS.json")
        if state["stage"] == "COMPLETE":
            break
        if state["stage"] == "FAILED" or not process_alive(state["pid"]):
            raise RuntimeError("Primary CPU check did not complete")
        time.sleep(20)
    data, receipt = load("train")
    indices = representative(data)
    shape = (len(indices), 3)
    # TRAIN metadata is used solely to select one actually jammed pulse per family.
    jnr = data["snr_db"][indices, None] + data["jsr_db"][indices]
    pulses = np.nanargmax(jnr, axis=-1)
    s, j, e = [np.take_along_axis(data[key][indices], pulses[..., None, None], axis=2).squeeze(2)
               .reshape(-1, 1800).astype(np.complex128) for key in ("s", "j", "epsilon")]
    y = torch.from_numpy(np.stack(((s + j + e).real, (s + j + e).imag), 1).astype(np.float32))
    target = torch.from_numpy(j)
    target = target / target.abs().square().mean(-1, keepdim=True).sqrt()
    summaries = {}
    for coordinate in ("RAW_IQ", "DECHIRP"):
        model = SmoothVPNet(coordinate)
        checkpoint = PRIVATE / coordinate / "overfit.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        with torch.inference_mode():
            parts = model.factor(y)
        delay = torch.nn.Parameter(parts["delay"].detach().double())
        logits = torch.nn.Parameter(torch.logit(parts["mask"].detach().double().clamp(.001, .999)))
        # Fit per-waveform amplitude freedom with a complex128 projection on J.
        # This removes the encoder and precision gate as optimization confounders.
        optimizer = torch.optim.Adam([delay, logits], lr=.02)
        trajectory = []
        per_waveform = None
        for step in range(STEPS + 1):
            masks = F.avg_pool1d(torch.sigmoid(logits), 9, stride=1, padding=4, count_include_pad=False)
            b = model.dictionary(delay, masks)
            gram = b.mH @ b
            normal = gram + MIN_PRECISION * torch.eye(8, dtype=torch.complex128)[None]
            beta = torch.linalg.solve(normal, b.mH @ target[..., None])
            fitted = (b @ beta).squeeze(-1)
            residual = (target - fitted).abs().square().sum(-1) / target.abs().square().sum(-1)
            loss = residual.mean()
            if step == 0:
                initial = residual.detach().numpy()
            if step % 32 == 0:
                trajectory.append({"step": step, "mean_relative_j_residual": float(loss.detach())})
                atomic_json(OUT / "EXPRESSION_STATUS.json", {
                    "stage": "RUNNING", "coordinates": coordinate, "step": step,
                    "started_utc": start, "updated_utc": now(), "pid": os.getpid()})
            if step == STEPS:
                per_waveform = residual.detach().numpy()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delay, logits], 5., error_if_nonfinite=True)
            optimizer.step()
        raw = PRIVATE / coordinate / "dictionary_only_expression.npz"
        atomic_write(raw, lambda stream: np.savez_compressed(
            stream, parent_id=data["parent_id"][indices], before=initial.reshape(shape),
            after=per_waveform.reshape(shape), selected_pulses=pulses,
            delays=delay.detach().numpy(), masks=masks.detach().numpy()))
        from vp_1103_common import stratified
        group_jnr = np.take_along_axis(jnr, pulses[..., None], axis=-1).squeeze(-1)
        summaries[coordinate] = {"steps": STEPS, "trajectory": trajectory,
            "metrics": stratified({"before_j_relative_residual": initial.reshape(shape),
                                   "after_j_relative_residual": per_waveform.reshape(shape)}, group_jnr),
            "private_metrics_sha256": sha(raw), "initial_checkpoint_sha256": sha(checkpoint)}
    result = {**provenance(__file__, Path(__file__).with_name("vp_1103_p2_smoothvp.py")),
              "data": {"train": receipt}, "started_utc": start, "ended_utc": now(), "stage": "COMPLETE",
              "scope": "Oracle J-supervised dictionary-only TRAIN16 fit. Not a blind inference method or P2 pass.",
              "coordinates": summaries}
    atomic_json(OUT / "EXPRESSION_SUMMARY.json", result)
    lines = ["# TRAIN16 dictionary-only expression diagnosis", "",
             "The test fits each training waveform's eight nonnegative masks and continuous delays directly to J.",
             "It isolates the learned encoder and precision gate. It supplies no inference performance claim.",
             "", "| Coordinates | Initial J residual ratio | Final J residual ratio | Final ≥35 dB J residual ratio |",
             "|---|---:|---:|---:|"]
    for coordinate, row in summaries.items():
        m = row["metrics"]
        lines.append(f"| {coordinate} | {m['global']['before_j_relative_residual']['mean']:.6g} | "
                     f"{m['global']['after_j_relative_residual']['mean']:.6g} | "
                     f"{m['jnr/>=35']['after_j_relative_residual']['mean']:.6g} |")
    atomic_write(OUT / "EXPRESSION_REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "EXPRESSION_STATUS.json", {"stage": "COMPLETE", "started_utc": start, "ended_utc": now(),
                                               "summary_sha256": sha(OUT / "EXPRESSION_SUMMARY.json")})


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "EXPRESSION.lock"):
        run()
