"""Fixed CPU convergence extension after preserving the 768-step checkpoint."""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, provenance, read, sha,
                           stratified, worst_cases, write_cases)
from vp_1103_p2_preflight import load, subset
from vp_1103_p2_smoothvp import SmoothVPNet
from vp_1103_p2_admission import fit_evidence
from vp_1103_p0 import evaluate
from paper2.train_1103 import make_batch
from paper2.train_256_8seed import atomic_torch_save
from tools.correction_tail_study import objective as objective_module
from tools.durable_io import ProcessLock, atomic_write, process_alive

OUT = WORK / "evidence/outputs/p2_smoothvp_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"
EXTRA_STEPS = 3072


def run():
    torch.set_num_threads(2)
    start = now()
    while True:
        state = read(OUT / "STATUS.json")
        if state["stage"] == "COMPLETE":
            break
        if state["stage"] == "FAILED" or not process_alive(state["pid"]):
            raise RuntimeError("Base CPU check did not finish")
        time.sleep(20)
    # Wait for dictionary-only diagnosis to read the exact same base checkpoint.
    while not (OUT / "EXPRESSION_SUMMARY.json").exists():
        time.sleep(20)
    archive = OUT / "attempt_002"
    archive.mkdir(exist_ok=True)
    for name in ("SUMMARY.json", "STATUS.json", "REPORT.md"):
        shutil.copy2(OUT / name, archive / name)
    for coordinates in ("RAW_IQ", "DECHIRP"):
        shutil.copytree(PRIVATE / coordinates,
                        WORK / "outputs/p2_smoothvp_preflight_attempt002" / coordinates)
    training, data_id = load("train")
    summary = read(archive / "SUMMARY.json")
    base_source = read(archive / "STATUS.json")
    status = {**base_source, **provenance(__file__, Path(__file__).with_name("vp_1103_p2_smoothvp.py")),
              "stage": "RUNNING", "pid": os.getpid(), "started_utc": start, "ended_utc": None,
              "phase": "CPU_CONVERGENCE_EXTENSION", "extra_steps": EXTRA_STEPS}

    def update(**fields):
        status.update(fields, updated_utc=now())
        atomic_json(OUT / "STATUS.json", status)

    update()
    objective_module.ARM = "tail"
    for coordinates, row in summary["coordinates"].items():
        checkpoint = PRIVATE / coordinates / "overfit.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        indices = np.searchsorted(training["parent_id"], saved["parent_ids"])
        small = subset(training, indices)
        model = SmoothVPNet(coordinates)
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(20261126)
        schedule = np.concatenate([rng.permutation(len(indices) * 3)
                                   for _ in range(EXTRA_STEPS // (len(indices) * 3))])
        trajectory = []
        for step, record in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            batch = make_batch(small, np.array([record]), "cpu")
            correction = model(batch["y"]).reshape(-1, 4, 2, 1800)
            terms = objective_module.study_objective(correction, batch, 1)
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError("CPU continuation diverged")
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            trajectory.append({key: float(v.detach()) for key, v in terms.items()})
            if step % 32 == 0:
                update(coordinates=coordinates, step=step, total_steps=768 + step, losses=trajectory[-1])
                atomic_json(PRIVATE / coordinates / "EXTENDED_TRAJECTORY.json", trajectory)
        after = evaluate(model.eval(), small)
        base_path = WORK / "outputs/p2_smoothvp_preflight_attempt002" / coordinates / "paired_metrics.npz"
        with np.load(base_path, allow_pickle=False) as z:
            before = {key.removeprefix("before__"): z[key] for key in z.files if key.startswith("before__")}
        jnr = frame_jnr(small)
        high = jnr >= 35.
        residual_ratio = float(after["background_nmse"][high].mean() / before["background_nmse"][high].mean())
        pfa_gain = float((before["frame_pfa_jammer"] - after["frame_pfa_jammer"]).mean())
        fit = fit_evidence(before, after, jnr)
        passed = fit["passes"]
        row["checks"]["train16_expression"] = "PASS" if passed else "FAIL"
        row["status"] = "PASS" if all(v == "PASS" for v in row["checks"].values()) else "FAIL"
        row.update(train16_high_jnr_residual_ratio=residual_ratio, train16_pfa_decrease=pfa_gain,
                   fit_evidence=fit,
                   overfit_steps=768 + EXTRA_STEPS, after=stratified(after, jnr),
                   worst_cases=worst_cases(after, small["parent_id"], jnr))
        target = PRIVATE / coordinates / "extended_metrics.npz"
        atomic_write(target, lambda stream: np.savez_compressed(
            stream, parent_id=small["parent_id"], **{f"after__{key}": v for key, v in after.items()},
            **{f"before__{key}": v for key, v in before.items()}))
        row["private_metrics_sha256"] = sha(target)
        row["private_metrics_file"] = target.relative_to(WORK).as_posix()
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "coordinates": coordinates, "parent_ids": saved["parent_ids"],
                          "steps": 768 + EXTRA_STEPS, "purpose": "TRAIN16_CPU_ENGINEERING_ONLY"},
                         PRIVATE / coordinates / "extended_overfit.pt")
        summary["scope"] = ("CPU health and TRAIN16 convergence check with 768 initial steps plus "
                            "a fixed 3072-step extension. Adam is restarted at lr=3e-4 for the extension. "
                            "No extra DEV fitting and no full P2 training.")
        summary["expression_diagnostic_sha256"] = sha(OUT / "EXPRESSION_SUMMARY.json")
        atomic_json(OUT / "SUMMARY.json", summary)
    lines = ["# SmoothVP-Net CPU preflight", "",
             "| Coordinates | Real parameters | Initial clean NMSE | TRAIN16 high-JNR residual ratio | TRAIN16 Pfa decrease | Result |",
             "|---|---:|---:|---:|---:|---|"]
    for key, row in summary["coordinates"].items():
        lines.append(f"| {key} | {row['health']['effective_real_parameters']} | "
                     f"{row['health']['clean_pooled_nmse']:.6g} | {row['train16_high_jnr_residual_ratio']:.6g} | "
                     f"{row['train16_pfa_decrease']:.3%} | {row['status']} |")
    lines += ["", "All 192-step and 768-step results are preserved in attempt_001 and attempt_002.",
              "Final CPU budget is 3840 updates per coordinate. The convergence extension restarts Adam at lr=3e-4.",
              "Each result uses the final iteration. The full P2 model is initialized afresh.", ""]
    for key, row in summary["coordinates"].items():
        lines += [f"## {key} worst TRAIN16 parents", ""]
        write_cases(lines, row["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    update(stage="COMPLETE", ended_utc=now(), summary_sha256=sha(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    with ProcessLock(PRIVATE / "EXTENSION.lock"):
        run()
