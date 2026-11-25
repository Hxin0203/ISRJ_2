"""CPU-only numerical and small-training-set checks before P2 admission."""
import argparse
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from vp_1103_common import (WORK, atomic_json, frame_jnr, now, parent_roster,
                           provenance, read, sha, stratified, validate_role, worst_cases, write_cases)
from vp_1103_p2_smoothvp import SmoothVPNet, real_parameters
from vp_1103_p2_admission import fit_evidence
from vp_1103_p0 import evaluate
from paper2.train_1103 import make_batch
from paper2.train_256_8seed import seed_rng, atomic_torch_save
from tools.correction_tail_study import objective as objective_module
from tools.durable_io import ProcessLock, atomic_write

OUT = WORK / "evidence/outputs/p2_smoothvp_preflight"
PRIVATE = WORK / "outputs/p2_smoothvp_preflight"
OVERFIT_STEPS = 768
OVERFIT_THRESHOLDS = {"paired_high_jnr_relative_change_ci95_upper_max_exclusive": 0.,
                      "paired_frame_pfa_decrease_ci95_lower_min_exclusive": 0.}


def load(role):
    path, identity = validate_role(role)
    with np.load(path, allow_pickle=False) as z:
        parent_roster(z, 512 if role == "train" else 256)
        arrays = {key: z[key] for key in z.files}
    return arrays, identity


def subset(data, indices):
    return {key: array[indices] if array.ndim and len(array) == len(data["s"]) else array
            for key, array in data.items()}


def representative(data, count=16):
    jnr = frame_jnr(data)
    # Four evenly spaced high-risk groups, four closest to 45 dB, and eight
    # across the full range. Only train/DEV stratification metadata is used.
    strength = jnr.max(axis=1)
    high = np.where(strength >= 35.)[0]
    ordered = high[np.argsort(strength[high])]
    indices = list(ordered[np.linspace(0, len(ordered) - 1, 4).astype(int)])
    indices.extend(np.argsort(abs(strength - 45.))[:4])
    ranking = np.argsort(strength)
    indices.extend(ranking[np.linspace(0, len(ranking) - 1, 32).astype(int)])
    return np.asarray(list(dict.fromkeys(map(int, indices)))[:count])


def health(model, data, indices):
    gradients = {}
    head_gradients = {key: [] for key in ("delay", "mask", "log_precision")}
    keep = []
    clean_error, clean_energy = 0., 0.
    model.eval()
    objective_module.ARM = "tail"
    for parent in indices:
        # All three families and all eight pulses in all four states.
        for family in range(3):
            batch = make_batch(data, np.array([parent * 3 + family]), "cpu")
            model.zero_grad(set_to_none=True)
            parts = model.factor(batch["y"])
            for key in head_gradients:
                parts[key].retain_grad()
            correction = torch.stack((parts["correction"].real.float(), parts["correction"].imag.float()), 1)
            terms = objective_module.study_objective(correction.reshape(-1, 4, 2, 1800), batch, 1)
            terms["total"].backward()
            if not torch.isfinite(correction).all():
                raise FloatingPointError("nonfinite initial prediction")
            for name, p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise FloatingPointError(f"Missing or nonfinite gradient for {name}")
                gradients[name] = max(gradients.get(name, 0.), float(p.grad.abs().max()))
            for key in head_gradients:
                gradient = parts[key].grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Missing or nonfinite intermediate gradient for {key}")
                head_gradients[key].append(float(gradient.abs().max()))
            c = correction.detach().reshape(8, 4, 2, 1800)
            y = batch["y"].reshape(8, 4, 2, 1800)
            clean_error += float(c[:, :2].square().sum())
            clean_energy += float(y[:, :2].square().sum())
            keep.append(float(parts["precision"].detach().min()))
    return {"effective_real_parameters": real_parameters(model), "gradient_max_by_parameter": gradients,
            "gradient_max_by_head": {key: max(value) for key, value in head_gradients.items()},
            "all_gradients_finite_and_connected": True,
            "all_parameter_tensors_have_nonzero_gradient": all(value > 0 for value in gradients.values()),
            "clean_pooled_nmse": clean_error / clean_energy, "minimum_initial_precision": min(keep)}


def measure(model, data):
    # Existing evaluator uses the actual subset length, three types, paired states.
    metrics = evaluate(model.eval(), data)
    return metrics


def coordinate_test(coordinates, dev, training, update):
    folder = PRIVATE / coordinates
    folder.mkdir(parents=True, exist_ok=True)
    indices_dev, indices_train = representative(dev), representative(training)
    seed_rng(17)
    model = SmoothVPNet(coordinates).cpu()
    update(coordinates=coordinates, phase="GRADIENT_HEALTH")
    diagnostics = health(model, dev, indices_dev)
    train_small = subset(training, indices_train)
    before = measure(model, train_small)
    no_jammer = float(before["no_jammer_pooled_nmse"].mean())
    count = diagnostics["effective_real_parameters"]
    prerequisites = (56728 <= count <= 62700 and
                     diagnostics["all_parameter_tensors_have_nonzero_gradient"] and
                     diagnostics["clean_pooled_nmse"] <= 1e-4 and no_jammer <= 1e-4)
    records = np.arange(len(indices_train) * 3)
    rng = np.random.default_rng(20261125)
    schedule = np.concatenate([rng.permutation(records) for _ in range(OVERFIT_STEPS // len(records) + 1)])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trajectory = []
    objective_module.ARM = "tail"
    for step, record in enumerate(schedule[:OVERFIT_STEPS], 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(train_small, np.array([record]), "cpu")
        correction = model(batch["y"]).reshape(-1, 4, 2, 1800)
        terms = objective_module.study_objective(correction, batch, 1)
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError("Overfit objective nonfinite")
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        trajectory.append({key: float(value.detach()) for key, value in terms.items()})
        if step % 8 == 0:
            update(coordinates=coordinates, phase="CPU_SMALL_TRAIN_CHECK", step=step,
                   budget=OVERFIT_STEPS, losses=trajectory[-1])
            atomic_json(folder / "TRAJECTORY.json", trajectory)
    after = measure(model, train_small)
    # Freeze one model after the entire fixed budget. Full P2 always starts fresh.
    atomic_torch_save({"model": model.state_dict(), "coordinates": coordinates,
                      "purpose": "TRAIN16_CPU_ENGINEERING_ONLY", "steps": OVERFIT_STEPS,
                      "parent_ids": train_small["parent_id"].tolist()}, folder / "overfit.pt")
    target = folder / "paired_metrics.npz"
    atomic_write(target, lambda stream: np.savez_compressed(
        stream, parent_id=train_small["parent_id"],
        **{f"before__{k}": v for k, v in before.items()},
        **{f"after__{k}": v for k, v in after.items()}))
    jnr = frame_jnr(train_small)
    high = jnr >= 35
    residual_ratio = float(after["background_nmse"][high].mean() / before["background_nmse"][high].mean())
    pfa_decrease = float((before["frame_pfa_jammer"] - after["frame_pfa_jammer"]).mean())
    fit = fit_evidence(before, after, jnr)
    expression_pass = fit["passes"]
    checks = {"budget": "PASS" if 56728 <= count <= 62700 else "FAIL",
              "finite_and_nonzero_gradients": "PASS" if prerequisites else "FAIL",
              "initial_no_jammer": "PASS" if max(no_jammer, diagnostics["clean_pooled_nmse"]) <= 1e-4 else "FAIL",
              "train16_expression": "PASS" if expression_pass else "FAIL"}
    return {"checks": checks, "status": "PASS" if prerequisites and expression_pass else "FAIL",
            "health": diagnostics, "initial_train16_no_jammer_nmse": no_jammer,
            "train16_high_jnr_residual_ratio": residual_ratio,
            "train16_pfa_decrease": pfa_decrease, "overfit_steps": OVERFIT_STEPS,
            "fit_evidence": fit,
            "before": stratified(before, jnr), "after": stratified(after, jnr),
            "worst_cases": worst_cases(after, train_small["parent_id"], jnr),
            "private_metrics_sha256": sha(target),
            "selected_dev_parent_ids": dev["parent_id"][indices_dev].tolist(),
            "selected_dev_frame_jnr": frame_jnr(dev)[indices_dev].tolist(),
            "selected_train_parent_ids": train_small["parent_id"].tolist()}


def run(coordinates):
    torch.set_num_threads(2)
    # Process-level CUDA mask is set by the launcher. Every tensor stays on CPU.
    dev, dev_receipt = load("dev")
    training, train_receipt = load("train")
    status = {**provenance(__file__, Path(__file__).with_name("vp_1103_p2_smoothvp.py")),
              "stage": "RUNNING", "started_utc": now(), "ended_utc": None,
              "data": {"train": train_receipt, "dev": dev_receipt},
              "device": "cpu", "pid": os.getpid(), "seed": 17,
              "micro_batch_records": 1, "overfit_steps": OVERFIT_STEPS}

    def update(**fields):
        status.update(fields, updated_utc=now())
        atomic_json(OUT / "STATUS.json", status)

    update()
    summary = {"coordinates": {}, "thresholds": OVERFIT_THRESHOLDS, "p2_training_started": False,
               "scope": "CPU finite-gradient check and fixed 768-step TRAIN16 optimization. "
                        "No DEV optimization and no full P2 budget.",
               "solver_dtype": "complex128", "writeback_dtype": "complex64"}
    for coord in coordinates:
        summary["coordinates"][coord] = coordinate_test(coord, dev, training, update)
        atomic_json(OUT / "SUMMARY.json", summary)
        lines = ["# SmoothVP-Net CPU preflight", "",
                 "| Coordinates | Real parameters | Initial clean NMSE | TRAIN16 high-JNR residual ratio | TRAIN16 Pfa decrease | Result |",
                 "|---|---:|---:|---:|---:|---|"]
        for key, row in summary["coordinates"].items():
            lines.append(f"| {key} | {row['health']['effective_real_parameters']} | "
                         f"{row['health']['clean_pooled_nmse']:.6g} | {row['train16_high_jnr_residual_ratio']:.6g} | "
                         f"{row['train16_pfa_decrease']:.3%} | {row['status']} |")
            lines += ["", f"## {key} worst TRAIN16 parents", ""]
            write_cases(lines, row["worst_cases"])
        lines += ["", "The full model starts from a fresh seed. A failed expression check blocks GPU training."]
        atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    update(stage="COMPLETE", ended_utc=now(), summary_sha256=sha(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", nargs="+", choices=("RAW_IQ", "DECHIRP"), default=["RAW_IQ", "DECHIRP"])
    with ProcessLock(PRIVATE / "PROCESS.lock"):
        try:
            run(parser.parse_args().coordinates)
        except Exception as exc:
            prior = read(OUT / "STATUS.json")
            atomic_json(OUT / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "utc": now()})
            atomic_json(OUT / "STATUS.json", {**prior, "stage": "FAILED", "ended_utc": now(), "error": repr(exc)})
            raise
