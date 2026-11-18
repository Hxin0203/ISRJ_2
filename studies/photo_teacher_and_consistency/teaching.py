"""Fit, freeze, then evaluate a horizontal-mirror teacher on new photographic accounts."""

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from photo_study import (
    CFG, HERE, PREVIOUS, DigitUNet, average, block_part, consistent, cpu_setup, crop_batch,
    lift, load_photos, paired, photo_scores, predict, save_json, setup_model, text_hash, update,
)

SPEC = json.loads((HERE / "teaching_protocol.json").read_text(encoding="utf-8"))
FIXED = tuple(list(SPEC["arms"])[:5])
CONTROLS = tuple(arm for arm in SPEC["arms"] if arm != "mean_orientation")


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_model(path, sha, width):
    if fingerprint(path) != sha:
        raise ValueError(f"Checkpoint identity changed for {path.name}")
    model = DigitUNet(width)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False)


def sources(dev_path):
    return {p.name: text_hash(p) for p in (HERE / "protocol.json", HERE / "teaching_protocol.json",
            PREVIOUS / "run.py", HERE / "photo_study.py", Path(__file__), dev_path)}


def step(model, opt, teacher, clean, mode, orientation_rng):
    if mode == "augmented_plain":
        flip = torch.rand((len(clean), 1, 1, 1), generator=orientation_rng) < 0.5
        clean = torch.where(flip, clean.flip((-1,)), clean)
    if mode in ("plain", "augmented_plain"):
        update(model, opt, clean, train_detail=True)
        return
    up = block_part(clean)
    with torch.no_grad():
        native = consistent(teacher(up), up)
        if mode == "ordinary":
            tutor = native
        else:
            mirrored = consistent(teacher(up.flip((-1,))).flip((-1,)), up)
            if mode == "mean_orientation":
                tutor = (native + mirrored) / 2
            elif mode == "random_orientation":
                flip = torch.rand((len(clean), 1, 1, 1), generator=orientation_rng) < 0.5
                tutor = torch.where(flip, mirrored, native)
            else:
                raise ValueError(mode)
        mixed = torch.lerp(clean, tutor, SPEC["teacher_weight"])
    loss = F.mse_loss(consistent(model(up), up), mixed)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite teaching loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def fit_student(images, teacher, seed, mode, deadlines=None, replay=None):
    def fresh():
        model = setup_model(CFG["student_width"], seed)
        return model, torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])

    warm, opt = fresh()
    warm_rng = torch.Generator().manual_seed(270000 + seed)
    for _ in range(CFG["warmup_steps"]):
        step(warm, opt, teacher, crop_batch(images, warm_rng), mode, warm_rng)
    model, opt = fresh()
    crops = torch.Generator().manual_seed(CFG["patch_seed_offset"] + seed)
    orientations = torch.Generator().manual_seed(SPEC["orientation_seed_offset"] + seed)
    models, times = {}, {}
    start, paused = time.perf_counter(), 0.0
    for i in itertools.count(1):
        step(model, opt, teacher, crop_batch(images, crops), mode, orientations)
        now = time.perf_counter()
        elapsed = now - start - paused
        if deadlines is None:
            if i == SPEC["updates"]:
                return {mode: model.eval()}, {mode: {"steps": i, "seconds": elapsed}}
            continue
        for arm, target in deadlines.items():
            reached = i == replay[arm]["steps"] if replay else elapsed >= target
            if reached and arm not in models:
                models[arm] = copy.deepcopy(model).eval()
                times[arm] = {"steps": i, "seconds": elapsed, "target_seconds": target}
        if len(models) == len(deadlines):
            return models, times
        paused += time.perf_counter() - now


def fit_all(args):
    dev = json.loads(args.development.read_text(encoding="utf-8"))
    if not dev["teacher_qualifies"]:
        raise ValueError("Wide teacher did not meet the prespecified development decision")
    data, _, provenance = load_photos(args.archive, ("train",))
    if provenance["splits"]["train"] != dev["dataset"]["splits"]["train"]:
        raise ValueError("Training population changed")
    args.scratch.mkdir(parents=True, exist_ok=True)
    replay = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else None
    freeze = {"source_sha256": sources(args.development), "accounts": provenance["splits"],
              "decoded_in_followup_fit": ["train"], "teaching_mode": "recorded steps" if replay else "live",
              "fits": {}, "teacher_sha256": {}, "checkpoint_sha256": {}}
    for i, seed in enumerate(CFG["seeds"]):
        key = str(seed)
        filename = f"wide-{seed}.pt"
        teacher_sha = dev["checkpoint_sha256"][filename]
        teacher = read_model(args.dev_scratch / filename, teacher_sha, CFG["teacher_width"])
        freeze["teacher_sha256"][filename] = teacher_sha
        order = FIXED[i % len(FIXED):] + FIXED[:i % len(FIXED)]
        times = {"fixed_arm_order": list(order)}
        models = {}
        for mode in order:
            snapshots, durations = fit_student(data["train"], teacher, seed, mode)
            times.update(durations)
            models.update(snapshots)
        marginal = times["mean_orientation"]["seconds"]
        teacher_time = dev["timing"][key]["wide"]["timed"]["seconds"]
        limits = {"plain_marginal_time": marginal, "plain_total_time": teacher_time + marginal}
        replay_times = replay["fits"][key] if replay else None
        snapshots, durations = fit_student(data["train"], teacher, seed, "plain", deadlines=limits, replay=replay_times)
        times.update(durations)
        models.update(snapshots)
        freeze["fits"][key] = times
        for arm, model in models.items():
            path = args.scratch / f"{arm}-{seed}.pt"
            torch.save(model.state_dict(), path)
            freeze["checkpoint_sha256"][path.name] = fingerprint(path)
        print(f"Froze student arms for seed {seed}, candidate {marginal:.2f}s, one-off budget {teacher_time+marginal:.2f}s", flush=True)
    save_json(args.manifest, freeze)


@torch.inference_mode()
def timed_inference(plain, candidate, low):
    samples = {"plain": [], "mean_orientation": []}
    names = tuple(samples)
    models = {"plain": plain, "mean_orientation": candidate}
    for i in range(80):
        for key in names if i % 2 == 0 else reversed(names):
            start = time.perf_counter_ns()
            up = lift(low)
            consistent(models[key](up), up)
            ms = (time.perf_counter_ns() - start) / 1e6
            if i >= 10:
                samples[key].append(ms)
    return samples


def score(args):
    freeze = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sources(args.development) != freeze["source_sha256"]:
        raise ValueError("Source or development evidence changed after fit freeze")
    dev = json.loads(args.development.read_text(encoding="utf-8"))
    # Verify every active checkpoint before decoding the account holdout.
    for folder, recorded in ((args.scratch, freeze["checkpoint_sha256"]),
                             (args.dev_scratch, freeze["teacher_sha256"])):
        for filename, expected in recorded.items():
            if fingerprint(folder / filename) != expected:
                raise ValueError(f"Frozen checkpoint changed: {filename}")
    data, _, provenance = load_photos(args.archive, ("train", "A", "B", "C"))
    for split, original in dev["dataset"]["splits"].items():
        if any(provenance["splits"][split][k] != v for k, v in original.items()):
            raise ValueError(f"Changed account membership or pixels for {split}")
    clean = data["C"].double()
    up = block_part(clean)
    bicubic = F.interpolate(F.avg_pool2d(clean, 2), scale_factor=2, mode="bicubic", align_corners=False)
    base = {"bicubic": photo_scores(bicubic, clean), "bicubic_dc": photo_scores(consistent(bicubic, up), clean)}
    rows, orientation = [], {}
    latency = {"plain": [], "mean_orientation": []}
    for seed in CFG["seeds"]:
        row = dict(base)
        frozen_students = {}
        for arm in SPEC["arms"]:
            filename = f"{arm}-{seed}.pt"
            model = read_model(args.scratch / filename, freeze["checkpoint_sha256"][filename], CFG["student_width"])
            row[arm] = photo_scores(consistent(predict(model, up), up), clean)
            frozen_students[arm] = model
        lat = timed_inference(frozen_students["plain"], frozen_students["mean_orientation"], F.avg_pool2d(data["train"][:1], 2))
        for arm, values in lat.items():
            latency[arm].extend(values)
        filename = f"wide-{seed}.pt"
        teacher = read_model(args.dev_scratch / filename, freeze["teacher_sha256"][filename], CFG["teacher_width"])
        native = consistent(predict(teacher, up), up)
        flipped = consistent(predict(teacher, up.flip((-1,))).flip((-1,)), up)
        midpoint = (native + flipped) / 2
        row["teacher_native"] = photo_scores(native, clean)
        row["teacher_mean"] = photo_scores(midpoint, clean)
        gap = ((native - flipped).square() / 4).mean((1, 2, 3))
        observed_gap = (((native - clean).square() + (flipped - clean).square()) / 2 - (midpoint - clean).square()).mean((1, 2, 3))
        err = (observed_gap - gap).abs().max().item()
        if err > 1e-12:
            raise ArithmeticError("Orientation-averaging MSE identity failed")
        orientation[str(seed)] = {"mean_random_orientation_minus_mean_teacher_mse": float(gap.mean()),
                                  "max_abs_identity_error": err}
        if dev["detail_training_qualifies"]:
            for name in ("small", "detail-matched", "detail-timed"):
                filename = f"{name}-{seed}.pt"
                model = read_model(args.dev_scratch / filename, dev["checkpoint_sha256"][filename], CFG["student_width"])
                row[name] = photo_scores(consistent(predict(model, up), up), clean)
        rows.append(row)
    comparisons = {arm: paired(rows, "mean_orientation", arm) for arm in CONTROLS}
    lat_summary = {arm: {"n": len(v), "median_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, 0.95))}
                   for arm, v in latency.items()}
    latency_ratio = lat_summary["mean_orientation"]["median_ms"] / lat_summary["plain"]["median_ms"]
    max_time_error = max(abs(d[arm]["seconds"] / d[arm]["target_seconds"] - 1)
                         for d in freeze["fits"].values() for arm in ("plain_marginal_time", "plain_total_time"))
    gates = {"quality_all_controls": all(c["passes"] for c in comparisons.values()),
             "time_controls_matched": max_time_error <= CFG["max_wall_relative_difference"],
             "latency_within_margin": latency_ratio <= 1.10}
    result = {"data": provenance, "pre_scoring_manifest_sha256": text_hash(args.manifest),
              "source_sha256": freeze["source_sha256"], "timing_mode": freeze["teaching_mode"],
              "environment": dev["environment"], "timing_max_relative_error": max_time_error,
              "inference_latency": lat_summary, "candidate_over_plain_latency": latency_ratio,
              "ordinary_teacher_to_mean_teacher": paired(rows, "teacher_mean", "teacher_native"),
              "orientation_mse_check": orientation, "candidate_minus_controls": comparisons,
              "C_means": {arm: {metric: float(np.mean([r[arm][metric] for r in rows]))
                                for metric in rows[0][arm]} for arm in rows[0]},
              "per_seed_means": {str(seed): {arm: average(v) for arm, v in row.items()}
                                 for seed, row in zip(CFG["seeds"], rows)},
              "criteria": gates, "teaching_passes": all(gates.values()),
              "earlier_detail_training_eligible": dev["detail_training_qualifies"]}
    if dev["detail_training_qualifies"]:
        result["direct_training_C"] = {arm: paired(rows, arm, "small") for arm in ("detail-matched", "detail-timed")}
    save_json(args.output, result)
    print(f"C teaching decision: {gates}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--development", type=Path, default=HERE / "results.json")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--dev-scratch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path, help="Fit time controls at steps recorded in an earlier freeze")
    args = parser.parse_args()
    cpu_setup()
    if args.phase == "fit":
        fit_all(args)
    else:
        if args.output is None:
            parser.error("score requires --output")
        score(args)
