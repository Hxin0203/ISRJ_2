"""Test a character-class teacher selector on a disjoint public handwriting cohort."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.io import arff
from sklearn.datasets import load_digits
from torch import nn

from run import CFG, HERE, DigitUNet, digest, image_scores, load_splits, noisy, summarize, text_hash, train_models


SPEC = json.loads((HERE / "selection_protocol.json").read_text(encoding="utf-8"))


def external_cohort(path):
    if digest(path.read_bytes()) != SPEC["file_sha256"]:
        raise ValueError("Unexpected OpenML file identity")
    data, metadata = arff.loadarff(path)
    cols = metadata.names()
    pixels = np.column_stack([data[col] for col in cols[:-1]]).astype(np.uint8).reshape(-1, 8, 8)
    labels = data[cols[-1]].astype(np.uint8)
    known = load_digits()
    if not (np.array_equal(pixels[-1797:], known.images) and np.array_equal(labels[-1797:], known.target)):
        raise ValueError("OpenML data order or original cohort identity changed")
    if len(pixels) != 5620 or len(np.unique(pixels.reshape(len(pixels), -1), axis=0)) != len(pixels):
        raise ValueError("Unexpected parent count or duplicate parent images")
    external = pixels[:-1797]
    clean = torch.from_numpy(external).float().unsqueeze(1) / 16
    rng = torch.Generator().manual_seed(SPEC["external_noise_seed"])
    return noisy(clean, rng), clean, digest(external.tobytes())


def original_digit_labels(images):
    known = load_digits()
    lookup = {image.tobytes(): int(y) for image, y in zip(known.images.astype(np.uint8), known.target)}
    pixels = (images[:, 0] * 16).numpy().astype(np.uint8)
    return torch.tensor([lookup[image.tobytes()] for image in pixels])


def fit_selected(model, train, labels, seed, teacher, chosen, mode):
    rng = torch.Generator().manual_seed(20000 + seed)
    shuffle = torch.Generator().manual_seed(SPEC["shuffle_mask_seed_offset"] + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    use_teacher = torch.isin(labels, torch.tensor(chosen))
    for _ in range(CFG["student_steps"]):
        idx = torch.randint(len(train), (CFG["batch_size"],), generator=rng)
        clean = train[idx]
        inputs = noisy(clean, rng)
        weights = use_teacher[idx].float() * CFG["teacher_weight"]
        if mode == "matched_uniform":
            weights = weights.mean().expand_as(weights)
        elif mode == "shuffled":
            weights = weights[torch.randperm(len(weights), generator=shuffle)]
        elif mode != "selected":
            raise ValueError(mode)
        with torch.no_grad():
            targets = torch.lerp(clean, teacher(inputs), weights.view(-1, 1, 1, 1))
        loss = F.mse_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite pixel MSE")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def external_scores(model, inputs, clean):
    chunks = [image_scores(model, batch, ref) for batch, ref in zip(inputs.split(128), clean.split(128))]
    return {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}


def compare(values, condition, control):
    diffs = np.stack([row[condition]["psnr_db"] - row[control]["psnr_db"] for row in values])
    mean_by_parent = diffs.mean(axis=0, dtype=np.float64)
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    sampled_means = []
    for offset in range(0, CFG["bootstrap_draws"], 100):
        count = min(100, CFG["bootstrap_draws"] - offset)
        idx = rng.integers(len(mean_by_parent), size=(count, len(mean_by_parent)))
        sampled_means.extend(mean_by_parent[idx].mean(axis=1))
    def mean_gradient(arm):
        return float(np.mean([row[arm]["gradient_mae"].mean(dtype=np.float64) for row in values]))

    gains = diffs.mean(axis=1, dtype=np.float64)
    edge_change = mean_gradient(condition) / mean_gradient(control) - 1
    return {
        "psnr_gain_db_by_seed": dict(zip(map(str, CFG["train_seeds"]), gains.tolist())),
        "mean_gain_db": float(mean_by_parent.mean()),
        "parent_ci95_db_conditional_on_seeds": np.quantile(sampled_means, [0.025, 0.975]).tolist(),
        "relative_gradient_mae_change": edge_change,
        "passes": bool(np.all(gains > 0) and mean_by_parent.mean() >= 0.10 and edge_change <= 0.01),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arff", type=Path, help="OpenML optical digits dataset 28v1 ARFF")
    parser.add_argument("--output", type=Path, default=HERE / "selection_results.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs, clean, cohort_hash = external_cohort(args.arff)
    train, evaluation, provenance = load_splits()
    train_labels = original_digit_labels(train)
    dev_inputs, dev_clean = evaluation["dev"]
    dev_labels = original_digit_labels(dev_clean).numpy()
    rows, parent_scores = [], []
    for seed in CFG["train_seeds"]:
        torch.manual_seed(seed)
        student = DigitUNet(CFG["student_width"])
        torch.manual_seed(10000 + seed)
        teacher = train_models(DigitUNet(CFG["teacher_width"]), train, seed, [CFG["teacher_steps"]])[CFG["teacher_steps"]]
        teacher.requires_grad_(False)
        checkpoints = [CFG["student_steps"], CFG["long_student_steps"]]
        normal = train_models(copy.deepcopy(student), train, seed, checkpoints)
        vanilla = normal[CFG["student_steps"]]
        advantage = image_scores(teacher, dev_inputs, dev_clean)["psnr_db"] - image_scores(vanilla, dev_inputs, dev_clean)["psnr_db"]
        by_class = {i: float(advantage[dev_labels == i].mean(dtype=np.float64)) for i in range(10)}
        chosen = sorted(by_class, key=lambda i: (-by_class[i], i))[:SPEC["top_character_classes"]]
        models = {
            "input": nn.Identity(), "teacher": teacher, "vanilla": vanilla,
            "long_vanilla": normal[CFG["long_student_steps"]],
            "full_kd": train_models(copy.deepcopy(student), train, seed, [CFG["student_steps"]], teacher)[CFG["student_steps"]],
        }
        for mode in ("matched_uniform", "shuffled", "selected"):
            models[mode] = fit_selected(copy.deepcopy(student), train, train_labels, seed, teacher, chosen, mode)
        scores = {name: external_scores(model, inputs, clean) for name, model in models.items()}
        parent_scores.append(scores)
        rows.append({
            "seed": seed, "selected_characters": chosen, "dev_teacher_advantage_db_by_character": by_class,
            "dev_mean_teacher_advantage_db": float(advantage.mean(dtype=np.float64)),
            "external": {name: summarize(value) for name, value in scores.items()},
        })
        print(f"Completed character-selection study for seed {seed}", flush=True)
    comparisons = {name: compare(parent_scores, "selected", name) for name in SPEC["required_controls"]}
    means = {
        name: {metric: float(np.mean([r["external"][name][metric] for r in rows]))
               for metric in ("psnr_db", "gradient_mae")}
        for name in rows[0]["external"]
    }
    report = {
        "source_sha256_utf8_lf": {name: text_hash(HERE / name) for name in ("run.py", "protocol.json", "selective_digits.py", "selection_protocol.json")},
        "scipy": scipy.__version__, "torch": str(torch.__version__),
        "public_arff_sha256": SPEC["file_sha256"], "external_parent_count": len(clean),
        "external_uint8_pixels_sha256": cohort_hash, "original_provenance": provenance,
        "per_seed": rows, "external_means": means, "selective_vs_controls": comparisons,
        "passes_pilot": all(item["passes"] for item in comparisons.values())
                        and all(r["dev_mean_teacher_advantage_db"] > 0 for r in rows),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
