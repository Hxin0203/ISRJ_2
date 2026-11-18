"""Paired CPU-only output-distillation experiment on public 8x8 handwriting images."""

import copy
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_hash(path):
    return digest(path.read_text(encoding="utf-8").encode("utf-8"))


class DigitUNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(3 * width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, image):
        fine = self.enc(image)
        coarse = self.mid(F.avg_pool2d(fine, 2))
        fused = torch.cat((fine, F.interpolate(coarse, scale_factor=2, mode="nearest")), dim=1)
        return image + self.dec(fused)


def noisy(clean, rng):
    sample = torch.randn(clean.shape, generator=rng)
    return (clean + CFG["sigma"] * sample).clamp(0, 1)


def load_splits():
    digits = load_digits()
    pixels = digits.images.astype(np.uint8)
    labels = digits.target.astype(np.uint8)
    ids = np.arange(len(labels))
    if len(np.unique(pixels.reshape(len(ids), -1), axis=0)) != len(ids):
        raise ValueError("Dataset contains duplicate parent images; review the split protocol")
    train_dev, test = train_test_split(
        ids, test_size=CFG["test_parents"], random_state=CFG["split_seed"], stratify=labels,
    )
    train, dev = train_test_split(
        train_dev, test_size=CFG["dev_parents"], random_state=CFG["split_seed"] + 1,
        stratify=labels[train_dev],
    )
    images = torch.from_numpy(pixels).float().unsqueeze(1) / 16
    evaluation = {}
    for name, indices in (("dev", dev), ("test", test)):
        clean = images[indices]
        rng = torch.Generator().manual_seed(CFG[f"{name}_noise_seed"])
        evaluation[name] = (noisy(clean, rng), clean)
    provenance = {
        "data_sha256_uint8_images_then_labels": digest(pixels.tobytes() + labels.tobytes()),
        "split_sha256_ordered_image_bytes": {
            name: digest(pixels[idx].tobytes())
            for name, idx in (("train", train), ("dev", dev), ("test", test))
        },
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
    }
    return images[train], evaluation, provenance


def train_models(model, clean_images, seed, checkpoints, teacher=None):
    rng = torch.Generator().manual_seed(20000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    saved = {}
    for step in range(1, max(checkpoints) + 1):
        idx = torch.randint(len(clean_images), (CFG["batch_size"],), generator=rng)
        clean = clean_images[idx]
        inputs = noisy(clean, rng)
        prediction = model(inputs)
        target = clean
        if teacher is not None:
            with torch.no_grad():
                target = torch.lerp(clean, teacher(inputs), CFG["teacher_weight"])
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            saved[step] = copy.deepcopy(model).eval()
    return saved


@torch.inference_mode()
def image_scores(model, inputs, clean):
    prediction = model(inputs).clamp(0, 1)
    mse = (prediction - clean).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    edges = []
    for direction in (2, 3):
        error = prediction.diff(dim=direction) - clean.diff(dim=direction)
        edges.append(error.abs().mean(dim=(1, 2, 3)))
    return {"psnr_db": (-10 * mse.log10()).numpy(), "gradient_mae": torch.stack(edges).mean(0).numpy()}


def summarize(scores):
    return {key: float(value.mean(dtype=np.float64)) for key, value in scores.items()}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train, evaluation, provenance = load_splits()
    rows, paired_test = [], []
    for seed in CFG["train_seeds"]:
        torch.manual_seed(seed)
        student = DigitUNet(CFG["student_width"])
        torch.manual_seed(10000 + seed)
        teacher = DigitUNet(CFG["teacher_width"])
        teacher = train_models(teacher, train, seed, [CFG["teacher_steps"]])[CFG["teacher_steps"]]
        teacher.requires_grad_(False)
        normal = train_models(copy.deepcopy(student), train, seed, [CFG["student_steps"], CFG["long_student_steps"]])
        kd = train_models(copy.deepcopy(student), train, seed, [CFG["student_steps"]], teacher)[CFG["student_steps"]]
        models = {
            "input": nn.Identity(), "teacher": teacher, "vanilla": normal[CFG["student_steps"]],
            "distilled": kd, "long_vanilla": normal[CFG["long_student_steps"]],
        }
        row = {"seed": seed}
        for split, (inputs, clean) in evaluation.items():
            scores = {name: image_scores(model, inputs, clean) for name, model in models.items()}
            row[split] = {name: summarize(value) for name, value in scores.items()}
            if split == "test":
                paired_test.append(scores["distilled"]["psnr_db"] - scores["vanilla"]["psnr_db"])
        rows.append(row)
        print(f"Completed paired arms for seed {seed}", flush=True)

    gains = [float(values.mean(dtype=np.float64)) for values in paired_test]
    parent_gain = np.stack(paired_test).mean(axis=0, dtype=np.float64)
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    draws = rng.integers(len(parent_gain), size=(CFG["bootstrap_draws"], len(parent_gain)))
    ci = np.quantile(parent_gain[draws].mean(axis=1), [0.025, 0.975])
    means = {
        split: {
            name: {
                metric: float(np.mean([row[split][name][metric] for row in rows]))
                for metric in ("psnr_db", "gradient_mae")
            } for name in models
        } for split in evaluation
    }
    test = means["test"]
    relative_edge_error = test["distilled"]["gradient_mae"] / test["vanilla"]["gradient_mae"] - 1
    gate = CFG["pilot_criteria"]
    passed = {
        "teacher_beats_vanilla_every_dev_seed": all(
            row["dev"]["teacher"]["psnr_db"] > row["dev"]["vanilla"]["psnr_db"] for row in rows
        ),
        "test_psnr_gain_positive_every_seed": all(gain > 0 for gain in gains),
        "mean_gain_meets_minimum": np.mean(gains).item() >= gate["min_mean_test_psnr_gain_db"],
        "gradient_mae_within_margin": relative_edge_error <= gate["max_relative_test_gradient_mae_increase"],
    }
    report = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": np.__version__, "sklearn": sklearn.__version__, "device": "CPU, 1 thread"},
        "hashes_utf8_lf_sha256": {name: text_hash(HERE / name) for name in ("protocol.json", "run.py")},
        "dataset": provenance,
        "parameters": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        "per_seed": rows, "mean_of_seed_means": means,
        "test_paired_psnr_gain_db_by_seed": dict(zip(map(str, CFG["train_seeds"]), gains)),
        "test_mean_paired_psnr_gain_db": float(parent_gain.mean()),
        "test_parent_paired_ci95_db_conditional_on_trained_seeds": ci.tolist(),
        "test_relative_gradient_mae_change": relative_edge_error,
        "test_distilled_minus_long_vanilla_psnr_db": test["distilled"]["psnr_db"] - test["long_vanilla"]["psnr_db"],
        "criteria": passed, "passes_pilot": all(passed.values()),
    }
    if not all(math.isfinite(value) for value in gains):
        raise FloatingPointError("Nonfinite evaluation contrast")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pilot gates: {passed}", flush=True)


if __name__ == "__main__":
    main()
