"""Immutable 256-epoch SmoothVP seed runner, admitted by the CPU controller."""
import os
from pathlib import Path
import time

import numpy as np
import torch

from vp_1103_common import WORK, atomic_json, now, read, sha
from vp_1103_p2_smoothvp import SmoothVPNet, real_parameters
from vp_1103_p2_evaluate import evaluate_seed
from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import learning_rate, optimizer_for, seed_rng, rng_snapshot, restore_rng
from tools.correction_tail_study import objective as objective_module
from tools.correction_tail_study.train import FORMAT, choose_resume, publish
from tools.durable_io import ProcessLock
from tools.unet_ablation.runtime import configure_precision, user_control

OUT = WORK / "evidence/outputs/p2_smoothvp_screen"
PRIVATE = WORK / "outputs/p2_smoothvp_screen"


def train(seed, coordinates, train_path, dev, config, update, arm="M3-SmoothVP"):
    folder = PRIVATE / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"run": "p2_smoothvp_screen", "arm": arm, "seed": seed,
                "config_sha256": sha(PRIVATE / "CONFIG.json")}
    with ProcessLock(folder / "PROCESS.lock"):
        complete = OUT / arm / f"seed_{seed}.json"
        if complete.exists():
            row = read(complete)
            if sha(folder / "epoch_256.pt") != row["checkpoint_sha256"]:
                raise ValueError("Completed model changed")
            return row
        configure_precision()
        torch.set_num_threads(2)
        seed_rng(seed)
        model = SmoothVPNet(coordinates, fixed_lambda=arm == "M3-FixedLambda").cuda()
        optimizer = optimizer_for(model)
        saved = choose_resume(folder, identity)
        start = 0
        if saved:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
            start = int(saved["epoch"])

        def save(epoch, record):
            return publish(folder, {"format": FORMAT, "identity": identity, "seed": seed,
                                    "epoch": epoch, "global_step": epoch * 48, "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(), "rng": rng_snapshot(), "epoch_record": record})

        if not saved:
            save(0, None)
        training = load_arrays(train_path)
        objective_module.ARM = "tail"
        for epoch in range(start + 1, 257):
            user_control(PRIVATE)
            model.train()
            totals = {}
            started = time.perf_counter()
            for step, indices in enumerate(batches(512, epoch, seed)):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(epoch, step)
                optimizer.zero_grad(set_to_none=True)
                for lo in range(0, len(indices), 8):
                    part = indices[lo:lo + 8]
                    batch = make_batch(training, part, "cuda")
                    correction = model(batch["y"]).reshape(-1, 4, 2, 1800)
                    losses = objective_module.study_objective(correction, batch, len(part))
                    if not all(torch.isfinite(value).all() for value in losses.values()):
                        raise FloatingPointError(f"{arm} seed {seed} epoch {epoch} nonfinite")
                    weight = len(part) / len(indices)
                    (weight * losses["total"]).backward()
                    for key, value in losses.items():
                        totals[key] = totals.get(key, 0.) + float(value.detach()) * weight / 48
                    del batch, correction, losses
                if epoch == 1 and step == 0:
                    active = sum(p.numel() * (2 if p.is_complex() else 1)
                                 for p in model.parameters() if p.grad is not None)
                    if active != real_parameters(model):
                        raise ValueError("An in-graph parameter is disconnected")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                if step % 12 == 0:
                    update(stage="TRAINING", arm=arm, seed=seed, epoch=epoch, step=step + 1,
                           last_committed_epoch=epoch - 1)
            torch.cuda.synchronize()
            record = {"run": "p2_smoothvp_screen", "arm": arm, "seed": seed, "epoch": epoch,
                      "global_step": epoch * 48, "epoch_seconds": time.perf_counter() - started, **totals}
            digest = save(epoch, record)
            update(stage="TRAINING", arm=arm, seed=seed, epoch=epoch, last_committed_epoch=epoch,
                   checkpoint_sha256=digest, epoch_seconds=record["epoch_seconds"])
        for epoch in range(257):
            cp = folder / f"recovery/epoch_{epoch:03d}.pt"
            if sha(cp) != read(cp.with_suffix(".json"))["checkpoint_sha256"]:
                raise ValueError(f"Recovery audit failed at epoch {epoch}")
        update(stage="EVALUATING", arm=arm, seed=seed, epoch=256)
        row = evaluate_seed(model.eval(), dev, seed, arm, coordinates, folder / "epoch_256.pt")
        atomic_json(folder / "TRAINING_COMPLETE.json", {
            "identity": identity, "epochs": 256, "updates": 12288, "ended_utc": now(),
            "checkpoint_sha256": row["checkpoint_sha256"], "active_real_parameters": real_parameters(model)})
        del model, optimizer, training
        torch.cuda.empty_cache()
        return row
