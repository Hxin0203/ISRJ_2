"""Run the two missing 256-epoch CorrectionNet coordinates-by-loss cells."""
import argparse
import hashlib
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from vp_1103_common import (DATA, HISTORICAL, SEEDS, SNAPSHOT, WORK, atomic_json,
                           distribution, frame_jnr, now, parent_roster, provenance,
                           read, sha, stratified, validate_role, worst_cases, write_cases)

os.environ["ISRJ2_CORRECTION_ARM"] = "original"
from paper2.evaluation_1103_v4 import (
    group_metrics, radar_metrics, restore_model, states_from_components)
from paper2.network import CorrectionNet
from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import learning_rate, optimizer_for, restore_rng, rng_snapshot, seed_rng
from tools.correction_tail_study import objective as frozen_objective
from tools.correction_tail_study.train import FORMAT, choose_resume, publish
from tools.durable_io import ProcessLock, atomic_write
from tools.unet_ablation.models import make_model
from tools.unet_ablation.runtime import configure_precision, user_control


RUN = "p0_dechirp_tail_2x2"
OUT = WORK / "evidence" / "outputs" / RUN
PRIVATE = WORK / "outputs" / RUN
ARMS = ("P0-NoDechirp-Base", "P0-NoDechirp-Tail")
KEYS = ("frame_pfa_jammer", "pd", "damage", "sjrif_db", "u_nmse", "no_jammer_pooled_nmse")


def evaluate(model, dev):
    metrics = {}
    doppler = dev["doppler_hz"]
    s, j, e = [dev[key] for key in ("s", "j", "epsilon")]
    for lo in range(0, len(s), 8):
        hi = min(lo + 8, len(s))
        ss, jj, ee = [array[lo:hi].reshape(-1, 8, 1800).astype(np.complex128)
                      for array in (s, j, e)]
        y = states_from_components(ss, jj, ee)
        out = restore_model(model, y, next(model.parameters()).device)
        values = group_metrics(ss, jj, ee, out)
        values.update(radar_metrics(ss, y, out, np.repeat(doppler[lo:hi], 3)))
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite metric {name}")
            metrics.setdefault(name, []).append(value.reshape(hi - lo, 3))
    return {name: np.concatenate(value) for name, value in metrics.items()}


def train(arm, seed, config, training, dev, status):
    folder = PRIVATE / arm / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    expected = {"run": RUN, "arm": arm, "seed": seed, "config_sha256": sha(PRIVATE / "CONFIG.json")}
    frozen_objective.ARM = "tail" if arm.endswith("Tail") else "original"
    seed_rng(seed)
    baseline = CorrectionNet()
    seed_rng(seed)
    model = make_model("no_dechirp")
    if any(not torch.equal(value, model.state_dict()[key])
           for key, value in baseline.state_dict().items()):
        raise ValueError("initial tensors differ from CorrectionNet")
    init_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    model = model.cuda()
    optimizer = optimizer_for(model)
    payload = choose_resume(folder, expected)
    start = 0
    if payload:
        start = payload["epoch"]
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])

    def checkpoint(epoch, record):
        return publish(folder, {"format": FORMAT, "identity": expected, "seed": seed,
                               "epoch": epoch, "global_step": 48 * epoch,
                               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": rng_snapshot(), "epoch_record": record})

    if not payload:
        checkpoint(0, None)
    for epoch in range(start + 1, 257):
        user_control(PRIVATE)
        model.train()
        totals = {}
        tick = now()
        for step, indices in enumerate(batches(512, epoch, seed)):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch, step)
            optimizer.zero_grad(set_to_none=True)
            for lo in range(0, len(indices), 8):
                sub = indices[lo:lo + 8]
                batch = make_batch(training, sub, "cuda")
                correction = model(batch["y"]).reshape(-1, 4, 2, 1800)
                terms = frozen_objective.study_objective(correction, batch, len(sub))
                if not torch.isfinite(terms["total"]):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}, step {step}")
                weight = len(sub) / len(indices)
                (weight * terms["total"]).backward()
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.) + float(value.detach()) * weight / 48
                del correction, terms, batch
            if epoch == 1 and step == 0:
                active = sum(parameter.numel() for parameter in model.parameters() if parameter.grad is not None)
                if active != 59714:
                    raise ValueError(f"unexpected active parameter count {active}")
                status["active_real_parameters"] = active
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if step % 12 == 0:
                status.update(epoch=epoch, step=step + 1, updated_utc=now())
                atomic_json(OUT / "STATUS.json", status)
        torch.cuda.synchronize()
        digest = checkpoint(epoch, {"run": RUN, "seed": seed, "epoch": epoch,
                                    "started_utc": tick, "ended_utc": now(),
                                    "global_step": 48 * epoch, **totals})
        status.update(last_committed_epoch=epoch, checkpoint_sha256=digest, updated_utc=now())
        atomic_json(OUT / "STATUS.json", status)

    cp = folder / "epoch_256.pt"
    digest = sha(cp)
    if digest != read(cp.with_suffix(".json"))["checkpoint_sha256"]:
        raise ValueError("invalid epoch 256 checkpoint digest")
    if any(not (folder / f"recovery/epoch_{epoch:03d}.json").exists() for epoch in range(257)):
        raise ValueError("missing immutable epoch receipt")
    metrics = evaluate(model.eval(), dev)
    ids = parent_roster(dev, 256)
    raw = folder / "dev_parent_metrics.npz"
    atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=ids, **metrics))
    jnr = frame_jnr(dev)
    return {"status": "COMPLETE", "epoch": 256, "checkpoint_sha256": digest,
            "initial_state_sha256": init_hash, "parent_metrics_sha256": sha(raw),
            "metrics": stratified(metrics, jnr), "worst_cases": worst_cases(metrics, ids, jnr)}


def report(summary, dev):
    historical = summary["historical"]
    lines = ["# P0 · 去斜与尾部损失", "",
             "所有比较使用同一批 256 个 DEV parent。新实验固定使用第 256 轮。",
             "历史四格均值来自冻结收据。新实验的均值与区间先在每个 parent 内平均三种干扰，再跨 seed 平均。", "",
             "| 方法 | seed | 虚警率 | Pd | 目标损伤 | SJRIF dB | 端点 NMSE | 无干扰误差 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in historical["rows"]:
        for arm in ("unet_original", "unet_tail", "correction_original", "correction_tail"):
            values = " | ".join(f"{row[arm][name]['mean']:.6g}" for name in KEYS)
            lines.append(f"| {arm} | {row['seed']} | {values} |")
    for arm, rows in summary["new_arms"].items():
        raw_rows = []
        for seed, row in rows.items():
            if row["status"] == "COMPLETE":
                values = " | ".join(f"{row['metrics']['global'][name]['mean']:.6g}" for name in KEYS)
                lines.append(f"| {arm} | {seed} | {values} |")
                with np.load(PRIVATE / arm / str(seed) / "dev_parent_metrics.npz", allow_pickle=False) as archive:
                    raw_rows.append({name: archive[name] for name in KEYS})
            else:
                lines.append(f"| {arm} | {seed} | {row['status']} |  |  |  |  |  |")
        if len(raw_rows) == len(SEEDS):
            summary.setdefault("seed_averaged", {})[arm] = stratified(
                {name: np.mean([row[name] for row in raw_rows], axis=0) for name in KEYS},
                frame_jnr(dev))
            values = " | ".join(f"{summary['seed_averaged'][arm]['global'][name]['mean']:.6g}" for name in KEYS)
            lines.append(f"| {arm} | 平均 | {values} |")
    for arm in ("unet_original", "unet_tail", "correction_original", "correction_tail"):
        means = {name: float(np.mean([row[arm][name]["mean"] for row in historical["rows"]]))
                 for name in KEYS}
        summary.setdefault("historical_seed_mean", {})[arm] = means
        values = " | ".join(f"{means[name]:.6g}" for name in KEYS)
        lines.append(f"| {arm} | 平均 | {values} |")
    decision = "PENDING"
    if len(summary.get("seed_averaged", {})) == 2:
        raw = summary["seed_averaged"][ARMS[1]]["global"]
        rotated = summary["historical_seed_mean"]["correction_tail"]
        low_false_alarms = raw["frame_pfa_jammer"]["mean"] < rotated["frame_pfa_jammer"]
        pd_kept = raw["pd"]["mean"] >= rotated["pd"] - .02
        damage_kept = raw["damage"]["mean"] <= rotated["damage"] + .005
        decision = "RAW_IQ" if low_false_alarms and pd_kept and damage_kept else "DECHIRP"
    summary["coordinates"] = decision
    lines += ["", f"坐标筛查结果 **{decision}**。",
              "尾部损失下，原始 I/Q 需降低虚警，Pd 降幅须在 2 个百分点内，且损伤增量不得超过 0.005。",
              "历史四格原收据仅提供全局的逐 seed 区间，均值行沿用该覆盖范围。", ""]
    for arm, rows in summary["new_arms"].items():
        for seed, row in rows.items():
            if row["status"] == "COMPLETE":
                lines += [f"## {arm} · seed {seed} 的最差 parent", ""]
                write_cases(lines, row["worst_cases"])
    atomic_write(OUT / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
    atomic_json(OUT / "SUMMARY.json", summary)


def run(source):
    train_path, train_receipt = validate_role("train", source)
    dev_path, dev_receipt = validate_role("dev", source)
    historical = read(SNAPSHOT / HISTORICAL)
    if historical["dev_data_sha256"] != dev_receipt["sha256"]:
        raise ValueError("historical four-cell data mismatch")
    config = {**provenance(__file__), "data": {"train": train_receipt, "dev": dev_receipt},
              "seeds": list(SEEDS), "epochs": 256, "micro_batch": 8, "updates_per_epoch": 48,
              "precision": configure_precision(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name()}
    path = PRIVATE / "CONFIG.json"
    if path.exists() and read(path) != config:
        raise ValueError("P0 source or data changed")
    atomic_json(path, config)
    status = {**config, "stage": "TRAINING", "started_utc": now(), "ended_utc": None, "pid": os.getpid()}
    atomic_json(OUT / "STATUS.json", status)
    torch.set_num_threads(2)
    training = load_arrays(train_path)
    with np.load(train_path, allow_pickle=False) as archive:
        parent_roster(archive, 512)
    with np.load(dev_path, allow_pickle=False) as archive:
        dev = {name: archive[name] for name in archive.files}
    parent_roster(dev, 256)
    summary_path = OUT / "SUMMARY.json"
    summary = read(summary_path) if summary_path.exists() else {
        "historical": historical, "historical_sha256": sha(SNAPSHOT / HISTORICAL),
        "new_arms": {arm: {str(seed): {"status": "PENDING"} for seed in SEEDS} for arm in ARMS}}
    report(summary, dev)
    for seed in SEEDS:
        for arm in ARMS:
            rows = summary["new_arms"][arm]
            old = rows[str(seed)]
            if old["status"] == "COMPLETE":
                continue
            status.update(arm=arm, seed=seed, epoch=0, step=0, last_committed_epoch=0)
            atomic_json(OUT / "STATUS.json", status)
            try:
                row = train(arm, seed, config, training, dev, status)
                if "failures" in old:
                    row["failures"] = old["failures"]
            except Exception as exc:
                failure = {"error": repr(exc), "utc": now()}
                atomic_json(PRIVATE / arm / str(seed) / f"failure_{len(old.get('failures', []))}.json",
                            {**failure, "traceback": traceback.format_exc()})
                row = {"status": "FAILED", "failures": [*old.get("failures", []), failure]}
                if "out of memory" in str(exc).lower():
                    rows[str(seed)] = row
                    report(summary, dev)
                    raise
            rows[str(seed)] = row
            report(summary, dev)
    complete = all(row["status"] == "COMPLETE" for arm in ARMS for row in summary["new_arms"][arm].values())
    status.update(stage="COMPLETE" if complete else "FAILED", ended_utc=now(), updated_utc=now())
    atomic_json(OUT / "STATUS.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-data-from", type=Path)
    options = parser.parse_args()
    with ProcessLock(PRIVATE / "PROCESS.lock"):
        try:
            run(options.import_data_from)
        except Exception as exc:
            existing = OUT / "STATUS.json"
            prior = read(existing) if existing.exists() else {}
            atomic_json(existing, {**prior, "stage": "FAILED", "ended_utc": now(), "error": repr(exc)})
            raise
