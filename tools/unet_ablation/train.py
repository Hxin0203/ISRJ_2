"""One durable variant/seed training process; original frozen files are read-only."""
from __future__ import annotations

import argparse
import os
import time
import traceback
import numpy as np
import torch

from paper2.io import now, read, sha
from paper2.train_1103 import batches, load_arrays, make_batch
from paper2.train_256_8seed import (atomic_torch_save, seed_rng, rng_snapshot,
                                  restore_rng, optimizer_for, learning_rate, append_event)
from tools.durable_io import ProcessLock, atomic_json, atomic_copy, replace_retry
from .models import VARIANTS, make_model, objective
from .protocol import BASE, DATA, CONFIG, EVIDENCE, MILESTONES, folder_for, validate_config
from .evaluate import evaluate
from .runtime import require_server, user_control, configure_precision

FORMAT = "paper2-unet-ablation-epoch-v1"


def valid_checkpoint(path, expected):
    try:
        receipt = read(path.with_suffix(".json"))
        if receipt["checkpoint_sha256"] != sha(path):
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["format"] != FORMAT or payload["identity"] != expected or payload["epoch"] != receipt["epoch"]:
            return None
        if payload["global_step"] != payload["epoch"] * 48:
            return None
        return payload
    except Exception:
        return None


def choose_resume(folder, expected):
    candidates = []
    for path in [*folder.glob("recovery/epoch_*.pt"), folder / "latest.pt", *folder.glob("epoch_*.pt")]:
        try:
            candidates.append((int(read(path.with_suffix(".json"))["epoch"]), str(path), path))
        except (OSError, ValueError, KeyError):
            pass
    for _, _, path in sorted(candidates, reverse=True):
        value = valid_checkpoint(path, expected)
        if value is not None:
            return value
    if any((folder / "recovery").glob("*.pt")) or (folder / "latest.pt").exists():
        raise RuntimeError("no valid recovery checkpoint; preserve this run for inspection")
    return None


def preserve_invalid(path):
    if path.exists():
        replace_retry(path, path.with_name(path.name + f".invalid_{time.time_ns()}"))


def publish_checkpoint(folder, payload):
    epoch = payload["epoch"]
    archive = folder / "recovery" / f"epoch_{epoch:03d}.pt"
    existing = valid_checkpoint(archive, payload["identity"])
    if existing is not None:
        if existing["epoch_record"] != payload["epoch_record"] or any(
                not torch.equal(v.cpu(), existing["model"][k].cpu()) for k, v in payload["model"].items()):
            raise RuntimeError("immutable recovery checkpoint changed")
        digest = sha(archive)
    else:
        preserve_invalid(archive)
        preserve_invalid(archive.with_suffix(".json"))
        digest = atomic_torch_save(payload, archive)
        atomic_json(archive.with_suffix(".json"), {"epoch": epoch, "identity": payload["identity"], "checkpoint_sha256": digest})
    targets = [folder / "latest.pt"]
    if epoch in MILESTONES:
        targets.append(folder / f"epoch_{epoch:03d}.pt")
    for path in targets:
        if path.name != "latest.pt" and path.exists() and sha(path) != digest:
            raise RuntimeError(f"immutable milestone changed: {path}")
        atomic_copy(archive, path)
        atomic_json(path.with_suffix(".json"), {"epoch": epoch, "identity": payload["identity"], "checkpoint_sha256": digest})
    record = payload["epoch_record"]
    if record is not None:
        path = folder / f"training_epoch_{epoch:03d}.json"
        value = {**record, "checkpoint_sha256": digest}
        if path.exists() and read(path) != value:
            raise RuntimeError("immutable epoch metrics changed")
        atomic_json(path, value)
        done = set()
        if (folder / "events.jsonl").exists():
            import json
            for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["epoch"])
                except (ValueError, KeyError):
                    # A torn last append is retained; authoritative epoch JSON is complete.
                    pass
        if epoch not in done:
            # Prefix a newline to separate any torn trailing JSON from the next event.
            events = folder / "events.jsonl"
            if events.exists() and not events.read_bytes().endswith(b"\n"):
                with events.open("ab") as stream:
                    stream.write(b"\n"); stream.flush(); os.fsync(stream.fileno())
            append_event(events, value)
    return digest


def minibatch_update(net, optimizer, arrays, ids, variant, device, microbatch_records=8):
    optimizer.zero_grad(set_to_none=True)
    sums = {k: 0.0 for k in ("total", "endpoint", "pc", "damage")}
    for lo in range(0, len(ids), microbatch_records):
        subids = ids[lo:lo + microbatch_records]
        b = make_batch(arrays, subids, device)
        y = b["y"]
        c = net(y).reshape(-1, 4, 2, y.shape[-1])
        losses = objective(c, b["s"], b["j"], b["e"], variant)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("nonfinite training loss")
        weight = len(subids) / len(ids)
        (losses["total"] * weight).backward()
        for key, value in losses.items():
            sums[key] += float(value.detach()) * weight
        del b, y, c, losses
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
    optimizer.step()
    return {**sums, "gradient_norm": float(norm)}


def epoch_updates(net, optimizer, arrays, epoch, seed, variant, device, progress=None):
    rows = []
    net.train()
    for i, ids in enumerate(batches(len(arrays["s"]), epoch, seed)):
        lr = learning_rate(epoch, i)
        for group in optimizer.param_groups:
            group["lr"] = lr
        row = minibatch_update(net, optimizer, arrays, ids, variant, device,
                               microbatch_records=8 if variant == "unet" else 32)
        row.update(lr=lr)
        rows.append(row)
        if progress and (i == 0 or (i + 1) % 12 == 0):
            progress(i + 1, row)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, len(rows)


def train(variant, seed):
    require_server()
    configure_precision()
    cfg = validate_config()
    folder = folder_for(variant, seed)
    folder.mkdir(parents=True, exist_ok=True)
    expected = {"variant": variant, "seed": seed, "config_sha256": sha(CONFIG),
                "data_sha256": cfg["data_sha256"]["train.npz"], "source_sha256": cfg["source_sha256"]}
    with ProcessLock(folder / "PROCESS.lock"):
        if (folder / "TRAINING_COMPLETE.json").exists():
            saved = read(folder / "TRAINING_COMPLETE.json")
            if saved["identity"] != expected:
                raise RuntimeError("completed run identity mismatch")
            for epoch, digest in saved["milestones"].items():
                cp = folder / f"epoch_{int(epoch):03d}.pt"
                if sha(cp) != digest or valid_checkpoint(cp, expected) is None:
                    raise RuntimeError("completed milestone damaged")
                evaluate(variant, seed, int(epoch))
            return saved
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for this scheduled study")
        device = torch.device("cuda")
        seed_rng(seed)
        net = make_model(variant).to(device)
        optimizer = optimizer_for(net)
        recovered = choose_resume(folder, expected)
        if recovered:
            net.load_state_dict(recovered["model"], strict=True)
            optimizer.load_state_dict(recovered["optimizer"])
            restore_rng(recovered["rng"])
            start_epoch = recovered["epoch"]
            publish_checkpoint(folder, recovered)  # repair any interrupted publication
        else:
            start_epoch = 0
            recovered = {"format": FORMAT, "identity": expected, "seed": seed, "epoch": 0, "global_step": 0,
                         "model": net.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_snapshot(),
                         "epoch_record": None, "architecture": read(BASE / variant / "ARCHITECTURE.json")}
            publish_checkpoint(folder, recovered)
        for epoch in MILESTONES:
            if epoch <= start_epoch:
                evaluate(variant, seed, epoch)
        arrays = load_arrays(DATA / "train.npz")
        if len(arrays["s"]) != 512:
            raise RuntimeError("unexpected training parent count")
        for epoch in range(start_epoch + 1, 257):
            user_control(BASE)
            started = time.perf_counter()
            def progress(batch, loss):
                atomic_json(BASE / "STATUS.json", {"stage": "TRAINING", "variant": variant, "seed": seed,
                            "epoch": epoch, "batch": batch, "batches": 48, "last_committed_epoch": epoch - 1,
                            "loss": loss, "pid": os.getpid(), "updated_utc": now()})
            metrics, updates = epoch_updates(net, optimizer, arrays, epoch, seed, variant, device, progress)
            if updates != 48:
                raise RuntimeError("training update budget mismatch")
            torch.cuda.synchronize()
            record = {"variant": variant, "seed": seed, "epoch": epoch, "step": epoch * 48,
                      "updates_this_epoch": updates, **metrics, "epoch_elapsed_seconds": time.perf_counter() - started}
            payload = {"format": FORMAT, "identity": expected, "seed": seed, "epoch": epoch,
                       "global_step": epoch * 48, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng_snapshot(), "epoch_record": record, "architecture": recovered["architecture"]}
            digest = publish_checkpoint(folder, payload)
            atomic_json(BASE / "STATUS.json", {"stage": "EPOCH_SAVED", "variant": variant, "seed": seed,
                        "epoch": epoch, "checkpoint_sha256": digest, "metrics": metrics, "updated_utc": now()})
            print(f"{variant} seed={seed} epoch={epoch}/256 loss={metrics['total']:.7g} seconds={record['epoch_elapsed_seconds']:.2f}", flush=True)
            if epoch in MILESTONES:
                evaluate(variant, seed, epoch)
                from .summarize import refresh
                refresh()
        milestones = {}
        recovery = {}
        for epoch in range(0, 257):
            cp = folder / "recovery" / f"epoch_{epoch:03d}.pt"
            if valid_checkpoint(cp, expected) is None:
                raise RuntimeError(f"recovery audit failed: {cp}")
            recovery[str(epoch)] = sha(cp)
            if epoch:
                record = read(folder / f"training_epoch_{epoch:03d}.json")
                if record["checkpoint_sha256"] != recovery[str(epoch)]:
                    raise RuntimeError("epoch metric/checkpoint binding failed")
            if epoch in MILESTONES:
                cp = folder / f"epoch_{epoch:03d}.pt"
                if sha(cp) != recovery[str(epoch)]:
                    raise RuntimeError("milestone/recovery binding failed")
                milestones[str(epoch)] = sha(cp)
        result = {"identity": expected, "variant": variant, "seed": seed, "epochs": 256,
                  "updates": 12288, "milestones": milestones, "recovery_sha256": recovery, "completed_utc": now()}
        atomic_json(folder / "TRAINING_COMPLETE.json", result)
        atomic_json(EVIDENCE / variant / f"seed_{seed}" / "TRAINING_COMPLETE.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 41), required=True)
    args = parser.parse_args()
    folder = folder_for(args.variant, args.seed)
    try:
        train(args.variant, args.seed)
    except BaseException as exc:
        atomic_json(folder / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                    "retryable": isinstance(exc, OSError), "utc": now(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
