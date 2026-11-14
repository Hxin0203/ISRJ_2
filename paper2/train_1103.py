"""Reproducible 32/64/96/128-epoch CorrectionNet training for Paper 2."""
from __future__ import annotations
import hashlib
import os
import platform
import time
import traceback
import numpy as np
import torch

from .data_1103 import generate, verify
from .network import CorrectionNet, training_loss
from .io import ROOT, now, read, sha, write

EPOCHS = (32, 64, 96, 128)
SEEDS = (17, 29, 41)
BATCH = 32
JAM_TYPES = ("direct", "repeat", "cyclic")
# Keep the failed first engineering attempt immutable and run the repaired
# protocol under a distinct identity.
BASE = ROOT / "outputs" / "paper2_1103_retry_20261114"


def source_hashes():
    files = [ROOT / "paper2" / n for n in
             ("train_1103.py", "data_1103.py", "network.py", "science.py", "io.py")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def batches(count, epoch, seed):
    rng = np.random.default_rng(seed + 100_000 + epoch)
    order = rng.permutation(count * len(JAM_TYPES))
    for start in range(0, len(order), BATCH):
        yield order[start:start+BATCH]


def _iq(x):
    return torch.from_numpy(np.stack((x.real, x.imag), axis=-2).astype(np.float32))


def make_batch(arrays, flat_ids, device):
    parent = flat_ids // len(JAM_TYPES)
    jt = flat_ids % len(JAM_TYPES)
    s = arrays["s"][parent, jt].astype(np.complex128)
    j = arrays["j"][parent, jt].astype(np.complex128)
    e = arrays["epsilon"][parent, jt].astype(np.complex128)
    b, pulses, length = s.shape
    ss, jj, ee = (v.reshape(b*pulses, length) for v in (s, j, e))
    y, _ = four_states(ss, jj, ee)
    # One model call per state/pulse: [records * pulses * 4 states, I/Q, fast-time].
    y_iq = _iq(y).reshape(b * pulses * 4, 2, length)
    return dict(y=y_iq.to(device), s=_iq(ss).to(device), j=_iq(jj).to(device), e=_iq(ee).to(device))


def four_states(s, j, e):
    return np.stack((e, s+e, j+e, s+j+e), axis=1), np.stack((0*j, 0*j, j, j), axis=1)


def train_one(seed, arrays, dataset_sha, cfg, src):
    folder = BASE / f"seed_{seed}"
    started = folder / "STARTED.json"
    complete = folder / "TRAINING_COMPLETE.json"
    if complete.exists():
        old = read(complete)
        if old["config_sha256"] != sha(BASE / "CONFIG.json") or old["dataset_sha256"] != dataset_sha:
            raise RuntimeError(f"completed run identity mismatch: seed {seed}")
        for epoch in EPOCHS:
            cp = folder / f"epoch_{epoch:03d}.pt"
            if sha(cp) != old["checkpoints"][str(epoch)]:
                raise RuntimeError(f"checkpoint hash mismatch: {cp}")
        return old
    if started.exists():
        raise RuntimeError(f"incomplete prior training for seed {seed}; preserve and inspect before restart")
    folder.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CorrectionNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
    initial = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
    write(started, dict(seed=seed, started_utc=now(), pid=os.getpid(), device=str(device),
                        dataset_sha256=dataset_sha, config_sha256=sha(BASE / "CONFIG.json"),
                        source_sha256=src, initial_parameters_sha256=initial))
    history, checkpoint_hashes = [], {}
    n = arrays["s"].shape[0]
    steps_per_epoch = int(np.ceil(n * len(JAM_TYPES) / BATCH))
    total_steps = steps_per_epoch * max(EPOCHS)
    global_step = 0
    t0 = time.perf_counter()
    milestone_set = set(EPOCHS)
    for epoch in range(1, max(EPOCHS) + 1):
        epoch_losses = []
        for ids in batches(n, epoch, seed):
            global_step += 1
            # One-epoch linear warmup, followed by cosine decay through epoch 128.
            if epoch == 1:
                lr = 1e-3 * global_step / steps_per_epoch
            else:
                progress = (global_step - steps_per_epoch) / max(1, total_steps - steps_per_epoch)
                lr = 2e-5 + .5 * (1e-3 - 2e-5) * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
            batch = make_batch(arrays, ids, device)
            optimizer.zero_grad(set_to_none=True)
            y = batch["y"]
            corr = net(y).reshape(y.shape[0] // 4, 4, 2, y.shape[-1])
            losses = training_loss(corr, batch["s"], batch["j"], batch["e"], "B4")
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"nonfinite loss at epoch={epoch} step={global_step}")
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            item = {k: float(v.detach()) for k, v in losses.items()}
            item.update(epoch=epoch, step=global_step, lr=float(lr), gradient_norm=float(norm))
            epoch_losses.append(item)
            if global_step == 1 or global_step % 50 == 0:
                write(BASE / "STATUS.json", dict(stage="TRAINING", seed=seed, epoch=epoch,
                            epochs=max(EPOCHS), step=global_step, steps=total_steps,
                            loss=item, elapsed_seconds=time.perf_counter()-t0, updated_utc=now()))
        history.extend(epoch_losses)
        if epoch in milestone_set:
            if device.type == "cuda":
                torch.cuda.synchronize()
            cp = folder / f"epoch_{epoch:03d}.pt"
            tmp = cp.with_suffix(".tmp")
            torch.save(dict(state=net.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                            epoch=epoch, config=cfg, dataset_sha256=dataset_sha,
                            source_sha256=src, parameters=sum(p.numel() for p in net.parameters())), tmp)
            tmp.replace(cp)
            checkpoint_hashes[str(epoch)] = sha(cp)
            write(folder / "history.json", history)
            write(folder / f"epoch_{epoch:03d}.json", dict(seed=seed, epoch=epoch,
                  checkpoint_sha256=checkpoint_hashes[str(epoch)], source_sha256=src,
                  dataset_sha256=dataset_sha, current_epoch_loss=epoch_losses[-1],
                  elapsed_seconds=time.perf_counter()-t0))
            write(BASE / "STATUS.json", dict(stage="CHECKPOINT_SAVED", seed=seed, epoch=epoch,
                  checkpoint_sha256=checkpoint_hashes[str(epoch)], updated_utc=now()))
    rec = dict(seed=seed, epochs=list(EPOCHS), updates=global_step, steps_per_epoch=steps_per_epoch,
               training_parents=n, records_per_epoch=n*len(JAM_TYPES), batch_size=BATCH,
               completed_utc=now(), elapsed_seconds=time.perf_counter()-t0, device=str(device),
               gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
               torch=torch.__version__, python=platform.python_version(), parameters=sum(p.numel() for p in net.parameters()),
               initial_parameters_sha256=initial, dataset_sha256=dataset_sha, source_sha256=src,
               config_sha256=sha(BASE / "CONFIG.json"), checkpoints=checkpoint_hashes)
    write(complete, rec)
    started.unlink()
    return rec


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    lock = BASE / "PROCESS.lock"
    try:
        with lock.open("x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("existing 1103 training lock; inspect before starting another run")
    try:
        path, receipt = generate("train")
        identity = verify(path)
        cfg = dict(epochs=list(EPOCHS), seeds=list(SEEDS), batch_size=BATCH,
                   epoch_definition="one pass over each of 512 parents x 3 jammer variants",
                   steps_per_epoch=int(np.ceil(512*3/BATCH)), optimizer="Adam",
                   betas=[.9,.999], eps=1e-8, weight_decay=0,
                   loss="current B4 objective: endpoint + 0.1 pulse-compression + target-response difference",
                   lr_schedule="one-epoch warm-up to 1e-3, cosine decay to 2e-5 by epoch 128",
                   data_receipt_sha256=sha(receipt), data_sha256=sha(path), data_identity=identity)
        config_path = BASE / "CONFIG.json"
        if config_path.exists():
            old = read(config_path)
            if old != cfg:
                raise RuntimeError("frozen 1103 training config changed")
        else:
            write(config_path, cfg)
        src = source_hashes()
        cfg_sha = sha(config_path)
        arrays = load_arrays(path)
        all_runs = [train_one(seed, arrays, sha(path), cfg, src) for seed in SEEDS]
        write(BASE / "TRAINING_COMPLETE.json", dict(completed_utc=now(), config_sha256=cfg_sha,
              source_sha256=src, data_sha256=sha(path), seeds=list(SEEDS),
              checkpoint_epochs=list(EPOCHS), runs=all_runs))
        write(BASE / "STATUS.json", dict(stage="TRAINING_COMPLETE", updated_utc=now()))
    except BaseException as exc:
        write(BASE / "FAILURE.json", dict(time=now(), error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink(missing_ok=True)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("s", "j", "epsilon")}


if __name__ == "__main__":
    main()
