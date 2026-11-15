"""Shared v4 waveform and radar evaluation, with model identity binding."""
from __future__ import annotations

import time
import numpy as np
import torch

from paper2.evaluation_1103_v4 import (states_from_components, restore_model,
                                     group_metrics, radar_metrics, summarize)
from paper2.io import ROOT, now, read, sha
from tools.durable_io import atomic_write, atomic_json
from .models import make_model
from .protocol import BASE, DATA, EVIDENCE, ORIGINAL, CONFIG
from .runtime import require_server, user_control, configure_precision


def evaluate(variant, seed, epoch, *, data_path=None, scope="DEV"):
    require_server()
    configure_precision()
    user_control(BASE)
    cfg = read(CONFIG)
    if scope not in ("DEV", "CONFIRMATION"):
        raise ValueError(scope)
    if scope == "DEV":
        data_path = DATA / "dev.npz"
    elif data_path is None or not (BASE / "ALL_TRAINING_COMPLETE.json").exists():
        raise RuntimeError("confirmation may only run after all training completes")
    folder = ((ORIGINAL / f"seed_{seed}") if variant == "full" else
              BASE / variant / f"seed_{seed}")
    cp = folder / f"epoch_{epoch:03d}.pt"
    digest = sha(cp)
    if read(cp.with_suffix(".json"))["checkpoint_sha256"] != digest:
        raise RuntimeError(f"checkpoint damaged: {cp}")
    if variant == "full" and digest != cfg["original_epoch256"][str(seed)]["checkpoint_sha256"]:
        raise RuntimeError("original checkpoint differs from preregistered baseline")
    out_dir = BASE / variant / f"seed_{seed}"
    out = out_dir / f"{scope.lower()}_epoch_{epoch:03d}.json"
    raw = out.with_suffix(".npz")
    binding = {"variant": variant, "seed": seed, "epoch": epoch, "scope": scope,
               "config_sha256": sha(CONFIG), "checkpoint_sha256": digest,
               "data_sha256": sha(data_path), "source_sha256": cfg["source_sha256"]}
    if out.exists():
        saved = read(out)
        if any(saved.get(k) != v for k, v in binding.items()) or sha(raw) != saved["local_parent_values_sha256"]:
            raise RuntimeError(f"existing evaluation identity mismatch: {out}")
        return saved
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Evaluation construction must not advance training initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = make_model(variant).to(device).eval()
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    if payload["seed"] != seed or payload["epoch"] != epoch:
        raise RuntimeError("checkpoint seed/epoch mismatch")
    if variant != "full" and (payload["identity"]["variant"] != variant or payload["identity"]["config_sha256"] != sha(CONFIG)):
        raise RuntimeError("checkpoint model/config mismatch")
    model.load_state_dict(payload.get("model", payload.get("state")), strict=True)
    with np.load(data_path, allow_pickle=False) as archive:
        components = [archive[k] for k in ("s", "j", "epsilon")]
        ids = archive["parent_id"].copy()
        doppler = archive["doppler_hz"].copy()
    started = time.perf_counter()
    values = {}
    for lo in range(0, len(ids), 8):
        hi = min(lo + 8, len(ids))
        s, j, e = [v[lo:hi].reshape(-1, 8, 1800).astype(np.complex128) for v in components]
        observed = states_from_components(s, j, e)
        restored = restore_model(model, observed, device)
        metrics = group_metrics(s, j, e, restored)
        metrics.update(radar_metrics(s, observed, restored, np.repeat(doppler[lo:hi], 3)))
        for name, value in metrics.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"nonfinite evaluation: {name}")
            values.setdefault(name, []).append(value.reshape(hi - lo, 3).mean(1))
        atomic_json(BASE / "EVAL_STATUS.json", {**binding, "parents_done": hi, "parents": len(ids), "updated_utc": now()})
    parent_values = {k: np.concatenate(v) for k, v in values.items()}
    atomic_write(raw, lambda stream: np.savez_compressed(stream, parent_id=ids, **parent_values))
    result = {**binding, "parents": len(ids), "metrics": {k: summarize(v) for k, v in parent_values.items()},
              "local_parent_values_sha256": sha(raw), "elapsed_seconds": time.perf_counter() - started,
              "created_utc": now(), "note": "Three jammer variants averaged within physical parent; no pulse-level pseudoreplication."}
    atomic_json(out, result)
    atomic_json(EVIDENCE / variant / f"seed_{seed}" / out.name, result)
    return result
