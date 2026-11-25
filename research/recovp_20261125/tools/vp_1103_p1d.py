"""Run zero-training DEV bridge audits without touching the running P0 queue."""
import argparse
import importlib
import os
import traceback
from pathlib import Path

import numpy as np
from scipy.special import expit

from vp_1103_common import (DATA, DEV_HASH, WORK, atomic_json, frame_jnr, now,
                           parent_roster, provenance, sha, validate_role)
from vp_1103_p1 import geometry_grid
from tools.durable_io import ProcessLock, atomic_write

NAMES = {"d1": "p1d1_replay_dict_sensitivity", "d2": "p1d2_replay_first_drift_audit",
         "d3": "p1d3_hankel_noisy_audit"}
SEED_GEOMETRY = 20261124
SMOOTH = {"floor": 1e-4, "span": 100., "slope": 1.25, "threshold": 8.}
FIELDS = ("forward_delay_samples", "window_samples", "period_samples")


def noise_variance(y):
    # Smooth bounded-influence scale from an amplitude-only pilot estimate.
    # Smooth for every finite observation including an all-zero observation.
    power = np.abs(y)**2
    energy = power.mean(axis=-1, keepdims=True) + 1e-24
    weights = 1 / (1 + power / (3 * energy))**2
    return ((weights * power).sum(axis=-1) / weights.sum(axis=-1) / .69).clip(1e-24)


def precision(b, y, rule):
    norms2 = np.sum(np.abs(b)**2, axis=-2)
    matched = np.swapaxes(b.conj(), -1, -2) @ y[..., None]
    snr = np.abs(matched[..., 0])**2 / (norms2 * noise_variance(y)[..., None] + 1e-24)
    low, high = SMOOTH["floor"], SMOOTH["span"]
    if rule == "smooth":
        return low + high * expit(-SMOOTH["slope"] * (snr - SMOOTH["threshold"]))
    if rule == "hard":
        return np.where(snr >= SMOOTH["threshold"], low, low + high)
    return np.full_like(snr, float(rule))


def fit_operator(b, ridge, q=None, confidence=1.):
    """H = B (BᴴQB+Λ)⁻¹ BᴴQ, as B and the small right factor."""
    b = np.asarray(b, dtype=np.complex128)
    size = b.shape[-1]
    if not size:
        return b, np.swapaxes(b.conj(), -1, -2), np.zeros(b.shape[:-2])
    bh = np.swapaxes(b.conj(), -1, -2)
    if q is None:
        bq = bh
    else:
        qh = np.swapaxes(q.conj(), -1, -2)
        bq = bh - (bh @ q) @ qh * np.asarray(confidence)[..., None, None]
    gram = bh @ b
    normal = bq @ b
    normal = (normal + np.swapaxes(normal.conj(), -1, -2)) * .5
    normal[..., np.arange(size), np.arange(size)] += ridge
    right = np.linalg.solve(normal, bq)
    gain = np.einsum("...ij,...ji->...", gram, right @ np.swapaxes(right.conj(), -1, -2)).real
    return b, right, gain


def apply_operator(operator, y):
    b, right, _ = operator
    return (b @ (right @ y[..., None]))[..., 0]


def run(stages):
    path, receipt = validate_role("dev")
    with np.load(path, allow_pickle=False) as archive:
        dev = {name: archive[name] for name in archive.files}
    ids, jnr = parent_roster(dev, 256), frame_jnr(dev)
    for stage in stages:
        name = NAMES[stage]
        private = WORK / "outputs" / name
        out = WORK / "evidence" / "outputs" / name
        with ProcessLock(private / "PROCESS.lock"):
            status = {"stage": "RUNNING", "started_utc": now(), "ended_utc": None, "pid": os.getpid(),
                      "training_seeds": [], "bootstrap_seed": 20261122, "geometry_fill_seed": SEED_GEOMETRY,
                      "micro_batch": None, "data": {"dev": receipt}, "solver_dtype": "complex128",
                      "writeback_dtype": "complex64"}

            def update(**fields):
                status.update(fields, updated_utc=now())
                atomic_json(out / "STATUS.json", status)

            try:
                extra = "vp_1103_p1d3.py" if stage == "d3" else "vp_1103_p1d_replay.py"
                scripts = [__file__, *(Path(__file__).with_name(module) for module in
                           (extra, "vp_1103_p1.py", "vp_1103_geometry.py", "vp_1103_p1b.py",
                            "vp_1103_p1c.py", "vp_1103_references.py"))]
                update(**provenance(*scripts))
                if stage == "d3":
                    summary, lines = importlib.import_module("vp_1103_p1d3").audit(dev, ids, jnr, private, update)
                else:
                    grid = geometry_grid(path, dev)
                    update(geometry_sha256=sha(DATA / "geometry.npz"))
                    module = importlib.import_module("vp_1103_p1d_replay")
                    summary, lines = getattr(module, stage)(dev, ids, jnr, grid, private, update)
                summary.update(dev_sha256=DEV_HASH, parents=256, training_seeds=[],
                               frame_jnr_rule="Mean linear active-support pulse JNR, converted to dB")
                atomic_json(out / "SUMMARY.json", summary)
                atomic_write(out / "REPORT.md", lambda stream: stream.write(("\n".join(lines) + "\n").encode("utf8")))
                update(stage="COMPLETE", ended_utc=now(), summary_sha256=sha(out / "SUMMARY.json"))
            except Exception as exc:
                atomic_json(private / f"failed_{os.getpid()}.json",
                            {"error": repr(exc), "utc": now(), "traceback": traceback.format_exc()})
                update(stage="FAILED", ended_utc=now(), error=repr(exc))
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="+", choices=NAMES)
    run(parser.parse_args().stages)
