"""Dedicated eight-pulse simulator/data roles for the 1103-aligned study.

This module is separate from the frozen October pilot generator and never reads
legacy held-out assets. Complex components are stored locally as float32-backed
complex64 arrays; physical synthesis and integrity checks use complex128.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np

from .vendor.generate_v2 import GeneratorConfig, _draw_target, _draw_jammer, _draw_noise
from .io import ROOT, now, read, sha, write

JAM_TYPES = ("direct", "repeat", "cyclic")
PULSES = 8
LENGTH = 1800
TRAIN_SEED = 2026111301
DEV_SEED = 2026111302
TEST_SEED = 2026111303
ROLE_COUNTS = {"train": 512, "dev": 256, "test": 256, "engineering": 4}
ROLE_SEEDS = {"train": TRAIN_SEED, "dev": DEV_SEED, "test": TEST_SEED, "engineering": 2026111300}


def _config(snr, jsr, seed):
    return GeneratorConfig(
        num_scenes=1, signal_length=LENGTH, sample_rate_hz=60e6,
        pulse_width_s=10e-6, bandwidth_hz=30e6, min_guard_samples=64,
        target_amplitude_min=.15, target_amplitude_max=2.0,
        target_doppler_min_hz=-200e3, target_doppler_max_hz=200e3,
        snr_min_db=snr[0], snr_max_db=snr[1], jsr_min_db=jsr[0], jsr_max_db=jsr[1],
        jam_window_min_samples=6, jam_window_max_samples=72,
        jam_period_min_samples=36, jam_period_max_samples=260,
        jam_forward_delay_min_samples=1, jam_forward_delay_max_samples=96,
        jam_repeats_min=1, jam_repeats_max=10, jam_replay_gap_max_samples=64,
        jammer_freq_offset_min_hz=-200e3, jammer_freq_offset_max_hz=200e3,
        seed=seed, fixed_partition="train")


def generate(role: str, *, count: int | None = None, seed: int | None = None,
             snr_db: float | None = None, jsr_db: float | None = None,
             out_dir: Path | None = None) -> tuple[Path, Path]:
    """Generate parent-paired coherent pulse groups, reusing base noise/target.

    TRAIN follows the two 256-parent Main/Extreme strata from 1103. Supplying
    both `snr_db` and `jsr_db` creates a fixed-condition evaluation role.
    Arrays are parent × jammer-type × pulse × fast-time.
    """
    if role not in ROLE_COUNTS:
        raise ValueError(f"unsupported role {role!r}")
    count = ROLE_COUNTS[role] if count is None else int(count)
    seed = ROLE_SEEDS[role] if seed is None else int(seed)
    if count < 1:
        raise ValueError("count must be positive")
    if (snr_db is None) != (jsr_db is None):
        raise ValueError("fixed evaluation conditions require both SNR and JSR")
    out_dir = out_dir or ROOT / "data" / "paper2_1103"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{snr_db:g}_{jsr_db:g}" if snr_db is not None else ""
    path = out_dir / f"{role}{suffix}.npz"
    receipt = out_dir / f"{role}{suffix}.receipt.json"
    if path.exists() or receipt.exists():
        if not (path.exists() and receipt.exists()):
            raise RuntimeError(f"partial existing data role: {path}")
        saved = read(receipt)
        if saved["sha256"] != sha(path):
            raise RuntimeError(f"data identity mismatch: {path}")
        return path, receipt

    rng = np.random.default_rng(seed)
    s_all = np.zeros((count, len(JAM_TYPES), PULSES, LENGTH), np.complex64)
    e_all = np.zeros_like(s_all)
    j_all = np.zeros_like(s_all)
    ids = np.arange(seed * 1_000_000, seed * 1_000_000 + count, dtype=np.int64)
    fd_all = np.zeros(count, np.float64)
    delay_all = np.zeros(count, np.float64)
    snr_all = np.zeros((count, PULSES), np.float32)
    jsr_all = np.full((count, len(JAM_TYPES), PULSES), np.nan, np.float32)
    pulse_count_all = np.zeros(count, np.int8)
    cfg = _config((-30., 15.), (-12., 45.), seed)
    pri = LENGTH / 60e6

    for parent in range(count):
        if snr_db is not None:
            snr_lo = snr_hi = float(snr_db)
            jsr_lo = jsr_hi = float(jsr_db)
        elif role == "train" and parent < count // 2:
            snr_lo, snr_hi, jsr_lo, jsr_hi = -20., 15., -12., 35.
        elif role == "train":
            snr_lo, snr_hi, jsr_lo, jsr_hi = -30., np.nextafter(-20., -np.inf), -12., 45.
        else:
            snr_lo, snr_hi, jsr_lo, jsr_hi = -30., 15., -12., 45.
        local_cfg = _config((snr_lo, snr_hi), (jsr_lo, jsr_hi), seed + parent)
        target, active, meta = _draw_target(local_cfg, rng)
        fd = float(meta["doppler_hz"])
        fd_all[parent] = fd
        delay_all[parent] = float(meta["delay_samples"])
        n_jammed = int(rng.choice([1, 2, 4, 6, 8]))
        pulse_count_all[parent] = n_jammed
        jammed = np.zeros(PULSES, dtype=bool)
        jammed[rng.choice(PULSES, size=n_jammed, replace=False)] = True
        pulse_noise = []
        pulse_targets = []
        for pulse in range(PULSES):
            phase = np.exp(2j * np.pi * fd * pri * pulse)
            sp = target * phase
            pulse_targets.append(sp)
            # SNR is drawn once per pulse; exactly the same noise is shared by all jammer variants.
            if snr_db is not None:
                local_cfg = _config((snr_lo, snr_hi), (jsr_lo, jsr_hi), seed + parent)
            noise, snr, _ = _draw_noise(float(meta["signal_power_active"]), local_cfg, rng)
            pulse_noise.append(noise)
            snr_all[parent, pulse] = snr
        for jt, jam_type in enumerate(JAM_TYPES):
            for pulse, (sp, noise) in enumerate(zip(pulse_targets, pulse_noise)):
                s_all[parent, jt, pulse] = sp
                e_all[parent, jt, pulse] = noise
                if jammed[pulse]:
                    jammer, jmeta = _draw_jammer(sp, active, float(meta["signal_power_active"]),
                                                  jam_type, local_cfg, rng)
                    j_all[parent, jt, pulse] = jammer
                    jsr_all[parent, jt, pulse] = float(jmeta["actual_jsr_db"])

    np.savez_compressed(path, s=s_all, j=j_all, epsilon=e_all, parent_id=ids,
                        doppler_hz=fd_all, target_delay_samples=delay_all,
                        snr_db=snr_all, jsr_db=jsr_all, jammed_pulse_count=pulse_count_all,
                        jam_types=np.asarray(JAM_TYPES))
    role_config = dict(role=role, count=count, seed=seed, snr_db=snr_db, jsr_db=jsr_db,
                       config=asdict(cfg), pulses=PULSES, pulse_repetition_interval_s=pri,
                       target_amplitude_range=[.15, 2.0], coherent_slow_time_doppler=True,
                       shared_target_noise_across_jammer_types=True,
                       jammer_types=JAM_TYPES, parent_grouping="parent_id")
    write(receipt, dict(created_utc=now(), sha256=sha(path), bytes=path.stat().st_size,
                        **role_config))
    return path, receipt


def verify(path: Path):
    with np.load(path, allow_pickle=False) as z:
        s, j, e = [z[k].astype(np.complex128) for k in ("s", "j", "epsilon")]
        y = np.stack((e, s + e, j + e, s + j + e), axis=3)
        ids = z["parent_id"]
        assert s.shape == j.shape == e.shape == (len(ids), 3, 8, 1800)
        assert np.isfinite(s).all() and np.isfinite(j).all() and np.isfinite(e).all()
        assert np.array_equal(y[:, :, :, 3] - y[:, :, :, 2], s)
        for k in range(3):
            assert np.array_equal(s[:, 0], s[:, k]) and np.array_equal(e[:, 0], e[:, k])
        return dict(parents=len(ids), shape=list(s.shape), four_state_identity=True,
                    shared_target_noise=True, max_abs_component=float(max(np.abs(s).max(), np.abs(j).max())))
