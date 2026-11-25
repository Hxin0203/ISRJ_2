"""Recover DEV geometry from the historical generator and validate its roster."""
import tempfile

import numpy as np

from vp_1103_common import DATA, FAMILIES, atomic_json, now, sha
from paper2 import data_1103
from paper2.vendor.generate_v2 import _raw_jammer


INTEGER_FIELDS = ("window_samples", "period_samples", "sampling_phase_samples",
                  "forward_delay_samples", "repeat_count", "replay_gap_samples")
FREQUENCY_FIELD = "frequency_offset_hz"
SAMPLES = np.arange(1800)
FS = 60e6
WIDTH = 600


def recover_geometry(path, dev):
    """Replay DEV only and verify equality before exposing geometry to the fit."""
    geometry = []
    draw = data_1103._draw_jammer

    def keep_geometry(*args, **kwargs):
        waveform, metadata = draw(*args, **kwargs)
        geometry.append({name: metadata[name] for name in (*INTEGER_FIELDS, FREQUENCY_FIELD)})
        return waveform, metadata

    try:
        data_1103._draw_jammer = keep_geometry
        with tempfile.TemporaryDirectory(prefix="geometry_qa_", dir=DATA) as scratch:
            from pathlib import Path
            regenerated, _ = data_1103.generate("dev", out_dir=Path(scratch))
            with np.load(regenerated, allow_pickle=False) as recreated:
                if set(recreated.files) != set(dev):
                    raise ValueError("DEV replay schema differs")
                for name in recreated.files:
                    values = recreated[name]
                    equal = np.array_equal(values, dev[name], equal_nan=values.dtype.kind in "fc")
                    if not equal:
                        raise ValueError(f"DEV replay mismatch for {name}")
    finally:
        data_1103._draw_jammer = draw
    selected = np.isfinite(dev["jsr_db"])
    grid = np.empty(selected.shape, dtype=object)
    grid[~selected] = None
    if len(geometry) != int(selected.sum()):
        raise ValueError("replayed metadata count disagrees with DEV")
    grid[selected] = geometry
    atomic_json(DATA / "geometry_binding.json", {
        "created_utc": now(), "archive_sha256": sha(path), "per_array_exact_match": True,
        "fields": [*INTEGER_FIELDS, FREQUENCY_FIELD], "jammed_pulses": len(geometry)})
    return grid


def lfm_atom(delay, doppler, smooth=False, derivative=False):
    local = SAMPLES - delay
    centered = local - WIDTH / 2
    phase = np.exp(1j * (np.pi / (2 * WIDTH) * centered**2 + 2 * np.pi * doppler / FS * centered))
    if not smooth:
        return phase * ((local >= 0) & (local < WIDTH))
    # A differentiable edge template with a symmetric four-sample transition.
    # The frozen simulator still uses its original hard support.
    rising = np.clip((local + 2) / 4, 0, 1)
    falling = np.clip((WIDTH + 2 - local) / 4, 0, 1)
    window = np.sin(np.pi / 2 * rising)**2 * np.sin(np.pi / 2 * falling)**2
    if not derivative:
        return phase * window
    left = np.pi / 8 * np.sin(np.pi * rising) * (rising > 0) * (rising < 1)
    right = -np.pi / 8 * np.sin(np.pi * falling) * (falling > 0) * (falling < 1)
    window_grad = left * np.sin(np.pi / 2 * falling)**2 + right * np.sin(np.pi / 2 * rising)**2
    delay_grad = (-window_grad - 1j * window * (np.pi / WIDTH * centered + 2 * np.pi * doppler / FS)) * phase
    doppler_grad = 2j * np.pi / FS * centered * phase * window
    return phase * window, delay_grad, doppler_grad


def replay_atom(delay, doppler, family, geometry):
    atom = lfm_atom(delay, doppler)
    raw = _raw_jammer(atom, atom != 0, FAMILIES[family], geometry)
    return raw * np.exp(2j * np.pi / FS * geometry[FREQUENCY_FIELD] * SAMPLES)
