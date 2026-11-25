"""Recover geometry for the fixed TRAIN16 roster and enumerate replay shifts."""
import numpy as np

from vp_1103_common import FAMILIES, atomic_json, sha
from vp_1103_geometry import INTEGER_FIELDS, FREQUENCY_FIELD, lfm_atom
from paper2 import data_1103
from paper2.vendor.generate_v2 import _capture_segments, _draw_target, _draw_noise, _draw_jammer


def recover_selected(small, private, training_receipt):
    path = private / "train16_geometry.npz"
    fields = (*INTEGER_FIELDS, FREQUENCY_FIELD)
    if path.exists():
        from vp_1103_common import read
        receipt = read(path.with_suffix(".json"))
        if receipt["archive_sha256"] != sha(path) or receipt["train_sha256"] != training_receipt["sha256"]:
            raise ValueError("Geometry cache binding mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["parent_id"], small["parent_id"]):
                raise ValueError("Unexpected cached TRAIN16 roster")
            return archive["geometry"]
    seed = data_1103.TRAIN_SEED
    positions = small["parent_id"] - seed * 1_000_000
    selected = {int(parent): index for index, parent in enumerate(positions)}
    result = np.full((len(positions), 3, 8, len(fields)), np.nan)
    rng = np.random.default_rng(seed)
    pri = 1800 / 60e6
    for parent in range(int(max(positions)) + 1):
        limits = ((-20., 15.), (-12., 35.)) if parent < 256 else (
            (-30., np.nextafter(-20., -np.inf)), (-12., 45.))
        cfg = data_1103._config(*limits, seed + parent)
        target, active, meta = _draw_target(cfg, rng)
        fd = float(meta["doppler_hz"])
        n_jammed = int(rng.choice([1, 2, 4, 6, 8]))
        jammed = np.zeros(8, bool)
        jammed[rng.choice(8, size=n_jammed, replace=False)] = True
        targets = [target * np.exp(2j * np.pi * fd * pri * pulse) for pulse in range(8)]
        noises = [_draw_noise(float(meta["signal_power_active"]), cfg, rng)[0] for _ in range(8)]
        for family, name in enumerate(FAMILIES):
            for pulse in range(8):
                j = np.zeros(1800, complex)
                geometry = None
                if jammed[pulse]:
                    j, geometry = _draw_jammer(targets[pulse], active, float(meta["signal_power_active"]),
                                               name, cfg, rng)
                if parent in selected:
                    index = selected[parent]
                    for key, actual in (("s", targets[pulse]), ("j", j), ("epsilon", noises[pulse])):
                        if not np.array_equal(actual.astype(np.complex64), small[key][index, family, pulse]):
                            raise ValueError(f"Selected parent replay mismatch: {parent}/{name}/{pulse}/{key}")
                    if geometry is not None:
                        result[index, family, pulse] = [geometry[key] for key in fields]
    # Earlier parent draws only advance the frozen RNG stream. No arrays,
    # scores or training examples outside the selected 16 parents are retained.
    from tools.durable_io import atomic_write
    atomic_write(path, lambda stream: np.savez_compressed(stream, parent_id=small["parent_id"], geometry=result))
    atomic_json(path.with_suffix(".json"), {
        "archive_sha256": sha(path), "train_sha256": training_receipt["sha256"],
        "selected_parents": len(positions), "selected_arrays_exact": True,
        "fields": fields, "rng_advancement_only_for_unselected_parents": True})
    return result


def replay_components(delay, doppler, family, metadata):
    """Return grouped delay, real support and unit-amplitude physical waveforms."""
    if not np.isfinite(metadata).all():
        return []
    parameters = {key: int(value) for key, value in zip(INTEGER_FIELDS, metadata[:-1])}
    freq = float(metadata[-1])
    atom = lfm_atom(delay, doppler)
    segments = _capture_segments(atom, atom != 0, parameters["window_samples"],
                                  parameters["period_samples"], parameters["sampling_phase_samples"])
    window = parameters["window_samples"]
    forward = parameters["forward_delay_samples"]
    spacing = window + parameters["replay_gap_samples"]
    repeats = parameters["repeat_count"]
    groups = {}
    for index, (source_start, segment) in enumerate(segments):
        if family == 0:
            sources = [(source_start, segment)]
        elif family == 1:
            sources = [(source_start, segment)] * repeats
        else:
            sources = list(reversed(segments[:index + 1]))[:repeats]
        for replay, (origin, payload) in enumerate(sources):
            start = source_start + window + forward + replay * spacing
            amount = min(window, 1800 - start)
            if amount <= 0:
                continue
            shift = start - origin
            mask, wave = groups.setdefault(shift, (np.zeros(1800), np.zeros(1800, complex)))
            mask[start:start + amount] += 1.
            wave[start:start + amount] += payload[:amount]
    result = []
    modulation = np.exp(2j * np.pi * freq * np.arange(1800) / 60e6)
    for shift, (mask, wave) in groups.items():
        wave *= modulation
        result.append({"physical_delay": delay + shift,
                       "phase_delay": delay + shift - (doppler + freq) / 50_000.,
                       "mask": mask, "wave": wave, "energy": float(np.sum(abs(wave)**2)),
                       "frequency_shift_hz": doppler + freq})
    return sorted(result, key=lambda row: (-row["energy"], row["phase_delay"]))
