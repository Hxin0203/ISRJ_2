"""Generate a leakage-safe V2 dataset for joint ISRJ recovery.

Each physical scene contains one randomized LFM target and one receiver-noise
realization.  Several jammer variants can be generated from that same scene.
All variants inherit the same ``scene_id`` and are assigned to one split as a
group, which prevents a target/noise realization from leaking across
train/validation/test.  Development-asset builders may instead request one
fixed partition and a non-zero scene-ID namespace; the legacy three-way mode
remains the default.

The explicitly stored physical components are::

    u = s + epsilon
    y = u + j

``u`` is the primary recovery target.  The generator never creates or stores
a target-delay-derived noise gate for model input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:  # Support both ``python -m ...generate_v2`` and direct execution.
    from .label_builder import build_reliability_labels
except ImportError:  # pragma: no cover - direct-script convenience path
    from label_builder import build_reliability_labels


SCHEMA_VERSION = "2.1.0"
JAM_TYPES = ("repeat", "direct", "cyclic")
SPLIT_TO_CODE = {
    "train": 0,
    "val": 1,
    "test": 2,
    "validation": 3,
    "calibration": 4,
}
DEVELOPMENT_PARTITIONS = ("train", "validation", "calibration")


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for :func:`generate_dataset`.

    Defaults intentionally create a small dataset suitable for a smoke test.
    A scene produces ``len(jam_types) * variants_per_type`` samples.
    """

    num_scenes: int = 24
    variants_per_type: int = 1
    signal_length: int = 1800
    sample_rate_hz: float = 60e6
    pulse_width_s: float = 10e-6
    bandwidth_hz: float = 30e6
    min_guard_samples: int = 64

    target_amplitude_min: float = 0.60
    target_amplitude_max: float = 1.40
    target_phase_min_rad: float = 0.0
    target_phase_max_rad: float = 2.0 * np.pi
    target_doppler_min_hz: float = -50e3
    target_doppler_max_hz: float = 50e3

    snr_min_db: float = -20.0
    snr_max_db: float = -5.0
    jsr_min_db: float = 5.0
    jsr_max_db: float = 40.0

    jam_window_min_samples: int = 12
    jam_window_max_samples: int = 48
    jam_period_min_samples: int = 80
    jam_period_max_samples: int = 180
    jam_forward_delay_min_samples: int = 4
    jam_forward_delay_max_samples: int = 36
    jam_repeats_min: int = 2
    jam_repeats_max: int = 6
    jam_replay_gap_max_samples: int = 24
    jammer_freq_offset_min_hz: float = -50e3
    jammer_freq_offset_max_hz: float = 50e3

    clean_threshold_db: float = 0.0
    reliability_floor: float = 0.0
    reliability_exponent: float = 1.0

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    jam_types: Tuple[str, ...] = JAM_TYPES
    scene_id_start: int = 0
    fixed_partition: str | None = None

    @property
    def chirp_rate_hz_s(self) -> float:
        return self.bandwidth_hz / self.pulse_width_s

    @property
    def pri_s(self) -> float:
        return self.signal_length / self.sample_rate_hz

    @property
    def nominal_pulse_samples(self) -> float:
        return self.pulse_width_s * self.sample_rate_hz

    def validate(self) -> None:
        minimum_scenes = 1 if self.fixed_partition is not None else 3
        if self.num_scenes < minimum_scenes:
            if self.fixed_partition is None:
                raise ValueError("num_scenes must be at least 3 so all splits are non-empty")
            raise ValueError("num_scenes must be positive for a fixed partition")
        if self.variants_per_type < 1:
            raise ValueError("variants_per_type must be positive")
        if self.signal_length < 16:
            raise ValueError("signal_length is too small")
        if self.sample_rate_hz <= 0 or self.pulse_width_s <= 0 or self.bandwidth_hz <= 0:
            raise ValueError("sample rate, pulse width, and bandwidth must be positive")
        if self.nominal_pulse_samples + 2 * self.min_guard_samples >= self.signal_length:
            raise ValueError(
                "pulse plus two guard regions must fit inside signal_length; "
                f"got pulse={self.nominal_pulse_samples:.1f}, "
                f"guards={2 * self.min_guard_samples}, length={self.signal_length}"
            )
        if not 0 < self.target_amplitude_min <= self.target_amplitude_max:
            raise ValueError("invalid target amplitude range")
        if self.snr_min_db > self.snr_max_db or self.jsr_min_db > self.jsr_max_db:
            raise ValueError("invalid SNR or JSR range")
        if not 1 <= self.jam_window_min_samples <= self.jam_window_max_samples:
            raise ValueError("invalid jammer window range")
        if not 1 <= self.jam_period_min_samples <= self.jam_period_max_samples:
            raise ValueError("invalid jammer period range")
        if self.jam_period_max_samples >= self.nominal_pulse_samples:
            raise ValueError("jammer period must be shorter than the target pulse")
        if not 1 <= self.jam_repeats_min <= self.jam_repeats_max:
            raise ValueError("invalid repeat range")
        if any(jam_type not in JAM_TYPES for jam_type in self.jam_types):
            raise ValueError(f"jam_types must be chosen from {JAM_TYPES}")
        if len(set(self.jam_types)) != len(self.jam_types):
            raise ValueError("jam_types must not contain duplicates")
        if not isinstance(self.scene_id_start, int) or self.scene_id_start < 0:
            raise ValueError("scene_id_start must be a non-negative integer")
        if self.scene_id_start + self.num_scenes > np.iinfo(np.int64).max:
            raise ValueError("scene_id namespace exceeds int64 range")
        if self.fixed_partition is not None:
            if self.fixed_partition not in SPLIT_TO_CODE:
                raise ValueError(
                    f"fixed_partition must be one of {tuple(SPLIT_TO_CODE)} or None"
                )
        else:
            ratios = np.asarray(
                [self.train_ratio, self.val_ratio, self.test_ratio], dtype=float
            )
            if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-8):
                raise ValueError("train/val/test ratios must be positive and sum to one")


def _inclusive_randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(int(low), int(high) + 1))


def _draw_target(config: GeneratorConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Draw one continuously delayed, randomized complex LFM target."""

    pulse_samples = config.nominal_pulse_samples
    delay_low = float(config.min_guard_samples)
    delay_high = float(config.signal_length - config.min_guard_samples) - pulse_samples
    delay_samples = float(rng.uniform(delay_low, delay_high))
    delay_s = delay_samples / config.sample_rate_hz

    amplitude = float(rng.uniform(config.target_amplitude_min, config.target_amplitude_max))
    initial_phase = float(rng.uniform(config.target_phase_min_rad, config.target_phase_max_rad))
    doppler_hz = float(rng.uniform(config.target_doppler_min_hz, config.target_doppler_max_hz))

    sample_index = np.arange(config.signal_length, dtype=np.float64)
    local_time_s = (sample_index - delay_samples) / config.sample_rate_hz
    active = (local_time_s >= 0.0) & (local_time_s < config.pulse_width_s)
    centered_time_s = local_time_s - 0.5 * config.pulse_width_s

    phase = (
        initial_phase
        + np.pi * config.chirp_rate_hz_s * centered_time_s**2
        + 2.0 * np.pi * doppler_hz * centered_time_s
    )
    s = np.zeros(config.signal_length, dtype=np.complex128)
    s[active] = amplitude * np.exp(1j * phase[active])

    metadata = {
        "amplitude": amplitude,
        "initial_phase_rad": initial_phase,
        "doppler_hz": doppler_hz,
        "delay_s": delay_s,
        "delay_samples": delay_samples,
        "active_start_index": int(np.flatnonzero(active)[0]),
        "active_end_index_exclusive": int(np.flatnonzero(active)[-1] + 1),
        "signal_power_active": float(np.mean(np.abs(s[active]) ** 2)),
    }
    return s, active, metadata


def _draw_noise(
    signal_power: float,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    snr_db = float(rng.uniform(config.snr_min_db, config.snr_max_db))
    noise_power = float(signal_power / (10.0 ** (snr_db / 10.0)))
    sigma = np.sqrt(noise_power / 2.0)
    epsilon = sigma * (
        rng.standard_normal(config.signal_length)
        + 1j * rng.standard_normal(config.signal_length)
    )
    return epsilon.astype(np.complex128), snr_db, noise_power


def _add_segment(destination: np.ndarray, start: int, segment: np.ndarray) -> None:
    """Add a complex segment with safe right-edge clipping."""

    if start >= destination.shape[0] or start + segment.shape[0] <= 0:
        return
    src_left = max(0, -start)
    dst_left = max(0, start)
    length = min(segment.shape[0] - src_left, destination.shape[0] - dst_left)
    if length > 0:
        destination[dst_left : dst_left + length] += segment[src_left : src_left + length]


def _capture_segments(
    s: np.ndarray,
    active: np.ndarray,
    window: int,
    period: int,
    sampling_phase: int,
) -> List[Tuple[int, np.ndarray]]:
    active_idx = np.flatnonzero(active)
    pulse_start = int(active_idx[0])
    pulse_end = int(active_idx[-1] + 1)
    first_start = pulse_start + sampling_phase
    latest_start = pulse_end - window
    if first_start > latest_start:
        first_start = pulse_start
    segments: List[Tuple[int, np.ndarray]] = []
    for start in range(first_start, latest_start + 1, period):
        segments.append((start, s[start : start + window].copy()))
    if not segments:
        segments.append((pulse_start, s[pulse_start : pulse_start + window].copy()))
    return segments


def _raw_jammer(
    s: np.ndarray,
    active: np.ndarray,
    jam_type: str,
    params: Mapping[str, int],
) -> np.ndarray:
    raw = np.zeros_like(s, dtype=np.complex128)
    window = int(params["window_samples"])
    segments = _capture_segments(
        s,
        active,
        window,
        int(params["period_samples"]),
        int(params["sampling_phase_samples"]),
    )
    forward_delay = int(params["forward_delay_samples"])
    repeats = int(params["repeat_count"])
    replay_gap = int(params["replay_gap_samples"])
    spacing = window + replay_gap

    if jam_type == "direct":
        for source_start, segment in segments:
            _add_segment(raw, source_start + window + forward_delay, segment)
    elif jam_type == "repeat":
        for source_start, segment in segments:
            replay_start = source_start + window + forward_delay
            for replay_index in range(repeats):
                _add_segment(raw, replay_start + replay_index * spacing, segment)
    elif jam_type == "cyclic":
        memory: List[np.ndarray] = []
        for source_start, segment in segments:
            memory.append(segment)
            replay_start = source_start + window + forward_delay
            # Reversing the accumulated memory gives a cyclic/stacked replay
            # pattern distinct from direct and same-segment repetition.
            replay_order = list(reversed(memory))[:repeats]
            for replay_index, stored_segment in enumerate(replay_order):
                _add_segment(raw, replay_start + replay_index * spacing, stored_segment)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported jam_type: {jam_type}")
    return raw


def _draw_jammer(
    s: np.ndarray,
    active: np.ndarray,
    signal_power: float,
    jam_type: str,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float | int | str]]:
    window_max = min(config.jam_window_max_samples, max(1, int(active.sum()) // 3))
    window = _inclusive_randint(rng, config.jam_window_min_samples, window_max)
    period_low = max(config.jam_period_min_samples, window + 1)
    period = _inclusive_randint(rng, period_low, config.jam_period_max_samples)
    sampling_phase = _inclusive_randint(rng, 0, max(0, period - window))
    forward_delay = _inclusive_randint(
        rng,
        config.jam_forward_delay_min_samples,
        config.jam_forward_delay_max_samples,
    )
    repeats = _inclusive_randint(rng, config.jam_repeats_min, config.jam_repeats_max)
    replay_gap = _inclusive_randint(rng, 0, config.jam_replay_gap_max_samples)
    jsr_db = float(rng.uniform(config.jsr_min_db, config.jsr_max_db))
    channel_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    freq_offset_hz = float(
        rng.uniform(config.jammer_freq_offset_min_hz, config.jammer_freq_offset_max_hz)
    )

    integer_params = {
        "window_samples": window,
        "period_samples": period,
        "sampling_phase_samples": sampling_phase,
        "forward_delay_samples": forward_delay,
        "repeat_count": repeats,
        "replay_gap_samples": replay_gap,
    }
    raw = _raw_jammer(s, active, jam_type, integer_params)

    time_s = np.arange(config.signal_length, dtype=np.float64) / config.sample_rate_hz
    raw *= np.exp(1j * (channel_phase + 2.0 * np.pi * freq_offset_hz * time_s))
    support = np.abs(raw) > np.finfo(np.float64).eps
    if not np.any(support):
        raise RuntimeError("generated jammer has empty support; check jammer parameter ranges")

    raw_power_active = float(np.mean(np.abs(raw[support]) ** 2))
    desired_power_active = float(signal_power * 10.0 ** (jsr_db / 10.0))
    amplitude_scale = float(np.sqrt(desired_power_active / raw_power_active))
    j = raw * amplitude_scale
    actual_power_active = float(np.mean(np.abs(j[support]) ** 2))
    actual_jsr_db = float(10.0 * np.log10(actual_power_active / signal_power))

    metadata: Dict[str, float | int | str] = {
        "jam_type": jam_type,
        "jsr_db": jsr_db,
        "actual_jsr_db": actual_jsr_db,
        "window_samples": window,
        "period_samples": period,
        "sampling_phase_samples": sampling_phase,
        "forward_delay_samples": forward_delay,
        "repeat_count": repeats,
        "replay_gap_samples": replay_gap,
        "channel_phase_rad": channel_phase,
        "frequency_offset_hz": freq_offset_hz,
        "amplitude_scale": amplitude_scale,
        "support_fraction": float(np.mean(support)),
        "power_active": actual_power_active,
    }
    return j.astype(np.complex128), metadata


def _split_counts(num_scenes: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    """Return non-empty split counts that sum exactly to ``num_scenes``."""

    raw = np.asarray(ratios, dtype=np.float64) * num_scenes
    counts = np.floor(raw).astype(int)
    remainder = num_scenes - int(counts.sum())
    for index in np.argsort(-(raw - counts))[:remainder]:
        counts[index] += 1

    # For the supported minimum of three scenes, guarantee every split exists.
    for empty_index in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError("not enough scenes for non-empty train/val/test splits")
        counts[donor] -= 1
        counts[empty_index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def assign_scene_splits(config: GeneratorConfig) -> Dict[int, str]:
    """Assign each physical scene to exactly one deterministic split."""

    scene_ids = np.arange(
        config.scene_id_start,
        config.scene_id_start + config.num_scenes,
        dtype=np.int64,
    )
    if config.fixed_partition is not None:
        return {int(scene_id): config.fixed_partition for scene_id in scene_ids}

    n_train, n_val, n_test = _split_counts(
        config.num_scenes,
        (config.train_ratio, config.val_ratio, config.test_ratio),
    )
    split_rng = np.random.default_rng(config.seed + 104_729)
    split_rng.shuffle(scene_ids)

    assignment: Dict[int, str] = {}
    for scene_id in scene_ids[:n_train]:
        assignment[int(scene_id)] = "train"
    for scene_id in scene_ids[n_train : n_train + n_val]:
        assignment[int(scene_id)] = "val"
    for scene_id in scene_ids[n_train + n_val : n_train + n_val + n_test]:
        assignment[int(scene_id)] = "test"
    return assignment


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


def generate_dataset(
    output_path: str | Path,
    config: GeneratorConfig | None = None,
    *,
    write_manifest: bool = True,
) -> Path:
    """Generate and save one compressed V2 NPZ dataset.

    Returns the resolved NPZ path.  A sibling ``.manifest.json`` is written by
    default and contains the full generator configuration plus per-scene and
    per-sample physical metadata.
    """

    config = config or GeneratorConfig()
    config.validate()
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    scene_splits = assign_scene_splits(config)

    component_lists: Dict[str, List[np.ndarray]] = {
        "s": [],
        "epsilon": [],
        "u": [],
        "j": [],
        "y": [],
        "reliability": [],
        "z": [],
    }
    scalar_lists: Dict[str, List] = {
        "sample_id": [],
        "scene_id": [],
        "variant_index": [],
        "split": [],
        "split_code": [],
        "jam_type": [],
        "snr_db": [],
        "jsr_db": [],
        "noise_power": [],
        "target_amplitude": [],
        "target_initial_phase_rad": [],
        "target_doppler_hz": [],
        "target_delay_s": [],
        "target_delay_samples": [],
        "target_active_start_index": [],
        "target_active_end_index_exclusive": [],
        "jam_window_samples": [],
        "jam_period_samples": [],
        "jam_sampling_phase_samples": [],
        "jam_forward_delay_samples": [],
        "jam_repeat_count": [],
        "jam_replay_gap_samples": [],
        "jam_channel_phase_rad": [],
        "jam_frequency_offset_hz": [],
        "jam_support_fraction": [],
    }
    scene_manifest: List[dict] = []
    sample_manifest: List[dict] = []
    sample_id = 0

    for local_scene_index in range(config.num_scenes):
        scene_id = config.scene_id_start + local_scene_index
        s, active, target_meta = _draw_target(config, rng)
        epsilon, snr_db, noise_power = _draw_noise(
            target_meta["signal_power_active"], config, rng
        )
        u = s + epsilon
        split_name = scene_splits[scene_id]

        scene_manifest.append(
            {
                "scene_id": scene_id,
                "split": split_name,
                "snr_db": snr_db,
                "noise_power": noise_power,
                "target": target_meta,
            }
        )

        for jam_type in config.jam_types:
            for variant_index in range(config.variants_per_type):
                j, jam_meta = _draw_jammer(
                    s,
                    active,
                    target_meta["signal_power_active"],
                    jam_type,
                    config,
                    rng,
                )
                y = u + j
                labels = build_reliability_labels(
                    j,
                    noise_power=noise_power,
                    clean_threshold_db=config.clean_threshold_db,
                    reliability_floor=config.reliability_floor,
                    reliability_exponent=config.reliability_exponent,
                )

                component_lists["s"].append(s.astype(np.complex64))
                component_lists["epsilon"].append(epsilon.astype(np.complex64))
                component_lists["u"].append(u.astype(np.complex64))
                component_lists["j"].append(j.astype(np.complex64))
                component_lists["y"].append(y.astype(np.complex64))
                component_lists["reliability"].append(labels.reliability.astype(np.float32))
                component_lists["z"].append(labels.z.astype(np.float32))

                scalar_lists["sample_id"].append(sample_id)
                scalar_lists["scene_id"].append(scene_id)
                scalar_lists["variant_index"].append(variant_index)
                scalar_lists["split"].append(split_name)
                scalar_lists["split_code"].append(SPLIT_TO_CODE[split_name])
                scalar_lists["jam_type"].append(jam_type)
                scalar_lists["snr_db"].append(snr_db)
                scalar_lists["jsr_db"].append(jam_meta["jsr_db"])
                scalar_lists["noise_power"].append(noise_power)
                scalar_lists["target_amplitude"].append(target_meta["amplitude"])
                scalar_lists["target_initial_phase_rad"].append(target_meta["initial_phase_rad"])
                scalar_lists["target_doppler_hz"].append(target_meta["doppler_hz"])
                scalar_lists["target_delay_s"].append(target_meta["delay_s"])
                scalar_lists["target_delay_samples"].append(target_meta["delay_samples"])
                scalar_lists["target_active_start_index"].append(target_meta["active_start_index"])
                scalar_lists["target_active_end_index_exclusive"].append(
                    target_meta["active_end_index_exclusive"]
                )
                scalar_lists["jam_window_samples"].append(jam_meta["window_samples"])
                scalar_lists["jam_period_samples"].append(jam_meta["period_samples"])
                scalar_lists["jam_sampling_phase_samples"].append(
                    jam_meta["sampling_phase_samples"]
                )
                scalar_lists["jam_forward_delay_samples"].append(
                    jam_meta["forward_delay_samples"]
                )
                scalar_lists["jam_repeat_count"].append(jam_meta["repeat_count"])
                scalar_lists["jam_replay_gap_samples"].append(jam_meta["replay_gap_samples"])
                scalar_lists["jam_channel_phase_rad"].append(jam_meta["channel_phase_rad"])
                scalar_lists["jam_frequency_offset_hz"].append(jam_meta["frequency_offset_hz"])
                scalar_lists["jam_support_fraction"].append(jam_meta["support_fraction"])

                sample_manifest.append(
                    {
                        "sample_id": sample_id,
                        "scene_id": scene_id,
                        "variant_index": variant_index,
                        "split": split_name,
                        "snr_db": snr_db,
                        "jsr_db": jam_meta["jsr_db"],
                        "jammer": jam_meta,
                    }
                )
                sample_id += 1

    config_dict = asdict(config)
    config_dict.update(
        {
            "chirp_rate_hz_s": config.chirp_rate_hz_s,
            "pri_s": config.pri_s,
            "nominal_pulse_samples": config.nominal_pulse_samples,
        }
    )
    metadata_summary = {
        "schema_version": SCHEMA_VERSION,
        "observation_model": "u = s + epsilon; y = u + j",
        "primary_recovery_target": "u",
        "reliability_definition": "noise_power / (noise_power + |j|^2)",
        "clean_state_definition": (
            "z=1 when |j|^2 <= noise_power * 10^(clean_threshold_db/10)"
        ),
        "split_unit": "scene_id",
        "partition_strategy": (
            "fixed_partition" if config.fixed_partition is not None else "scene_group_ratios"
        ),
        "no_oracle_noise_gate": True,
        "config": config_dict,
    }

    arrays: Dict[str, np.ndarray] = {
        key: np.stack(values, axis=0) for key, values in component_lists.items()
    }
    arrays.update(
        {
            "sample_id": np.asarray(scalar_lists["sample_id"], dtype=np.int64),
            "scene_id": np.asarray(scalar_lists["scene_id"], dtype=np.int64),
            "variant_index": np.asarray(scalar_lists["variant_index"], dtype=np.int16),
            "split": np.asarray(scalar_lists["split"], dtype="<U16"),
            "split_code": np.asarray(scalar_lists["split_code"], dtype=np.int8),
            "jam_type": np.asarray(scalar_lists["jam_type"], dtype="<U8"),
            "snr_db": np.asarray(scalar_lists["snr_db"], dtype=np.float32),
            "jsr_db": np.asarray(scalar_lists["jsr_db"], dtype=np.float32),
            "noise_power": np.asarray(scalar_lists["noise_power"], dtype=np.float32),
            "target_amplitude": np.asarray(scalar_lists["target_amplitude"], dtype=np.float32),
            "target_initial_phase_rad": np.asarray(
                scalar_lists["target_initial_phase_rad"], dtype=np.float32
            ),
            "target_doppler_hz": np.asarray(
                scalar_lists["target_doppler_hz"], dtype=np.float32
            ),
            "target_delay_s": np.asarray(scalar_lists["target_delay_s"], dtype=np.float64),
            "target_delay_samples": np.asarray(
                scalar_lists["target_delay_samples"], dtype=np.float32
            ),
            "target_active_start_index": np.asarray(
                scalar_lists["target_active_start_index"], dtype=np.int32
            ),
            "target_active_end_index_exclusive": np.asarray(
                scalar_lists["target_active_end_index_exclusive"], dtype=np.int32
            ),
            "jam_window_samples": np.asarray(
                scalar_lists["jam_window_samples"], dtype=np.int16
            ),
            "jam_period_samples": np.asarray(
                scalar_lists["jam_period_samples"], dtype=np.int16
            ),
            "jam_sampling_phase_samples": np.asarray(
                scalar_lists["jam_sampling_phase_samples"], dtype=np.int16
            ),
            "jam_forward_delay_samples": np.asarray(
                scalar_lists["jam_forward_delay_samples"], dtype=np.int16
            ),
            "jam_repeat_count": np.asarray(
                scalar_lists["jam_repeat_count"], dtype=np.int16
            ),
            "jam_replay_gap_samples": np.asarray(
                scalar_lists["jam_replay_gap_samples"], dtype=np.int16
            ),
            "jam_channel_phase_rad": np.asarray(
                scalar_lists["jam_channel_phase_rad"], dtype=np.float32
            ),
            "jam_frequency_offset_hz": np.asarray(
                scalar_lists["jam_frequency_offset_hz"], dtype=np.float32
            ),
            "jam_support_fraction": np.asarray(
                scalar_lists["jam_support_fraction"], dtype=np.float32
            ),
            "sample_rate_hz": np.asarray(config.sample_rate_hz, dtype=np.float64),
            "pulse_width_s": np.asarray(config.pulse_width_s, dtype=np.float64),
            "bandwidth_hz": np.asarray(config.bandwidth_hz, dtype=np.float64),
            "chirp_rate_hz_s": np.asarray(config.chirp_rate_hz_s, dtype=np.float64),
            "pri_s": np.asarray(config.pri_s, dtype=np.float64),
            "schema_version": np.asarray(SCHEMA_VERSION),
            "metadata_json": np.asarray(json.dumps(metadata_summary, default=_json_ready)),
        }
    )
    np.savez_compressed(output, **arrays)

    if write_manifest:
        manifest = {
            **metadata_summary,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "npz_file": output.name,
            "num_samples": sample_id,
            "num_scenes": config.num_scenes,
            "split_scene_counts": {
                split: int(sum(value == split for value in scene_splits.values()))
                for split in sorted(set(scene_splits.values()))
            },
            "split_sample_counts": {
                split: int(sum(value == split for value in scalar_lists["split"]))
                for split in sorted(set(scalar_lists["split"]))
            },
            "scenes": scene_manifest,
            "samples": sample_manifest,
        }
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_ready),
            encoding="utf-8",
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "radar_v2_small.npz",
        help="Output compressed NPZ path.",
    )
    parser.add_argument("--num-scenes", type=int, default=24)
    parser.add_argument("--variants-per-type", type=int, default=1)
    parser.add_argument("--signal-length", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-20.0)
    parser.add_argument("--snr-max-db", type=float, default=-5.0)
    parser.add_argument("--jsr-min-db", type=float, default=5.0)
    parser.add_argument("--jsr-max-db", type=float, default=40.0)
    parser.add_argument("--jam-types", nargs="+", choices=JAM_TYPES, default=list(JAM_TYPES))
    parser.add_argument("--no-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    config = GeneratorConfig(
        num_scenes=args.num_scenes,
        variants_per_type=args.variants_per_type,
        signal_length=args.signal_length,
        seed=args.seed,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        jsr_min_db=args.jsr_min_db,
        jsr_max_db=args.jsr_max_db,
        jam_types=tuple(args.jam_types),
    )
    output = generate_dataset(args.output, config, write_manifest=not args.no_manifest)
    with np.load(output, allow_pickle=False) as data:
        split_counts = {
            split: int(np.sum(data["split"] == split))
            for split in sorted(set(data["split"].astype(str)))
        }
        print(
            f"Generated {data['y'].shape[0]} samples from "
            f"{np.unique(data['scene_id']).size} scenes at {output}"
        )
        print(f"Component shape: {data['y'].shape}; split samples: {split_counts}")
        print(
            "Identity max errors: "
            f"u-(s+epsilon)={np.max(np.abs(data['u'] - data['s'] - data['epsilon'])):.3e}, "
            f"y-(u+j)={np.max(np.abs(data['y'] - data['u'] - data['j'])):.3e}"
        )
    return output


if __name__ == "__main__":
    main()


__all__ = [
    "DEVELOPMENT_PARTITIONS",
    "GeneratorConfig",
    "JAM_TYPES",
    "SCHEMA_VERSION",
    "assign_scene_splits",
    "generate_dataset",
    "main",
]
