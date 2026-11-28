"""Make transparent insets from one paired observation in the registered archive."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MASTER = ROOT / "data/hybrid_condition_master_20261127"
LOCAL = ROOT / "outputs/jest_signal_insets"
SNR_DB = 10
JSR_DB = 30


def draw_trace(samples, interval, name, color, line_width):
    """Use the original I samples throughout the stated half-open interval."""
    start, stop = interval
    fig = plt.figure(figsize=(35 / 25.4, 11 / 25.4))
    ax = fig.add_axes([0.008, 0.045, 0.984, 0.91])
    ax.plot(np.arange(start, stop), samples[start:stop].real,
            linewidth=line_width, color=color, solid_capstyle="round",
            solid_joinstyle="round")
    peak = float(np.max(np.abs(samples[start:stop].real)))
    ax.set(xlim=(start, stop - 1), ylim=(-peak * 1.045, peak * 1.045))
    ax.set_axis_off()
    fig.savefig(HERE / f"{name}.svg", transparent=True, metadata={"Date": None})
    fig.savefig(HERE / f"{name}.png", transparent=True, dpi=900)
    plt.close(fig)


def main():
    archive = MASTER / "components/test_0_0.npz"
    registration = json.loads((MASTER / "REGISTRATION.json").read_text())
    assert SNR_DB in registration["snr_grid_db"] and JSR_DB in registration["jsr_grid_db"]
    expected = json.loads((MASTER / "DATA_MANIFEST.json").read_text())["component_sha256"]
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("Component archive SHA-256 differs from the registered value")
    with np.load(archive, allow_pickle=False) as data:
        parent = 0
        kind = data["jam_types"].tolist().index("repeat")
        j0 = data["j"][parent, kind]
        pulse = int(np.flatnonzero(np.any(j0 != 0, axis=1))[0])
        parent_id = int(data["parent_id"][parent])
        signal = data["s"][parent, kind, pulse]
        noise = data["epsilon"][parent, kind, pulse] * np.float32(10 ** (-SNR_DB / 20))
        interference = j0[pulse] * np.float32(10 ** (JSR_DB / 20))
        observation = signal + interference + noise
    LOCAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LOCAL / "observations.npz", s=signal, clean=signal, noise=noise,
        unjammed=signal + noise, y=observation, j=interference,
        parent_id=parent_id, parent_index=parent, pulse_index=pulse, jam_type="repeat",
        snr_db=SNR_DB, jsr_db=JSR_DB, source_sha256=digest,
    )
    # Central 192 samples of the clean pulse retain readable oscillations at 35 mm.
    tx = np.flatnonzero(np.abs(signal) > np.abs(signal).max() / 2)
    center = (int(tx[0]) + int(tx[-1]) + 1) // 2
    clean_crop = (center - 96, center + 96)
    active = np.flatnonzero(np.abs(interference) > np.abs(interference).max() / 100)
    rx_crop = (max(0, int(active[0]) - 15), min(signal.size, int(active[-1]) + 16))
    full = (0, signal.size)
    with plt.rc_context({"path.simplify": False, "svg.hashsalt": "jest-signal-inset"}):
        draw_trace(signal, clean_crop, "01_lfm_clean_detail", "#185583", .52)
        draw_trace(observation, rx_crop, "02_isrrj_observation_detail", "#A64220", .34)
        draw_trace(signal, full, "01_lfm_clean_full", "#185583", .26)
        draw_trace(observation, full, "02_isrrj_observation_full", "#A64220", .26)
        draw_trace(signal + noise, clean_crop, "01_lfm_unjammed_10dB_detail", "#185583", .44)
    print(json.dumps(dict(parent_id=parent_id, pulse_index=pulse, clean_crop=clean_crop,
                          observation_crop=rx_crop, full=full, component_sha256=digest)))


if __name__ == "__main__":
    main()
