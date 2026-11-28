"""Plot an archived LFM reference and its paired receiver record, at native sampling."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTER = ROOT / "data/hybrid_condition_master_20261127"
LOCAL = ROOT / "outputs/jest_signal_insets/receiver_view"
RATE = 60e6
INK = "#144970"


def load_record():
    path = MASTER / "components/test_0_0.npz"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = json.loads((MASTER / "DATA_MANIFEST.json").read_text())["component_sha256"]
    if digest != expected:
        raise ValueError("Source data identity mismatch")
    with np.load(path, allow_pickle=False) as data:
        parent, pulse = 17, 2
        kind = data["jam_types"].tolist().index("repeat")
        index = (parent, kind, pulse)
        s = data["s"][index]
        j = data["j"][index] * np.float32(10 ** (20 / 20))
        noise = data["epsilon"][index] * np.float32(10 ** (-10 / 20))
        parent_id = int(data["parent_id"][parent])
    y = s + j + noise
    LOCAL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOCAL / "observations.npz", s=s, clean=s, j=j, noise=noise,
                        y=y, unjammed=s + noise, parent_id=parent_id, parent_index=parent,
                        pulse_index=pulse, jam_type="repeat", snr_db=10, jsr_db=20,
                        source_sha256=digest)
    return s, noise, y, parent_id, digest


def receiver_plot(t, s, y):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.8, 4.8))
    fig.subplots_adjust(left=.12, right=.975, top=.91, bottom=.14, hspace=.45)
    for ax, z, label in zip(axes, (s, y), ("(a)  Clean LFM echo", "(b)  Received signal with ISRRJ")):
        ax.plot(t, z.real, color=INK, linewidth=.68)
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_ylabel("I amplitude (a.u.)")
        ax.set_ylim(-1.08 * abs(z.real).max(), 1.08 * abs(z.real).max())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.grid(color="#e2e7eb", linewidth=.45)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set(xlim=(0, 30), xticks=np.arange(0, 31, 5), xlabel="Fast time (µs)")
    fig.savefig(HERE / "receiver_records.png", dpi=200)
    fig.savefig(HERE / "receiver_records.svg", metadata={"Date": None})
    plt.close(fig)


def inset(t, z, filename):
    # One shared window includes the entire target pulse and all nonzero interference.
    keep = t < 16
    fig = plt.figure(figsize=(42 / 25.4, 12 / 25.4))
    ax = fig.add_axes([.008, .025, .984, .95])
    ax.plot(t[keep], z[keep].real, color=INK, linewidth=.24)
    peak = np.abs(z[keep].real).max()
    ax.set(xlim=(0, 16), ylim=(-peak * 1.05, peak * 1.05))
    ax.set_axis_off()
    fig.savefig(HERE / (filename + ".svg"), transparent=True, metadata={"Date": None})
    fig.savefig(HERE / (filename + ".png"), transparent=True, dpi=900)
    plt.close(fig)


def main():
    s, noise, y, parent, digest = load_record()
    t = np.arange(s.size) / RATE * 1e6
    active = np.flatnonzero(s)
    phase = np.unwrap(np.angle(s[active].astype(np.complex128)))
    freq_mhz = np.diff(phase) * RATE / (2 * np.pi * 1e6)
    slope = np.polyfit(t[active][1:], freq_mhz, 1)[0]
    if active.size != 600 or not np.isclose(slope, 3.0, rtol=1e-5):
        raise ValueError("Unexpected LFM pulse width or instantaneous frequency slope")
    with plt.rc_context({"path.simplify": False, "svg.hashsalt": "receiver-record", "font.size": 10}):
        receiver_plot(t, s, y)
        inset(t, s, "clean_lfm")
        inset(t, y, "isrrj_received")
        inset(t, s + noise, "lfm_in_noise")
    record = dict(parent_id=parent, pulse_index=2, source_sha256=digest,
                  number_of_samples=int(t.size), sample_rate_hz=RATE,
                  nominal_snr_db=10, nominal_jsr_db=20,
                  target_sample_range=[int(active[0]), int(active[-1]) + 1],
                  instantaneous_frequency_mhz=[freq_mhz[0], freq_mhz[-1]],
                  slope_mhz_per_us=slope, full_window_us=[0, 30], inset_window_us=[0, 16])
    (HERE / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
