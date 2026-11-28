"""Plot the frozen dictionary over the time window shown in the ISRRJ inset."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
META = json.loads((HERE.parent / "dictionary_provenance.json").read_text())
START, END, BIN = 213, 686, 11
CMAP = LinearSegmentedColormap.from_list("blue_to_gold", [
    (0.0, "#112649"), (.28, "#19538a"), (.48, "#3989b2"),
    (.62, "#9dbfc0"), (.75, "#ebcf77"), (.88, "#ffdc54"), (1., "#fff4ae")])


def read_panel():
    path = ROOT / META["matrix"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != META["matrix_file_sha256"]:
        raise ValueError("Dictionary checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        amplitude = np.abs(data["B"]).T
    power = amplitude[:, START:END] ** 2
    accumulated = np.cumsum(power, axis=1)
    midpoints = (accumulated >= accumulated[:, -1:] * .5).argmax(axis=1)
    rows = np.argsort(midpoints, kind="stable")
    rms = np.sqrt(power[rows].reshape(16, -1, BIN).mean(axis=-1))
    return 20 * np.log10(rms / amplitude.max()), rows


def render(panel, rows, annotated):
    fig = plt.figure(figsize=(6.6, 3.55) if annotated else (1.75, .7))
    ax = fig.add_axes([.12, .29, .85, .66] if annotated else [0, 0, 1, 1])
    times = np.arange(START, END + 1, BIN) / 60
    mesh = ax.pcolormesh(times, np.arange(17), panel, shading="flat",
                         cmap=CMAP, vmin=-32, vmax=0, antialiased=False, linewidth=0)
    ax.set_ylim(16, 0)
    name = "dictionary_inset"
    if annotated:
        name = "dictionary_axes"
        ax.set(xlabel="Fast time (µs)", ylabel="Atom ID", yticks=np.arange(16) + .5,
               yticklabels=(rows + 1).tolist(), xticks=np.arange(4, 12))
        ax.tick_params(axis="y", length=0, labelsize=7)
        ax.tick_params(axis="x", length=3, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(mesh, cax=fig.add_axes([.12, .085, .42, .026]),
                           orientation="horizontal", ticks=[-32, -24, -16, -8, 0])
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8, length=2)
        fig.text(.565, .092, "RMS magnitude (dB rel. to peak)", va="center", fontsize=8)
    else:
        ax.set_axis_off()
    fig.savefig(HERE / (name + ".png"), dpi=240 if annotated else 900, transparent=not annotated)
    fig.savefig(HERE / (name + ".svg"), transparent=not annotated, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "local-dictionary-rms", "svg.fonttype": "none"})
    panel, rows = read_panel()
    for annotated in (False, True):
        render(panel, rows, annotated)
    print("Original IDs, top to bottom:", (rows + 1).tolist())
