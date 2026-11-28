"""Draw contiguous slices of the saved receiver record at native sampling."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[3] / "outputs/jest_signal_insets/receiver_view/observations.npz"
INK = "#144970"


def main():
    receipt = json.loads((HERE.parent / "dictionary_provenance.json").read_text())
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != receipt["observations_npz_sha256"]:
        raise ValueError("Unexpected receiver record")
    with np.load(SOURCE, allow_pickle=False) as record:
        s, j, y = (record[key] for key in ("s", "j", "y"))
    target, jammer = s != 0, j != 0
    panels = [
        ("clean_lfm", s, target),
        ("received_overlap", y, target & jammer),
        ("received_all_returns", y, target | jammer),
    ]
    plt.rcParams.update({"path.simplify": False, "svg.hashsalt": "receiver-crops"})
    sheet, axes = plt.subplots(2, 1, figsize=(7.2, 3.7))
    sheet.subplots_adjust(left=.035, right=.965, bottom=.045, top=.87, hspace=.85)
    for i, (name, z, mask) in enumerate(panels):
        bounds = np.flatnonzero(mask)
        start, stop = int(bounds[0]), int(bounds[-1] + 1)
        trace = z[start:stop].real
        fig = plt.figure(figsize=(50 / 25.4, 13 / 25.4))
        ax = fig.add_axes([.004, .02, .992, .96])
        for axis in ([ax, axes[i]] if i < 2 else [ax]):
            axis.plot(trace, color=INK, linewidth=.26 if axis is ax else .73)
            limit = 1.04 * np.abs(trace).max()
            axis.set(xlim=(0, len(trace) - 1), ylim=(-limit, limit))
            axis.set_axis_off()
        if i < 2:
            title = ("Clean LFM echo", "Received echo with ISRRJ")[i]
            axes[i].set_title(f"{title}     |     {start/60:.2f}–{stop/60:.2f} µs",
                              loc="left", fontsize=11, color=INK)
        for ext in ("svg", "png"):
            fig.savefig(HERE / f"{name}.{ext}", transparent=True, dpi=700,
                        metadata={"Date": None} if ext == "svg" else {})
        plt.close(fig)
    sheet.savefig(HERE / "preview.png", dpi=170)
    plt.close(sheet)


if __name__ == "__main__":
    main()
