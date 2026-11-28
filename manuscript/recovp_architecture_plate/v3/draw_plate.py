"""Compose Figure 1 at its physical print size from the verified illustration cache."""
from hashlib import sha256
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import detail
import top


HERE = Path(__file__).resolve().parent
STEM = HERE / "ReCoVP_Hybrid_plate_v3"
W, H = 178, 176
CACHE_SHA = "f0ed4f5f28a96716f35f85aca1299a8b9296450a008d27d409dd36ba732a2564"
mpl.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "svg.hashsalt": "recovp-figure-1-v3", "path.simplify": False,
    "font.size": 7.5, "text.color": "#111827", "axes.labelcolor": "#334155",
})


def axes(fig, x, y, w, h):
    return fig.add_axes((x / W, y / H, w / W, h / H), facecolor="none")


def dictionary(fig, data):
    first, end = data["window"]
    rate = data["sample_rate_hz"].item() / 1e6
    amplitude = data["dictionary_magnitude"]
    kernel = np.array([1, 2, 3, 2, 1]) / 9
    smooth = np.column_stack([np.convolve(row, kernel, mode="same") for row in amplitude.T])
    db = 20 * np.log10(np.maximum(smooth[first:end].T / amplitude.max(), 1e-12))
    heat = axes(fig, 9, 27, 33.5, 39)
    # Sample centres are at i / rate. Row boundaries remain discrete.
    image = heat.imshow(db, extent=((first-.5)/rate, (end-.5)/rate, 16.5, .5),
                        aspect="auto", origin="upper", interpolation="nearest",
                        cmap="cividis", vmin=-32, vmax=0, resample=False)
    heat.set(xticks=[4, 6, 8, 10], yticks=[1, 4, 8, 12, 16])
    heat.set_xlabel("Fast time (μs)", fontsize=7, labelpad=1)
    heat.set_ylabel("Atom", fontsize=7, labelpad=1)
    bars = axes(fig, 44, 27, 12, 39)
    bars.barh(np.arange(1, 17), np.log10(1+data["rho"]), height=.64,
              color="#AC7060", linewidth=0)
    bars.set(ylim=(16.5, .5), xlim=(0, 5), xticks=[0, 5], yticks=[])
    bars.set_xlabel(r"$\log_{10}(1+\rho_k)$", fontsize=7, labelpad=1)
    for ax in (heat, bars):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=1.6, width=.45, pad=1.3, labelsize=6.5, colors="#334155")
    cbax = axes(fig, 9, 14, 17, 1.6)
    cb = fig.colorbar(image, cax=cbax, orientation="horizontal", ticks=[-32, 0])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.2, pad=1, labelsize=6.5)
    fig.text(29/W, 14.6/H, "Magnitude (dB)", va="center", fontsize=6.8)
    fig.text(9/W, 69/H, r"$\boldsymbol{b}_k=\mathrm{norm}(\boldsymbol{m}_k\odot\boldsymbol{t}_k)$",
             va="center", fontsize=7.5)


def leader(fig, points):
    x,y=np.array(points).T
    fig.add_artist(Line2D(x/W, y/H, transform=fig.transFigure, color="#94A3B8",
                          linewidth=.6, linestyle=(0,(2,2)), zorder=.2))


def main():
    cache = HERE / "plate_sample_data_v3.npz"
    if sha256(cache.read_bytes()).hexdigest() != CACHE_SHA:
        raise ValueError("The figure cache differs from the verified illustration.")
    fig = plt.figure(figsize=(W/25.4, H/25.4), facecolor="white")
    a, b = axes(fig, 1.5, 84, 95, 85), axes(fig, 98, 84, 78, 85)
    with np.load(cache, allow_pickle=False) as data:
        window = slice(*data["window"])
        sources = top.draw(a, b, data["y"][window], data["p"], data["u"][window])
        dictionary(fig, data)
    d, e = axes(fig, 60, 10, 52, 65), axes(fig, 114, 10, 62, 65)
    detail.draw(d, e)
    for x, y, title in [
        (2, 172, "(a)  Evidence-gated variable projection"),
        (99, 172, "(b)  Equivariant residual refinement"),
        (2, 78, "(c)  Dictionary and evidence"),
        (60, 78, "(d)  Complex residual block"),
        (114, 78, "(e)  Four-level complex U-Net"),
    ]:
        fig.text(x/W, y/H, title, va="center", fontsize=9, weight="bold")
    dx, dy = sources["dictionary"]
    ux, uy = sources["unet"]
    leader(fig, [(1.5+95*dx,84+85*dy), (53,82), (53,67)])
    leader(fig, [(98+78*ux,84+85*uy), (166,84), (168,74)])
    leader(fig, [(119.6,10+65*.735), (113,62), (106,68), (102.4,68)])
    keys = [Line2D([],[],color=top.SIGNAL,lw=1),
            Line2D([],[],color=top.REAL,lw=.9,ls=(0,(3,2))),
            Patch(facecolor="white",edgecolor="#334155",lw=.75),
            Patch(facecolor="#DCE8F2",edgecolor="#3B5B78",lw=.75)]
    fig.legend(keys, ["Complex stream", "Phase-invariant real stream",
                      "Analytic operator", "Learnable module"],
               loc="center", bbox_to_anchor=(.5,3.6/H), frameon=False,
               ncol=4, fontsize=6.8, handlelength=2.1, columnspacing=1.7)
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, metadata={"CreationDate":None})
    fig.savefig(STEM.with_suffix(".png"), dpi=600)
    for mode,suffix in [("none","editable"),("path","outlined")]:
        with mpl.rc_context({"svg.fonttype":mode}):
            fig.savefig(STEM.with_name(STEM.name+"_"+suffix+".svg"), dpi=600,
                        metadata={"Date":None})
    plt.close(fig)


if __name__ == "__main__":
    main()
