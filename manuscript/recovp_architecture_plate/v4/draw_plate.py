"""Render the paired schematic and data views from the frozen v3 figure cache."""
from hashlib import sha256
from pathlib import Path
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 180, 104
STEM = HERE / "ReCoVP_Hybrid_plate_v4"
CACHE = HERE.parent / "v3" / "plate_sample_data_v3.npz"
CACHE_SHA = "f0ed4f5f28a96716f35f85aca1299a8b9296450a008d27d409dd36ba732a2564"
NAVY, TEAL = "#1B365D", "#186F6B"
INK, MUTED = "#152A40", "#475569"
BLUE, GREEN = ("#E8F0F8", "#3B6287"), ("#E9F4F3", "#1F6E6A")
WHITE, GOLD = ("white", MUTED), ("#FBF5E6", "#9A6B16")
DASH = (0, (2.5, 1.8))
mpl.rcParams.update({"font.family": "Arial",
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": "ReCoVP-plate-v4", "text.color": INK,
                     "axes.labelcolor": MUTED, "path.simplify": False})


def label(ax, x, y, value, size=7, color=INK, align="center", **kwargs):
    if "$" in value and any(symbol in value for symbol in ("_", "^")):
        size = max(size, 7.2)
    return ax.text(x, y, value, ha=align, va="center", color=color,
                   fontsize=size, linespacing=1.05, zorder=5, **kwargs)


def rounded(ax, x, y, w, h, palette, lw=.65, radius=.6, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=palette[0],
                 edgecolor=palette[1], linewidth=lw, zorder=zorder,
                 boxstyle=f"round,pad=0,rounding_size={radius}"))


def box(ax, x, y, w, h, name, sub=None, palette=WHITE, gate=False):
    rounded(ax, x, y, w, h, palette, lw=.8 if palette == GOLD else .65)
    if gate:
        ax.add_patch(Rectangle((x+.32, y+.4), 1, h-.8, color=TEAL, lw=0, zorder=4))
    mid = x+w/2+(.4 if gate else 0)
    label(ax, mid, y+h/2-(1.55 if sub else 0), name, 7)
    if sub:
        label(ax, mid, y+h/2+1.65, sub, 6.7)


def route(ax, *points, real=False, width=None, head=True):
    """Render orthogonal paths and arrowheads in physical millimetres."""
    color = TEAL if real else NAVY
    xy = np.asarray(points, dtype=float)
    ax.plot(*xy.T, lw=width or (.8 if real else .85), color=color,
            ls=DASH if real else "-", solid_capstyle="butt", zorder=2)
    if head:
        delta = xy[-1]-xy[-2]
        tangent = delta/np.linalg.norm(delta)
        base = xy[-1]-1.1*tangent
        normal = .57*tangent[::-1]*[-1, 1]
        ax.add_patch(Polygon([xy[-1], base+normal, base-normal], lw=0,
                             fc=color, zorder=4))


def operator(ax, x, y, sign):
    ax.add_patch(Circle((x, y), 1.9, lw=.8, ec=NAVY, fc="white", zorder=3))
    label(ax, x, y-.03, sign, 8.6)


def ref(ax, x, y, letter):
    label(ax, x, y, letter, 6.3, color=MUTED, weight="bold",
          bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="#A3B4C2", lw=.45))


def data_axis(fig, x, y, w, h):
    return fig.add_axes([x/WIDTH, (HEIGHT-y-h)/HEIGHT, w/WIDTH, h/HEIGHT],
                        facecolor="none")


def overview(ax):
    for x, w in [(1.2, 98), (100.5, 78.3)]:
        rounded(ax, x, 7.3, w, 42.5, ("#F6F8FA", "none"), 0, 1.5, 0)
    # Signal spine and two subtraction sites.
    for left, right in [(19, 91.1), (94.9, 154.1), (157.9, 161)]:
        route(ax, (left, 14), (right, 14))
    for x in [93, 156]:
        operator(ax, x, 14, "−")
    label(ax, 105, 10.9, r"$\widehat{\boldsymbol{u}}_1$")
    box(ax, 21, 21.5, 17, 8.5, "RMS norm", r"$\boldsymbol{x}=\boldsymbol{y}/r$")
    box(ax, 54, 20.5, 32, 10.5, "Variable projection",
        r"$(\mathbf{B}^H\mathbf{B}+\mathbf{\Lambda})\hat{\mathbf{c}}=\mathbf{B}^H\mathbf{x}$", GOLD)
    label(ax, 71, 18, "FP64", 6.3, MUTED)
    route(ax, (29, 14), (29, 21.5))
    route(ax, (38, 25.75), (54, 25.75))
    route(ax, (86, 25.75), (104, 25.75))
    route(ax, (93, 25.75), (93, 15.9))
    label(ax, 95, 29.2, r"$\hat{\mathbf{j}}_1=r\mathbf{B}\hat{\mathbf{c}}$")
    label(ax, 37, 18.9, r"$r$ (RMS)", 6.6, MUTED, align="left")
    box(ax, 21, 36, 17, 8.5, r"Invariant $\phi$", "4 channels")
    box(ax, 41, 36, 13.7, 8.5, "1D U-Net", r"$\mathbf{h}$", GREEN)
    box(ax, 58, 36, 17, 8.5, "Dictionary", r"$\mathbf{B}\in\mathbb{C}^{T\times16}$")
    ref(ax, 73.4, 33.7, "c")
    box(ax, 78.5, 36, 19.3, 8.5, "Evidence gate", r"$\rho_k\rightarrow\lambda_k$", GREEN, True)
    route(ax, (27, 30), (27, 36))
    route(ax, (18, 40.25), (21, 40.25))
    route(ax, (19.5, 40.25), (19.5, 47.3), (66, 47.3), (66, 44.5))
    for left, right in [(38, 41), (54.7, 58), (75, 78.5)]:
        route(ax, (left, 40.25), (right, 40.25), real=True)
    route(ax, (65, 36), (65, 31))
    route(ax, (84, 36), (84, 31), real=True)
    label(ax, 62.7, 33.5, r"$\mathbf{B}$")
    label(ax, 81.5, 33.5, r"$\mathbf{\Lambda}$", color=TEAL)
    box(ax, 104, 21.5, 15.5, 8.5, "Normalize", r"$(\hat{\mathbf{u}}_1,\hat{\mathbf{j}}_1)/r$")
    route(ax, (111, 14), (111, 21.5))
    box(ax, 124.5, 21.5, 20.7, 8.5, "Complex U-Net", palette=BLUE)
    ref(ax, 144.1, 19.2, "e")
    route(ax, (119.5, 25.75), (124.5, 25.75))
    route(ax, (145.2, 25.75), (154.1, 25.75))
    label(ax, 150, 23, r"$\widetilde{\boldsymbol{\delta}}$")
    operator(ax, 156, 25.75, "×")
    route(ax, (156, 23.85), (156, 15.9))
    label(ax, 152.9, 18.8, r"$\boldsymbol{\delta}_2$")
    # Explicit shared net label for r without a leftward return path.
    label(ax, 162.8, 30.8, r"$r$ (RMS)", 6.4, MUTED)
    route(ax, (157.3,28.6), (157.3,27.15), real=True)
    box(ax, 104, 36, 15.5, 8.5, r"Invariant $\psi$", "6 channels")
    route(ax, (111, 30), (111, 36))
    box(ax, 129, 36, 21.5, 8.5, "Pulse gate MLP", "20 statistics", GREEN, True)
    route(ax, (119.5, 40.25), (129, 40.25), real=True)
    route(ax, (122, 40.25), (122, 32.8), (133, 32.8), (133, 30), real=True)
    route(ax, (150.5, 40.25), (156, 40.25), (156, 27.65), real=True)
    label(ax, 158.4, 34.1, r"$\alpha$", color=TEAL)


def profiles(fig, canvas, data):
    axes = []
    template = data["p"]
    profiles = [np.abs(np.correlate(data[key], template, "valid")) for key in ("y", "u")]
    for x, values, name in zip([3, 161.5], profiles,
                               [r"$\mathbf{y}$", r"$\widehat{\mathbf{u}}$"]):
        ax = data_axis(fig, x, 10, 15.2, 8.5)
        db = 20*np.log10(np.maximum(values/profiles[0].max(), 1e-12))
        ax.plot(np.maximum(db, -65), color=NAVY, lw=.35, clip_on=True)
        for floor in (0, -60):
            ax.axhline(floor, color="#CCD6DF", linewidth=.35, zorder=0)
        ax.set(xlim=(0, len(db)-1), ylim=(-65, 1))
        ax.axis("off")
        label(canvas, x+7.6, 22.8, "MF "+name+" (dB)", 6.3)
        axes.append(ax)
    ax = data_axis(fig, 3, 37, 15, 6)
    ax.plot(template.real, lw=.3, color=NAVY)
    ax.set(xlim=(0, len(template)-1))
    ax.axis("off")
    label(canvas, 10.3, 45.9, r"Template $\mathbf{p}$", 6.4)
    return axes


def evidence(fig, canvas, data):
    start, stop = data["window"]
    rate = data["sample_rate_hz"].item()/1e6
    order = np.argsort(-data["rho"], kind="stable")
    mag = data["dictionary_magnitude"]
    kernel = np.array([1, 2, 3, 2, 1])/9
    smooth = np.array([np.convolve(row, kernel, "same")[start:stop] for row in mag.T[order]])
    db = 20*np.log10(np.maximum(smooth/mag.max(), 1e-12))
    heat = data_axis(fig, 8.6, 62, 31.5, 24.5)
    cmap = LinearSegmentedColormap.from_list("ink", ["#F7FAFC", "#7FA8C9", NAVY])
    im = heat.imshow(db, origin="upper", extent=((start-.5)/rate,(stop-.5)/rate,16.5,.5),
                     aspect="auto", cmap=cmap, interpolation="nearest", vmin=-32, vmax=0)
    heat.set(xticks=[4, 6, 8, 10], yticks=[1, 4, 8, 12, 16])
    label(canvas, 23.8, 58.6, r"$\mathbf{b}_k=\mathrm{norm}(\mathbf{m}_k\odot\mathbf{t}_k)$", 6.8)
    label(canvas, 1.8, 74.3, "Atom rank", 6.3, rotation=90, rotation_mode="anchor")
    label(canvas, 24.3, 92, "Fast time (μs)", 6.3)
    bars = data_axis(fig, 43, 62, 12.4, 24.5)
    bars.barh(np.arange(1, 17), data["rho"][order]/data["rho"].max(), height=.67,
              color="#5FA39F", linewidth=0)
    bars.set(ylim=(16.5,.5), xlim=(0,1), xticks=[0, 1], yticks=[])
    label(canvas, 49.2, 58.6, r"$\rho_k/\max\,\rho$", 6.7, TEAL)
    for ax in [heat, bars]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(labelsize=6.2, length=1.3, width=.4, pad=1.2, colors=MUTED)
    cb = data_axis(fig, 9, 95, 22, 1.25)
    fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[]).outline.set_visible(False)
    label(canvas, 8, 95.6, "−32", 6.2, align="right")
    label(canvas, 32, 95.6, "0 dB", 6.2, align="left")
    return heat, bars


def residual(ax):
    label(ax, 62.5, 69, r"$\mathbf{z}$")
    route(ax, (64.3, 69), (69, 69))
    box(ax, 69, 65, 18.2, 8, "Bias-free", "complex conv.", BLUE)
    route(ax, (87.2, 69), (96.6, 69))
    label(ax, 90.8, 66.3, r"$\mathbf{v}$")
    operator(ax, 98.5, 69, "×")
    operator(ax, 106.5, 69, "+")
    route(ax, (100.4, 69), (104.6, 69))
    route(ax, (108.4, 69), (111, 69))
    label(ax, 112.8, 69, r"$\mathbf{z}^+$")
    route(ax, (66, 69), (66, 60.6), (106.5, 60.6), (106.5, 67.1))
    box(ax, 72, 77, 20.5, 6.1, r"$\log|\mathbf{v}|_\epsilon\oplus\boldsymbol{\psi}$")
    route(ax, (90.8, 69), (90.8, 77), real=True)
    box(ax, 72, 87, 20.5, 8, "Real conv.", r"SiLU $\rightarrow\sigma$", GREEN)
    route(ax, (82.2, 83.1), (82.2, 87), real=True)
    route(ax, (92.5, 91), (98.5, 91), (98.5, 70.9), real=True)
    label(ax, 101, 81.4, r"$\mathbf{g}$", color=TEAL, align="left")


def network(ax):
    # Feature widths encode channel count ordinally; heights encode resolution.
    levels = [(121,67,2.6,13), (128,75,3.4,9.8), (135,82,4.2,7.2),
              (143.5,88,5.4,5.4), (152,82,4.2,7.2), (159,75,3.4,9.8),
              (166,67,2.6,13)]
    for x,y,w,h in levels:
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc=BLUE[0],ec=BLUE[1],lw=.7,zorder=3))
    for l,r in zip(levels, levels[1:]):
        x,y,w,h = l
        xx,yy,ww,hh = r
        if yy>y:
            route(ax, (x,y+h/2), (x,yy), (xx-ww/2,yy))
        else:
            route(ax, (x+w/2,y), (xx,y), (xx,yy+hh/2))
    for i, dim in enumerate([r"$T\times9$", r"$T/2\times13$", r"$T/4\times17$"]):
        x,y,w,_ = levels[i]
        xx,_,ww,_ = levels[-1-i]
        route(ax, (x+w/2,y), (xx-ww/2,y), width=.5)
        label(ax, 143.5, y-2.1, dim, 6.6)
    ref(ax, 119.7, 58.5, "d")
    label(ax, 143.5, 59.2, "Skip connections", 6.4, MUTED)
    box(ax, 170, 64.7, 6.3, 4.6, r"$1\times1$", palette=BLUE)
    route(ax, (167.3,67), (170,67))
    route(ax, (176.3,67), (179,67))
    label(ax, 177.6, 63.3, r"$\widetilde{\boldsymbol{\delta}}$")
    label(ax, 143.5, 93.3, r"$T/8\times21$", 6.7)
    label(ax, 143.5, 96.2, r"Dilation $1,2,4$", 6.4)
    label(ax, 123, 94.3, "AvgPool ↓2", 6.3, MUTED)
    label(ax, 165, 94.3, "Linear ↑2", 6.3, MUTED)
    # One invariant rail along the left contour, with separate labeled taps
    # beside the decoder to avoid crossing its skip connections.
    route(ax, (116.5,64), (116.5,89.5), head=False, real=True)
    label(ax, 116.5, 61.1, r"$\boldsymbol{\psi}$", color=TEAL)
    for x,y,w,h in levels[:4]:
        route(ax, (116.5,y+1.5), (x-w/2,y+1.5), real=True)
    for x,y,w,h in levels[4:]:
        route(ax, (x+w/2-.5,y+h/2+2.7), (x+w/2-.5,y+h/2), real=True)
        label(ax, x+w/2+1.1, y+h/2+3.4, r"$\psi$", 6.4, TEAL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--audit", action="store_true", help="run nature-figure alignment audit (requires its scripts on PYTHONPATH)")
    args = parser.parse_args()
    if sha256(args.cache.read_bytes()).hexdigest() != CACHE_SHA:
        raise ValueError("Unexpected figure-cache SHA-256")
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), facecolor="white")
    ax = fig.add_axes((0,0,1,1), xlim=(0,WIDTH), ylim=(HEIGHT,0), aspect="equal")
    ax.axis("off")
    overview(ax)
    with np.load(args.cache, allow_pickle=False) as data:
        pair = profiles(fig, ax, data)
        heat, bars = evidence(fig, ax, data)
    residual(ax)
    network(ax)
    for x,y,title in [(2,3.8,"(a)  Evidence-gated variable projection"),
                      (101,3.8,"(b)  Equivariant residual refinement"),
                      (2,54,"(c)  Local dictionary and evidence"),
                      (60.8,54,"(d)  Complex residual block"),
                      (116,54,"(e)  Four-level complex U-Net")]:
        label(ax,x,y,title,8.5,align="left",weight="bold")
    for x in (58.8,114.6):
        ax.plot([x,x],[58.5,96.5],color="#E2E8F0",lw=.5)
    handles = [Line2D([],[],color=NAVY,lw=.9), Line2D([],[],color=TEAL,lw=.8,ls=DASH)]
    handles.extend(Patch(fc=fill,ec=edge,lw=.65) for fill,edge in [WHITE,GOLD,BLUE,GREEN])
    fig.legend(handles, ["Complex stream","Phase-invariant real","Analytic op.","FP64 VP solve",
                         "Complex net","Real / gate net"], ncol=6, frameon=False,
               loc="center", bbox_to_anchor=(.5,2.6/HEIGHT), fontsize=6.4,
               handlelength=1.65, columnspacing=1.45, handletextpad=.5)
    fig.canvas.draw()
    if args.audit:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        for name, axes in [("profiles",pair),("atoms",[heat,bars])]:
            require_matplotlib_panel_alignment(fig, axes=axes,
                panel_ids=["left","right"], row_groups=[["left","right"]],
                json_out=HERE/f"alignment_{name}.json")
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, metadata={"CreationDate":None})
    fig.savefig(STEM.with_suffix(".png"), dpi=600)
    for mode, name in [("none","editable"),("path","outlined")]:
        with mpl.rc_context({"svg.fonttype":mode}):
            fig.savefig(HERE/f"{STEM.name}_{name}.svg",dpi=600,metadata={"Date":None})
    plt.close(fig)


if __name__ == "__main__":
    main()
