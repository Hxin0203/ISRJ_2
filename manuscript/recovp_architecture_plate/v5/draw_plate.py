"""Render the paired schematic and data views from the frozen v3 figure cache."""
from hashlib import sha256
from pathlib import Path
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Polygon, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 180, 104
STEM = HERE / "ReCoVP_Hybrid_plate_v5"
CACHE = HERE.parent / "v3" / "plate_sample_data_v3.npz"
CACHE_SHA = "f0ed4f5f28a96716f35f85aca1299a8b9296450a008d27d409dd36ba732a2564"
NAVY, TEAL = "#1B365D", "#186F6B"
INK, MUTED = "#152A40", "#475569"
BLUE, GREEN = ("#E8F0F8", "#3B6287"), ("#E9F4F3", "#1F6E6A")
WHITE, GOLD = ("white", MUTED), ("#FBF5E6", "#9A6B16")
DASH = (0, (2.5, 1.8))
mpl.rcParams.update({"font.family": "Arial",
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": "ReCoVP-plate-v5", "text.color": INK,
                     "axes.labelcolor": MUTED, "path.simplify": False})


def label(ax, x, y, value, size=7, color=INK, align="center", **kwargs):
    if "$" in value and any(symbol in value for symbol in ("_", "^")):
        size = max(size, 7.2)
    return ax.text(x, y, value, ha=align, va="center", color=color,
                   fontsize=size, linespacing=1.05, zorder=5, **kwargs)


def rounded(ax, x, y, w, h, palette, lw=.65, radius=.6, zorder=3):
    return ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=palette[0],
                 edgecolor=palette[1], linewidth=lw, zorder=zorder,
                 boxstyle=f"round,pad=0,rounding_size={radius}"))


def box(ax, x, y, w, h, name, sub=None, palette=WHITE, gate=False):
    outer = rounded(ax, x, y, w, h, palette, lw=.8 if palette == GOLD else .65)
    if gate:
        accent = ax.add_patch(Rectangle((x, y), .8, h, color=TEAL, lw=0, zorder=4))
        accent.set_clip_path(outer)
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
        base = xy[-1]-1.45*tangent
        normal = .44*tangent[::-1]*[-1, 1]
        ax.add_patch(Polygon([xy[-1], base+normal, base+.28*tangent, base-normal], lw=0,
                             fc=color, zorder=4))


def operator(ax, x, y, sign):
    ax.add_patch(Circle((x, y), 1.9, lw=.8, ec=NAVY, fc="white", zorder=3))
    label(ax, x, y-.03, sign, 8.6)


def ref(ax, x, y, letter):
    # Badge bottom edge meets the module top; the glyph stays clear of the border.
    label(ax, x, y-1.45, letter, 6, color=MUTED, weight="bold",
          bbox=dict(boxstyle="round,pad=0.14,rounding_size=.16", fc="#F8FAFC", ec="#64748B", lw=.45))


def junction(ax, x, y, real=False):
    ax.add_patch(Circle((x, y), .48, lw=0, fc=TEAL if real else NAVY, zorder=5))


def data_axis(fig, x, y, w, h):
    return fig.add_axes([x/WIDTH, (HEIGHT-y-h)/HEIGHT, w/WIDTH, h/HEIGHT],
                        facecolor="none")


def overview(ax):
    for x, w in [(1.2, 98), (100.5, 78.3)]:
        rounded(ax, x, 7.3, w, 42.5, ("#F6F8FA", "none"), 0, 1.5, 0)
    for left, right in [(19, 91.1), (94.9, 152.6), (156.4, 161.4)]:
        route(ax, (left, 14), (right, 14))
    for x in [93, 154.5]:
        operator(ax, x, 14, "−")
    label(ax, 104.5, 10.8, r"$\widehat{\boldsymbol{u}}_1$")
    box(ax, 20.5, 21.5, 15, 8.5, "RMS norm", r"$\boldsymbol{x}=\boldsymbol{y}/r$")
    box(ax, 57.5, 20.5, 31, 10.5, "Variable projection",
        r"$(\boldsymbol{B}^H\boldsymbol{B}+\boldsymbol{\Lambda})\hat{\boldsymbol{c}}=\boldsymbol{B}^H\boldsymbol{x}$", GOLD)
    label(ax, 73, 18.1, "FP64", 6.3, MUTED)
    route(ax, (28, 14), (28, 21.5))
    route(ax, (35.5, 25.75), (57.5, 25.75))
    route(ax, (88.5, 25.75), (103.5, 25.75))
    route(ax, (93, 25.75), (93, 15.9))
    label(ax, 96, 29.2, r"$\hat{\boldsymbol{j}}_1=r\boldsymbol{B}\hat{\boldsymbol{c}}$")
    route(ax, (33, 21.5), (33, 19), real=True)
    label(ax, 33, 17.6, r"$r$", 6.6, TEAL)
    box(ax, 20.5, 36, 15, 8.5, r"Invariant $\boldsymbol{\phi}$", "4 channels")
    box(ax, 38.5, 36, 12.5, 8.5, "1D U-Net", r"$\boldsymbol{h}$", GREEN)
    rounded(ax, 54.5, 36, 20, 8.5, WHITE)
    label(ax, 64.5, 38.3, r"Learned $\tau_k,\boldsymbol{m}_k$", 6.7, TEAL)
    label(ax, 64.5, 42.05, r"Dict. $\boldsymbol{B}\in\mathbb{C}^{T\times16}$", 6.8)
    ref(ax, 73.3, 36, "c")
    box(ax, 78, 36, 19, 8.5, "Evidence gate",
        r"$\rho_k(\boldsymbol{B},\boldsymbol{x})\to\lambda_k$", GREEN, True)
    route(ax, (28, 30), (28, 36))
    route(ax, (17.4, 40.25), (20.5, 40.25))
    route(ax, (19, 40.25), (19, 47.3), (65, 47.3), (65, 44.5))
    for left, right in [(35.5, 38.5), (51, 54.5), (74.5, 78)]:
        route(ax, (left, 40.25), (right, 40.25), real=True)
    route(ax, (55.8, 25.75), (55.8, 36))
    label(ax, 53.8, 32.1, r"$\boldsymbol{x}$")
    route(ax, (64.5, 36), (64.5, 31))
    route(ax, (85.5, 36), (85.5, 31), real=True)
    label(ax, 62.1, 33.4, r"$\boldsymbol{B}$")
    label(ax, 82.9, 33.4, r"$\boldsymbol{\Lambda}$", color=TEAL)
    for x,y in [(28,14),(55.8,25.75),(93,25.75),(19,40.25),(111,14)]:
        junction(ax,x,y)
    box(ax, 103.5, 21.5, 15, 8.5, "Normalize",
        r"$(\hat{\boldsymbol{u}}_1,\hat{\boldsymbol{j}}_1)/r$")
    route(ax, (111, 14), (111, 21.5))
    box(ax, 123.5, 21.5, 23, 8.5, "Complex U-Net", palette=BLUE)
    ref(ax, 145.3, 21.5, "e")
    route(ax, (118.5,25.75), (123.5,25.75))
    route(ax, (146.5,25.75), (152.6,25.75))
    label(ax, 149.6, 23.1, r"$\widetilde{\boldsymbol{\delta}}$")
    operator(ax, 154.5, 25.75, "×")
    route(ax, (154.5,23.85), (154.5,15.9))
    label(ax, 151.7, 18.9, r"$\boldsymbol{\delta}_2$")
    label(ax, 161.3, 25.75, r"$r$", color=TEAL)
    route(ax, (159.9,25.75), (156.4,25.75), real=True)
    box(ax, 103.5, 36, 15, 8.5, r"Invariant $\boldsymbol{\psi}$", "6 channels")
    route(ax, (111,30), (111,36))
    box(ax, 123.5, 36, 23, 8.5, "Pulse gate MLP", "20 statistics", GREEN, True)
    route(ax, (118.5,40.25), (123.5,40.25), real=True)
    route(ax, (121,40.25), (121,32.8), (134,32.8), (134,30), real=True)
    junction(ax,121,40.25,real=True)
    route(ax, (146.5,40.25), (154.5,40.25), (154.5,27.65), real=True)
    label(ax, 157, 34.1, r"$\alpha$", color=TEAL)


def profiles(fig, canvas, data):
    axes = []
    template = data["p"]
    amplitudes = [np.abs(np.correlate(data[key], template, "valid")) for key in ("y", "u")]
    reference = amplitudes[0].max()
    for i, (x, values, name) in enumerate(zip([5.5, 163], amplitudes,
                             [r"$\boldsymbol{y}$", r"$\widehat{\boldsymbol{u}}$"])):
        ax = data_axis(fig, x, 9.7, 13.3, 9)
        db = 20*np.log10(np.maximum(values/reference, 1e-12))
        bins = np.array_split(np.arange(db.size), 160)
        peaks = np.array([bucket[np.argmax(db[bucket])] for bucket in bins])
        shown = np.maximum(db[peaks], -65)
        ax.fill_between(peaks, -65, shown, color=NAVY, alpha=.10, lw=0)
        ax.plot(peaks, shown, color=NAVY, lw=.38)
        ax.set(xlim=(0, db.size-1), ylim=(-65, 1), xticks=[], yticks=[-60,0])
        ax.tick_params(axis="y", labelleft=(i==0), length=1, pad=.6, width=.4,
                       colors=MUTED, labelsize=5.5)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color("#94A3B8")
            spine.set_linewidth(.45)
        peak = np.argmax(db)
        ax.plot(peak, db[peak], ".", color=TEAL, ms=2, clip_on=False)
        if i:
            ax.text(.98, .97, f"−{abs(db[peak]):.1f} dB", transform=ax.transAxes,
                    va="top", ha="right", fontsize=5.8, color=TEAL)
        label(canvas, x+6.65, 22.6, "MF "+name+" (dB)", 6.3)
        axes.append(ax)
    ax = data_axis(fig, 3, 37, 14.5, 6)
    t = np.linspace(0, 1, 1000)
    ax.plot(t, np.cos(2*np.pi*(3*t+11*t*t)), lw=.45, color=NAVY)
    ax.set(xlim=(0,1), ylim=(-1.05,1.05))
    ax.axis("off")
    label(canvas, 10.2, 45.9, r"Template $\boldsymbol{p}$", 6.4)
    return axes


def evidence(fig, canvas, data):
    start, stop = data["window"]
    rate = data["sample_rate_hz"].item()/1e6
    order = np.argsort(-data["rho"], kind="stable")
    mag = data["dictionary_magnitude"]
    kernel = np.hanning(15)
    kernel /= kernel.sum()
    smooth = np.array([np.sqrt(np.convolve(row**2, kernel, "same")[start:stop])
                       for row in mag.T[order]])
    db = 20*np.log10(np.maximum(smooth/mag.max(), 1e-12))
    heat = data_axis(fig, 8.6, 62, 31.5, 24.5)
    cmap = LinearSegmentedColormap.from_list("ink", ["#F7FAFC", "#7FA8C9", NAVY])
    # Interpolate within each atom independently; no vertical interpolation between atoms.
    samples = np.arange(stop-start)
    centers = (np.arange(4*len(samples))+.5)/4-.5
    displayed = np.array([np.interp(centers, samples, row) for row in db])
    im = heat.imshow(displayed, origin="upper", extent=((start-.5)/rate,(stop-.5)/rate,16.5,.5),
                     aspect="auto", cmap=cmap, interpolation="nearest", vmin=-32, vmax=0)
    heat.set(xticks=[4, 6, 8, 10], yticks=[1, 4, 8, 12, 16])
    for boundary in np.arange(1.5,16):
        heat.axhline(boundary, color="white", lw=.38)
    label(canvas, 23.8, 58.6, r"$\boldsymbol{b}_k=\mathrm{norm}(\boldsymbol{m}_k\odot\boldsymbol{t}_k)$", 6.8)
    label(canvas, 1.8, 74.3, "Atom rank", 6.3, rotation=90, rotation_mode="anchor")
    label(canvas, 24.3, 92, "Fast time (μs)", 6.3)
    bars = data_axis(fig, 43, 62, 12.4, 24.5)
    strength = data["rho"][order]/data["rho"].max()
    for position in (.5,1):
        bars.axvline(position, ls=(0,(2,2)), lw=.4, color="#E2E8F0", zorder=0)
    bars.barh(np.arange(1,17), strength, height=.67,
              color=np.where(strength>.8, TEAL, "#7DB5B1"), linewidth=0)
    bars.axvline(0, lw=.5, color="#64748B", clip_on=False)
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
    label(ax, 62.5, 69, r"$\boldsymbol{z}$")
    route(ax, (64.3, 69), (68.5, 69))
    box(ax, 68.5, 65, 20, 8, "Bias-free", "complex conv.", BLUE)
    route(ax, (88.5, 69), (96.6, 69))
    label(ax, 90.8, 66.3, r"$\boldsymbol{v}$")
    operator(ax, 98.5, 69, "×")
    operator(ax, 106.5, 69, "+")
    route(ax, (100.4, 69), (104.6, 69))
    route(ax, (108.4, 69), (111, 69))
    label(ax, 112.8, 69, r"$\boldsymbol{z}^+$")
    route(ax, (66, 69), (66, 60.6), (106.5, 60.6), (106.5, 67.1))
    box(ax, 68.5, 77, 20, 6.1, r"$\log|\boldsymbol{v}|_\epsilon\oplus\boldsymbol{\psi}$")
    route(ax, (91.5, 69), (91.5, 80.05), (88.5, 80.05), real=True)
    junction(ax, 66, 69)
    junction(ax, 91.5, 69)
    box(ax, 68.5, 87, 20, 8, "Real conv.", r"SiLU $\rightarrow\sigma$", GREEN)
    route(ax, (78.5, 83.1), (78.5, 87), real=True)
    route(ax, (88.5, 91), (98.5, 91), (98.5, 70.9), real=True)
    label(ax, 101, 81.4, r"$\boldsymbol{g}$", color=TEAL, align="left")


def network(ax):
    # Feature widths encode channel count ordinally; heights encode resolution.
    levels = [(121,67,2.6,13), (128,75,3.4,9.8), (135,82,4.2,7.2),
              (143.5,88,5.4,5.4), (152,82,4.2,7.2), (159,75,3.4,9.8),
              (166,67,2.6,13)]
    for x,y,w,h in levels:
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,fc=BLUE[0],ec=BLUE[1],lw=.7,zorder=3))
    x,y,w,h = levels[3]
    for split in (-w/6, w/6):
        ax.plot([x+split,x+split],[y-h/2,y+h/2], color=BLUE[1], lw=.55, zorder=4)
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
    ref(ax, 120.4, 60.5, "d")
    for xpos, title in [(129,"Encoder"),(145,"Skip"),(165,"Decoder")]:
        label(ax, xpos, 58, title, 6.6, MUTED)
    box(ax, 170, 64.7, 6.3, 4.6, r"$1\times1$", palette=BLUE)
    route(ax, (167.3,67), (170,67))
    route(ax, (176.3,67), (179,67))
    label(ax, 177.6, 63.3, r"$\widetilde{\boldsymbol{\delta}}$")
    label(ax, 143.5, 93.3, r"$T/8\times21$", 6.7)
    label(ax, 143.5, 96.2, r"Dilation $1,2,4$", 6.4)
    label(ax, 123, 96.2, "AvgPool ↓2", 6.3, MUTED)
    label(ax, 165, 96.2, "Linear ↑2", 6.3, MUTED)
    label(ax, 124.2, 72.2, "↓2", 6.2, MUTED)
    label(ax, 162.8, 72.2, "↑2", 6.2, MUTED)
    route(ax, (116.5,64), (116.5,89.5), head=False, real=True)
    label(ax, 116.5, 61.1, r"$\boldsymbol{\psi}$", color=TEAL)
    for x,y,w,h in levels[:4]:
        route(ax, (116.5,y+1.5), (x-w/2,y+1.5), real=True)
        junction(ax,116.5,y+1.5,real=True)
    rail = 169
    route(ax, (rail,68.5), (rail,83.5), real=True, head=False)
    label(ax, 171.2, 71.5, r"$\boldsymbol{\psi}$", color=TEAL)
    for x,y,w,h in levels[4:]:
        route(ax, (rail,y+1.5), (x+w/2,y+1.5), real=True)
        junction(ax,rail,y+1.5,real=True)


class GateLegend(HandlerPatch):
    def create_artists(self, legend, original, xd, yd, width, height, size, transform):
        patch = Rectangle((-xd,-yd),width,height,fc=GREEN[0],ec=GREEN[1],
                          lw=.65,transform=transform)
        stripe = Rectangle((-xd,-yd),width*.16,height,fc=TEAL,lw=0,transform=transform)
        return [patch, stripe]


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
    gate_key = Patch()
    handles.append(gate_key)
    fig.legend(handles, ["Complex stream","Phase-invariant real","Analytic op.","FP64 VP solve",
                         "Complex net","Real net","Gate head"], ncol=7, frameon=False,
               handler_map={gate_key:GateLegend()},
               loc="center", bbox_to_anchor=(.5,3.4/HEIGHT), fontsize=6.4,
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
