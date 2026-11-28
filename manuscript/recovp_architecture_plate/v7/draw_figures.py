"""Draw the v7 paper figures. The only numerical input is the frozen plotting cache."""
import argparse
from hashlib import sha256
from pathlib import Path
import re

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "v3" / "plate_sample_data_v3.npz"
CACHE_SHA = "f0ed4f5f28a96716f35f85aca1299a8b9296450a008d27d409dd36ba732a2564"
WIDTH = 176
INK, EDGE = "#111827", "#1F2937"
BLUE, BROWN, RED = "#1B5E7A", "#9E4A16", "#B43B2E"
YELLOW, SKY, LILAC, CORAL, OLIVE = "#FBE69E", "#B8D9EE", "#D8C9EA", "#E5A29A", "#C6DCA8"
SECONDARY, HAIR = "#41505C", "#CBD5E1"
MATH_WITH_SCRIPT = 10.4  # STIX scripts at 70% remain larger than 7 pt in print.


def configure():
    for family in ("Times New Roman", "STIXGeneral", "Liberation Serif"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("A Times-compatible serif font is required.")
    mpl.rcParams.update({
        "font.family": family, "font.size": 8.5, "mathtext.fontset": "stix",
        "mathtext.fallback": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "recovp-v7-paper",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": EDGE,
        "axes.linewidth": .55, "xtick.color": INK, "ytick.color": INK,
        "path.simplify": False, "lines.solid_capstyle": "butt",
    })


def page(height):
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(height, 0), aspect="equal")
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, value, size=8.5, color=INK, ha="center", **style):
    return ax.text(x, y, value, ha=ha, va="center", fontsize=size, color=color,
                   linespacing=1.02, zorder=6, **style)


def path(ax, *points, color=BLUE, width=.88, head=True, head_length=1.5):
    """Place an arrowhead exactly at the terminal port, in millimetres."""
    xy = np.asarray(points, dtype=float)
    ax.plot(*xy.T, color=color, lw=width, solid_joinstyle="round", zorder=2)
    if head:
        direction = xy[-1] - xy[-2]
        direction /= np.linalg.norm(direction)
        base = xy[-1] - head_length * direction
        half = .34 * head_length * np.array([-direction[1], direction[0]])
        ax.add_patch(Polygon([xy[-1], base + half, base - half], fc=color, lw=0, zorder=4))


def tap(ax, x, y, color=BLUE):
    ax.add_patch(Circle((x, y), .40, fc=color, lw=0, zorder=4))


def operator(ax, x, y, symbol):
    ax.add_patch(Circle((x, y), 2.0, fc="white", ec=EDGE, lw=.77, zorder=3))
    if symbol == "×":
        strokes = [[(-.72, -.72), (.72, .72)], [(-.72, .72), (.72, -.72)]]
    else:
        strokes = [[(-1, 0), (1, 0)]]
        if symbol == "+":
            strokes.append([(0, -1), (0, 1)])
    for ends in strokes:
        coordinates = np.asarray(ends) + [x, y]
        ax.plot(*coordinates.T, color=INK, lw=.78, zorder=4)


def tile(ax, x, y, w, h, fill, primary, secondary=None, size=8.45, subsize=MATH_WITH_SCRIPT):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=EDGE, lw=.72, zorder=3))
    if secondary:
        label(ax, x+w/2, y+h*.28, primary, size)
        label(ax, x+w/2, y+h*.73, secondary, subsize)
    else:
        label(ax, x+w/2, y+h/2, primary, size)


def panel_title(ax, x, y, letter, title):
    label(ax, x, y, f"({letter})", 9.6, ha="left", weight="bold")
    label(ax, x+5.1, y, title, 9.6, ha="left")


def overview():
    fig, ax = page(64)
    spine, upper, lower = 13.7, 28.7, 45.0
    for x, w in [(8, 94.2), (105.8, 62.6)]:
        ax.add_patch(Rectangle((x, 17.5), w, 39.9, fc="none", ec=HAIR,
                               lw=.65, ls=(0, (3, 3)), zorder=0))
    label(ax, 8, 60.8, "Stage 1", 9.15, ha="left", weight="bold")
    label(ax, 20.6, 60.8, "Evidence-gated variable projection", 9.15, ha="left")
    label(ax, 105.8, 60.8, "Stage 2", 9.15, ha="left", weight="bold")
    label(ax, 118.4, 60.8, "Equivariant residual refinement", 9.15, ha="left")

    subtract1, subtract2 = 98.8, 158.7
    label(ax, 52, 6.5, "Complex Equivariant Stream", 9.5, BLUE, weight="bold")
    label(ax, 2.0, 8.2, r"$\boldsymbol{y}\in\mathbb{C}^{T}$", MATH_WITH_SCRIPT, ha="left")
    label(ax, 112.7, 9.3, r"$\widehat{\boldsymbol{u}}_1$", MATH_WITH_SCRIPT)
    label(ax, 171.4, 9.3, r"$\widehat{\boldsymbol{u}}$", 10.5)
    for left, right in [(2, subtract1-2), (subtract1+2, subtract2-2), (subtract2+2, 174)]:
        path(ax, (left, spine), (right, spine), width=1.35, head_length=1.9)
    for x in (subtract1, subtract2):
        operator(ax, x, spine, "−")

    # Observation tap serves both the real feature encoder and the complex solve.
    tap_x = 18.8
    path(ax, (tap_x, spine), (tap_x, 40.5))
    tap(ax, tap_x, spine)
    path(ax, (tap_x, upper), (54, upper))
    tap(ax, tap_x, upper)
    label(ax, 32.3, upper-3.1, r"$\boldsymbol{x}$", 9.0, BLUE)
    label(ax, 22.3, 20.7, r"$\div\,r$", 8.4, BLUE)
    label(ax, 42, 36.8, "Phase-Invariant Real Control", 8.75, BROWN, weight="bold")

    # Touching blocks share one boundary. The equation and its two inputs get their own space.
    tile(ax, 12.0, 40.5, 12.8, 9, OLIVE, r"Feat. $\boldsymbol{\phi}$")
    tile(ax, 24.8, 40.5, 13.8, 9, SKY, "1D U-Net")
    path(ax, (38.6, lower), (41.2, lower), color=BROWN)
    tile(ax, 41.2, 40.5, 14.9, 9, LILAC, "Delay &", "window net", 8.4, 8.0)
    label(ax, 48.7, 52, r"$K=16$", 8.15, RED)
    path(ax, (56.1, lower), (58.25, lower), color=BROWN)
    # Only the exposed top and side of the second sheet are visible.
    ax.add_patch(Rectangle((58.9, 39.85), 18.2, .65, fc="#EDF3E5", lw=0, zorder=2.8))
    ax.add_patch(Rectangle((76.45, 40.5), .65, 8.35, fc="#EDF3E5", lw=0, zorder=2.8))
    ax.plot([58.9, 58.9, 77.1, 77.1, 76.45], [40.5, 39.85, 39.85, 48.85, 48.85],
            color=EDGE, lw=.54, zorder=3)
    tile(ax, 58.25, 40.5, 18.2, 9, OLIVE, "LFM Dict.", r"$\boldsymbol{B}\in\mathbb{C}^{T\times16}$")
    path(ax, (77.1, lower), (79.2, lower))
    tile(ax, 79.2, 40.5, 20.5, 9, LILAC, "Evidence Gate",
         r"$\rho_k(\boldsymbol{B},\boldsymbol{x})$")
    label(ax, 67.5, 20.45, "closed-form solve", 7.5, SECONDARY)
    tile(ax, 54, 23.55, 40.2, 10.3, CORAL, "Variable Projection",
         r"$(\boldsymbol{B}^{H}\boldsymbol{B}+\boldsymbol{\Lambda})\widehat{\boldsymbol{c}}"
         r"=\boldsymbol{B}^{H}\boldsymbol{x}$", size=9.0)
    path(ax, (67.1, 39.85), (67.1, 33.85))
    path(ax, (89.5, 40.5), (89.5, 33.85), color=BROWN, width=1.05)
    label(ax, 70.2, 36.9, r"$\boldsymbol{B}$", 8.9, BLUE)
    label(ax, 92.5, 37.1, r"$\boldsymbol{\Lambda}$", 9, BROWN)

    # A schematic chirp identifies the known template; quantitative traces live in Fig. 3.
    t = np.linspace(0, 1, 300)
    ax.plot(2.0+8*t, 54.45-1.0*np.sin(2*np.pi*(1.3*t+3.7*t*t)), lw=.58, color=BLUE)
    label(ax, 28.3, 51.9, r"Template $\boldsymbol{p}$", 8.0)
    path(ax, (10.0, 54.45), (67.1, 54.45), (67.1, 49.5), width=.72)
    path(ax, (18.8, 54.45), (18.8, 49.5), width=.72)
    tap(ax, 18.8, 54.45)

    # The two complex estimates form the normalized pair at the stage-2 input.
    pair = 116.7
    net_left, net_right = 130, 151
    path(ax, (94.2, upper), (net_left, upper))
    path(ax, (subtract1, upper), (subtract1, spine+2))
    tap(ax, subtract1, upper)
    label(ax, 90.9, 20.5, r"$\widehat{\boldsymbol{j}}_1=r\boldsymbol{B}\widehat{\boldsymbol{c}}$",
          MATH_WITH_SCRIPT, BLUE)
    path(ax, (pair, spine), (pair, upper), head=False)
    tap(ax, pair, spine)
    tap(ax, pair, upper)
    label(ax, 124.55, 21.0, r"$(\widehat{\boldsymbol{u}}_1,\widehat{\boldsymbol{j}}_1)/r$",
          MATH_WITH_SCRIPT)
    tile(ax, net_left, 24.6, net_right-net_left, 8.2, SKY, "Complex U-Net")
    label(ax, 141.7, 36.45, "4 levels, Fig. 2", 7.5, SECONDARY)
    path(ax, (net_right, upper), (subtract2-2, upper))
    label(ax, 154.5, 25.05, r"$\widetilde{\boldsymbol{\delta}}$", 9.8, BLUE)
    operator(ax, subtract2, upper, "×")
    path(ax, (subtract2, upper-2), (subtract2, spine+2))
    label(ax, 163.0, 19.9, r"$\boldsymbol{\delta}_2$", MATH_WITH_SCRIPT, BLUE)
    label(ax, 163.4, 28.7, r"$\times\,r$", 8.3, BLUE)

    path(ax, (pair, upper), (pair, lower), (net_left, lower), color=BROWN, width=1.15)
    path(ax, (125.5, lower), (125.5, 38.5), (132.4, 38.5), (132.4, 32.8), color=BROWN)
    tap(ax, 125.5, lower, BROWN)
    label(ax, 120.0, 41.35, r"$\boldsymbol{\psi}$", 9.2, BROWN)
    tile(ax, net_left, 40.8, net_right-net_left, 8.4, LILAC, "Pulse Gate", "MLP", subsize=7.85)
    path(ax, (net_right, lower), (subtract2, lower), (subtract2, upper+2), color=BROWN, width=1.15)
    label(ax, subtract2+2.6, 38.4, r"$\alpha$", 9.3, BROWN)
    return fig


def tint(color, target, fraction):
    return (1-fraction)*np.asarray(to_rgb(color)) + fraction*np.asarray(to_rgb(target))


def tensor(ax, cx, cy, width, height, color, segments=1):
    """Thin feature slice with a common face and depth direction at every resolution."""
    x, y = cx-width/2, cy-height/2
    dx, dy = 1.5, -.95
    ax.add_patch(Polygon([(x, y), (x+dx, y+dy), (x+width+dx, y+dy), (x+width, y)],
                         fc=tint(color, "white", .34), ec=EDGE, lw=.59, zorder=3))
    ax.add_patch(Polygon([(x+width, y), (x+width+dx, y+dy), (x+width+dx, y+height+dy),
                          (x+width, y+height)], fc=tint(color, EDGE, .15), ec=EDGE, lw=.59, zorder=3))
    ax.add_patch(Rectangle((x, y), width, height, fc=color, ec=EDGE, lw=.68, zorder=3))
    for seam in range(1, segments):
        sx = x+width*seam/segments
        ax.plot([sx, sx, sx+dx], [y+height, y, y+dy], color=EDGE, lw=.6, zorder=4)


def refiner():
    fig, ax = page(82)
    panel_title(ax, 2.4, 3.55, "a", "Four-level complex U-Net")
    stages = [(26, 13.2, 4.8, 9.5), (45, 21.1, 5.4, 7.5), (64, 28.4, 6.1, 6.0),
              (85, 35.4, 13.8, 4.7), (106, 28.4, 6.1, 6.0),
              (125, 21.1, 5.4, 7.5), (144, 13.2, 4.8, 9.5)]
    for i, (cx, cy, w, h) in enumerate(stages):
        fill = (OLIVE, OLIVE, OLIVE, CORAL, SKY, SKY, SKY)[i]
        tensor(ax, cx, cy, w, h, fill, 3 if i == 3 else 1)

    label(ax, 85, 6.4, "Skip connections", 8.0, SECONDARY)
    for i, dim in enumerate([r"$T\times9$", r"$T/2\times13$", r"$T/4\times17$"]):
        cx, cy, w, _ = stages[i]
        tx, _, tw, _ = stages[-1-i]
        path(ax, (cx+w/2+1.5, cy), (tx-tw/2, cy), color="#607587", width=.57, head_length=1.35)
        label(ax, 85, cy-2.05, dim, 8.05)
    for i, (a, b) in enumerate(zip(stages, stages[1:])):
        x, y, w, h = a
        xx, yy, ww, hh = b
        # The descending/ascending arrows use different ports from the lateral skips.
        start_y = y + (.30*h if i < 3 else -.20*h)
        end_y = yy + (-.19*hh if i < 3 else .24*hh)
        path(ax, (x+w/2+1.5, start_y), (xx-ww/2, end_y), width=.93)

    label(ax, 12.4, 9.1, r"$(\widehat{\boldsymbol{u}}_1,\widehat{\boldsymbol{j}}_1)/r$",
          MATH_WITH_SCRIPT)
    path(ax, (3.6, 13.2), (23.6, 13.2), width=1.05)
    label(ax, 32.9, 22.25, "↓2 AvgPool", 7.8, SECONDARY)
    label(ax, 138.5, 22.25, "↑2 Linear", 7.8, SECONDARY)
    label(ax, 74.5, 40.65, r"$T/8\times21$", 8.05)
    label(ax, 92.7, 40.65, "dilation = 1, 2, 4", 8.0, RED)
    label(ax, 12.4, 24.6, r"$\boldsymbol{\psi}$ gate (b)", 7.8, BROWN)
    path(ax, (19, 23.7), (25.4, 17.95), color=BROWN, width=.83)
    label(ax, 154.4, 24.6, r"$\boldsymbol{\psi}$ gate", 7.8, BROWN)
    path(ax, (149.7, 23.7), (145, 17.95), color=BROWN, width=.83)
    path(ax, (147.9, 13.2), (151.2, 13.2))
    tile(ax, 151.2, 10.1, 15.6, 6.2, YELLOW, "CplxConv", size=8.2)
    label(ax, 159, 19.0, "kernel = 1", 7.55, SECONDARY)
    path(ax, (166.8, 13.2), (172.5, 13.2))
    label(ax, 171.6, 9.3, r"$\widetilde{\boldsymbol{\delta}}$", 10.0)

    ax.plot([2.4, 173.4], [43.2, 43.2], color="#DAE0E4", lw=.47, zorder=0)
    panel_title(ax, 2.4, 46.3, "b", "Phase-equivariant complex residual block")
    top, mid, low = 53.4, 62.6, 73.4
    add, mult, fork = 153.6, 137.3, 17.6
    label(ax, 5.0, top, r"$\boldsymbol{z}$", 10.4)
    path(ax, (8.3, top), (add-2, top), width=1.35, head_length=1.9)
    label(ax, 97.5, 50.35, "Complex Equivariant Branch", 9.0, BLUE, weight="bold")
    operator(ax, add, top, "+")
    path(ax, (add+2, top), (166.7, top), width=1.35)
    label(ax, 171.2, top, r"$\boldsymbol{z}^{+}$", MATH_WITH_SCRIPT)
    path(ax, (fork, top), (fork, mid), (27, mid))
    tap(ax, fork, top)
    tile(ax, 27, mid-3.5, 29, 7, YELLOW, "Bias-Free CplxConv", size=8.75)
    label(ax, 41.5, 68.15, "kernel = 3, no bias", 7.6, SECONDARY)
    path(ax, (56, mid), (mult-2, mid), width=1.05)
    label(ax, 67.4, mid-2.5, r"$\boldsymbol{v}$", 9.5, BLUE)
    operator(ax, mult, mid, "×")
    path(ax, (mult+2, mid), (add, mid), (add, top+2), width=1.05)

    band_top, band_h = low-3.55, 7.1
    x = 61.2
    cells = [(24.0, SKY, r"$\log|\boldsymbol{v}|_{\epsilon}\,\Vert\,\boldsymbol{\psi}$", MATH_WITH_SCRIPT),
             (18.6, YELLOW, "RealConv", 8.55), (12.0, LILAC, "SiLU", 8.55),
             (15.1, CORAL, "Sigmoid", 8.55)]
    for w, color, name, size in cells:
        tile(ax, x, band_top, w, band_h, color, name, size=size)
        x += w
    path(ax, (74.4, mid), (74.4, band_top), color=BROWN, width=.87)
    tap(ax, 74.4, mid)
    label(ax, 5.0, low, r"$\boldsymbol{\psi}$", 10.0, BROWN)
    path(ax, (8.3, low), (61.2, low), color=BROWN, width=1.15)
    path(ax, (x, low), (mult, low), (mult, mid+2), color=BROWN, width=1.15)
    label(ax, mult+2.2, 67.7, r"$\boldsymbol{g}$", 9.5, BROWN, ha="left")
    label(ax, 142.1, 73.4, "modulus scaling only", 7.55, RED, ha="left")
    label(ax, 96, 79.6, "Phase-Invariant Real Control Branch", 9, BROWN, weight="bold")
    return fig


def plot_area(fig, x, y, w, h):
    h_mm = fig.get_figheight()*25.4
    return fig.add_axes((x/WIDTH, (h_mm-y-h)/h_mm, w/WIDTH, h/h_mm))


def tidy_axis(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=1.9, width=.5, pad=1.65, labelsize=7.25)


def mechanism(cache, audit_dir=None):
    fig, canvas = page(54)
    panel_title(canvas, 2.3, 3.35, "a", "Matched-filter evolution")
    panel_title(canvas, 90.5, 3.35, "b", "Ranked dictionary atoms and evidence")
    np.testing.assert_array_equal(cache["u1"], cache["y"]-cache["j1"])
    profiles = [np.abs(np.correlate(cache[key], cache["p"], mode="valid"))
                for key in ("y", "u1", "u")]
    reference = profiles[0].max()
    rows = []
    for i, (mag, stage, color) in enumerate(zip(profiles, ("Input", "Stage 1", "Stage 2"),
                                                ("#445764", BROWN, BLUE))):
        ax = plot_area(fig, 13.6, 9.6+i*12.2, 68.4, 9.65)
        rows.append(ax)
        lag = np.arange(mag.size)
        db = 20*np.log10(np.maximum(mag/reference, 1e-12))
        peak = mag.argmax()
        ax.plot(lag, np.maximum(db, -65), color=color, lw=.42, zorder=2)
        ax.plot(peak, db[peak], "o", ms=2.85, mew=.44, mec="white",
                mfc=RED if i < 2 else BLUE, clip_on=False, zorder=3)
        reading = "0 dB" if i == 0 else f"{db[peak]:.1f} dB".replace("-", "−")
        ax.text(.99, .96, f"{stage}   {reading}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=color, weight="bold", zorder=4)
        ax.set(xlim=(0, len(lag)-1), ylim=(-65, 0), xticks=[0, 300, 600, 900, 1200],
               yticks=[-60, -30, 0])
        tidy_axis(ax)
        ax.tick_params(axis="x", labelbottom=i == 2)
        print(f"{stage}: peak at {peak}, {db[peak]:.8f} dB")
    label(canvas, 2.6, 26.7, "Relative magnitude (dB)", 8.0, rotation=90, rotation_mode="anchor")
    label(canvas, 47.8, 50.8, "Delay lag (samples)", 8.1)

    start, stop = map(int, cache["window"])
    rate_mhz = cache["sample_rate_hz"].item()/1e6
    rho = cache["rho"]
    order = np.argsort(-rho, kind="stable")
    strengths = rho[order] / rho.max()
    mag = cache["dictionary_magnitude"]
    kernel = np.hanning(15)
    kernel /= kernel.sum()
    rms = np.stack([np.sqrt(np.convolve(row**2, kernel, "same")[start:stop])
                    for row in mag.T[order]])
    db = 20*np.log10(np.maximum(rms/mag.max(), 1e-12))
    ranks = np.arange(1, len(order)+1)
    edges = (np.arange(start, stop+1)-.5)/rate_mhz
    top, bottom = 11.1, 43.65
    heat = plot_area(fig, 101.2, top, 47.4, bottom-top)
    bars = plot_area(fig, 153.0, top, 19.1, bottom-top)
    cmap = mpl.colormaps["cividis"]
    # A vector cell per sample/atom: no raster embedding or interpolation between ranks.
    heat.pcolormesh(edges, np.arange(.5, len(order)+1), db, cmap=cmap,
                    vmin=-32, vmax=0, shading="flat", edgecolors="face",
                    linewidth=.065, antialiased=False)
    heat.hlines(ranks[:-1]+.5, edges[0], edges[-1], color="white", lw=.32)
    heat.set(xlim=(edges[0], edges[-1]), ylim=(ranks[-1]+.5, .5),
             xticks=[4, 6, 8, 10], yticks=[1, 4, 8, 12, 16])
    tidy_axis(heat)
    for spine in heat.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.4)
        spine.set_color("#536575")
    label(canvas, 91.7, 27.4, "Atom rank", 8.0, rotation=90, rotation_mode="anchor")
    label(canvas, 124.9, 50.8, "Fast time (μs)", 8.1)

    bars.barh(ranks, strengths, height=.7, color=BLUE, lw=0)
    bars.set(xlim=(0, 1), ylim=heat.get_ylim(), yticks=[], xticks=[0, .5, 1],
             xticklabels=["0", "0.5", "1"])
    tidy_axis(bars)
    bars.axvline(.5, ls=(0, (2, 2)), color=HAIR, lw=.4, zorder=0)
    label(canvas, 162.6, 50.8, r"$\rho_k/\max\,\rho$", MATH_WITH_SCRIPT)
    # Compact vector colour key; the endpoints refer to the full, unsmoothed dictionary.
    key_ax = plot_area(fig, 113.4, 7.75, 24.1, 1.48)
    key_ax.pcolormesh(np.arange(257), [0, 1], np.arange(256)[None], cmap=cmap,
                      shading="flat", edgecolors="face", linewidth=.06, antialiased=False)
    key_ax.set_axis_off()
    label(canvas, 111.8, 8.5, "−32", 7.2, ha="right")
    label(canvas, 139.0, 8.5, "0 dB", 7.2, ha="left")
    fig.canvas.draw()
    np.testing.assert_allclose(heat.get_position().bounds[1::2], bars.get_position().bounds[1::2])
    if audit_dir is not None:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        audit_dir.mkdir(parents=True, exist_ok=True)
        ids = ["input", "stage1", "stage2"]
        require_matplotlib_panel_alignment(fig, axes=rows, panel_ids=ids, column_groups=[ids],
                                          json_out=audit_dir/"alignment_profiles.json")
        require_matplotlib_panel_alignment(
            fig, axes=[heat, bars], panel_ids=["dictionary", "evidence"],
            row_groups=[["dictionary", "evidence"]],
            exemptions=[{"panels": ["evidence"], "checks": ["panel-width"],
                         "reason": "One scalar bar per dictionary row. Only row alignment is shared."}],
            json_out=audit_dir/"alignment_atoms.json")
    print("Atom IDs by displayed rank (one based):", (order+1).tolist())
    return fig


def export(fig, folder, stem):
    fig.savefig(folder/f"{stem}.pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(folder/f"{stem}.png", dpi=600)
    for output, mode in (("editable", "none"), ("outlined", "path")):
        with mpl.rc_context({"svg.fonttype": mode}):
            svg_path = folder/f"{stem}_{output}.svg"
            fig.savefig(svg_path, dpi=600, metadata={"Date": None})
        # Matplotlib declares an external DTD which the drawing never uses.
        markup = re.sub(r"<!DOCTYPE svg PUBLIC[^>]*>\s*", "", svg_path.read_text("utf-8"), count=1)
        if mode == "none":
            # SVG presentation attributes use the same page units as the viewBox.
            # Here one user unit is one print point, including in STIX subscripts.
            markup = re.sub(r'(<(?:text|tspan)\b[^>]*?)style="([^"]*?)font-size: ([\d.]+)px;?\s*([^"]*)"',
                            r'\1font-size="\3" style="\2\4"', markup)
        svg_path.write_text(markup, encoding="utf-8", newline="\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--audit-dir", type=Path,
                        help="Optional alignment receipts (audit_panel_alignment.py on PYTHONPATH)")
    args = parser.parse_args()
    if sha256(args.cache.read_bytes()).hexdigest() != CACHE_SHA:
        raise ValueError("Unexpected plotting cache SHA-256.")
    configure()
    args.out.mkdir(parents=True, exist_ok=True)
    export(overview(), args.out, "fig1_recovp_overview")
    export(refiner(), args.out, "fig2_complex_refiner")
    with np.load(args.cache, allow_pickle=False) as cached:
        export(mechanism(cached, args.audit_dir), args.out, "fig3_single_pulse_mechanism")


if __name__ == "__main__":
    main()

