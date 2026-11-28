"""Typeset the manuscript's annotated two-stage diagram in vector formats."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 768, 416
fig = plt.figure(figsize=(183 / 25.4, 183 / 25.4 * H / W))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
ax.axis("off")
ink, muted, red = "#21394a", "#4e6c7a", "#ad584a"
teal, blue, gold, pink = "#bdd8d6", "#c5d6e1", "#efdaa1", "#efcfc3"
scale = fig.get_figwidth() * 72 / W


def text(x, y, label, size=12, *, bold=False, color=ink, align="center"):
    ax.text(x, y, label, ha=align, va="center", fontsize=size * scale,
            color=color, weight="bold" if bold else "normal", zorder=5)


def box(x, y, w, h, fill, edge=ink, lw=.72, ls="-"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=lw, linestyle=ls))


def route(points, *, color=ink, lw=.85, arrow=True, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=(0, (4, 3)) if dashed else "-",
            solid_capstyle="butt", zorder=3)
    if arrow:
        x, y = points[-1]
        dx, dy = x-points[-2][0], y-points[-2][1]
        length = (dx*dx + dy*dy)**.5
        dx, dy = dx/length, dy/length
        ax.add_patch(Polygon([(x, y), (x-4*dx+2*dy, y-4*dy-2*dx),
                              (x-4*dx-2*dy, y-4*dy+2*dx)],
                             fc=color, ec="none", zorder=4))


def block(x, y, w, h, label, fill):
    box(x, y, w, h, fill)
    text(x+w/2, y+h/2+.5, label, 13)


def subtract(x, y):
    ax.add_patch(Circle((x, y), 8, fc="white", ec=ink, lw=.9, zorder=4))
    text(x, y-.1, "−", 19)


def junction(x, y):
    ax.add_patch(Circle((x, y), 2, fc=ink, ec="none", zorder=4))


# The first paper's two framed stages and single red magnifier.
for x, w, heading in [(63, 315, "(a)  Estimate the interference"),
                       (389, 312, "(b)  Refine the initial recovery")]:
    box(x, 13, w, 158, "#f7f9f8", "#92a8ae", .57)
    text(x+12, 29, heading, 16, bold=True, align="left")
    route([(x+12, 42), (x+w-12, 42)], color="#c8d3d5", lw=.45, arrow=False)

block(7, 55, 37, 32, r"$y$", "white")
block(720, 55, 41, 32, r"$\hat u$", "white")
text(26, 99, "Input I/Q", 11)
text(740, 99, "Output", 11)
route([(44, 71), (348, 71)], lw=1.18)
route([(364, 71), (664, 71)], lw=1.18)
route([(680, 71), (720, 71)], lw=1.18)
subtract(356, 71)
subtract(672, 71)
text(411, 57, r"$\hat u_1$", 16)
route([(87, 71), (87, 136), (104, 136)])
junction(87, 71)
block(104, 117, 82, 38, "MF-guided\ndictionary", teal)
block(238, 117, 85, 38, "Variable\nprojection", gold)
block(237, 84, 87, 23, "Evidence gate", pink)
route([(186, 136), (238, 136)])
route([(146, 117), (146, 95), (237, 95)], color="#437571", lw=.68, dashed=True)
route([(280, 107), (280, 117)], color="#437571", lw=.68, dashed=True)
route([(323, 136), (451, 136)])
route([(356, 136), (356, 79)])
junction(356, 136)
text(406, 124, r"$\hat j_1$", 16)
route([(419, 71), (419, 116), (451, 116)])
junction(419, 71)
block(451, 103, 84, 47, "Complex\nU-Net", blue)
box(447, 99, 92, 55, "none", red, .72, (0, (4, 3)))
route([(535, 126), (580, 126)])
text(558, 114, r"$\widetilde\delta$", 15)
block(580, 109, 62, 34, "Gate", pink)
route([(642, 126), (672, 126), (672, 79)])
text(655, 113, r"$\delta_2$", 15)

box(63, 196, 638, 207, "#fcfbf9", red, .70, (0, (4, 3)))
route([(511, 154), (511, 168), (546, 196)], color=red, lw=.60,
      dashed=True, arrow=False)
text(75, 213, "(c)  Inside the complex U-Net", 15, bold=True, align="left")
text(75, 231, "From the two stage-one estimates to a proposed correction", 11.8,
     align="left", color=muted)
text(245, 253, "Encoder", 13, bold=True)
text(461, 253, "Decoder", 13, bold=True)
text(354, 259, "Skip connections", 10.7, color=muted)
for y in (278, 316, 354):
    block(200, y-10, 90, 20, "Complex Res", teal)
    block(416, y-10, 90, 20, "Complex Res", blue)
    route([(290, y), (416, y)], color=muted, lw=.55)
for top in (278, 316):
    route([(245, top+10), (245, top+28)])
    route([(461, top+28), (461, top+10)])
    text(228, top+19, "↓2", 10.5, color=muted)
    text(478, top+19, "↑2", 10.5, color=muted)
route([(245, 364), (245, 382), (299, 382)])
route([(407, 382), (461, 382), (461, 364)])
text(228, 375, "↓2", 10.5, color=muted)
text(478, 375, "↑2", 10.5, color=muted)
block(299, 372, 108, 20, "Dilated bottleneck", gold)
text(128, 275, r"$(\hat u_1,\hat j_1)/r$", 16)
text(128, 295, "Normalized inputs", 11, color=muted)
route([(168, 278), (200, 278)])
route([(506, 278), (530, 278)])
block(530, 268, 81, 20, "Complex conv", blue)
route([(611, 278), (639, 278)])
text(652, 277, r"$\widetilde\delta$", 17)
text(651, 295, "Proposal", 11, color=muted)
base = Path(__file__).with_name("ReCoVP_Hybrid_architecture_detail_revised")
for ext in ("pdf", "svg", "png"):
    fig.savefig(base.with_suffix("."+ext), dpi=300, facecolor="white")
