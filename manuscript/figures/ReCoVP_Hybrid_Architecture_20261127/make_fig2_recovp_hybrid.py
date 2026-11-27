"""Code-audited ReCoVP-Hybrid architecture. Run with Python + matplotlib.

SVG retains editable text; PDF embeds TrueType fonts. No measured data are used.
Source: tools/vp_1103_recovp_hybrid.py; provenance is recorded in ARCH_FACTS.md.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
C = dict(ink="#22313F", panel="#F2F6F7", edge="#8FA3A8", teal="#C9E1E1",
         blue="#B5CCDC", coral="#E3ABA3", gold="#E8D298", muted="#526873")
fig, ax = plt.subplots(figsize=(16, 10.8))
fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
ax.set(xlim=(0, 160), ylim=(0, 108)); ax.axis("off")

def label(x, y, text, size=11, ha="center", weight="normal", color=None):
    return ax.text(x, y, text, ha=ha, va="center", fontsize=size,
                   weight=weight, color=color or C["ink"], zorder=5)

def box(x, y, w, h, text, color="teal", size=11):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=C.get(color, color),
                           edgecolor=C["ink"], lw=1.1, zorder=3))
    label(x+w/2, y+h/2, text, size)

def panel(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=C["panel"],
                           edgecolor=C["edge"], lw=.8, zorder=0))

def arrow(points, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C["ink"], lw=1.1,
                ls="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color=C["ink"],
                                 linestyle="--" if dashed else "-", zorder=2,
                                 shrinkA=0, shrinkB=0))

label(2, 105, "(a)  Two-stage signal restoration", 15, "left", "bold")
panel(11, 78, 74, 22); panel(89, 78, 58, 22)
label(14, 97, "Stage 1 · structured correction", 12, "left", "bold")
label(92, 97, "Stage 2 · gated refinement", 12, "left", "bold")
label(4, 88, r"$y$", 17)
box(15, 84, 29, 9, "Learned dictionary\n+ analytic coefficients", size=10)
box(49, 84, 16, 9, r"$\hat{j}=s\Phi\hat{c}$", "coral", 13)
box(71, 84, 10, 9, r"$y-\hat{j}$", "white", 13)
arrow([(7, 88.5), (15, 88.5)]); arrow([(44, 88.5), (49, 88.5)])
arrow([(65, 88.5), (71, 88.5)])
arrow([(9, 88.5), (9, 80), (76, 80), (76, 84)])
box(93, 84, 22, 9, "Complex U-Net\n+ scalar gate", "blue", 10)
box(122, 84, 21, 9, r"$\hat{x}=u-\Delta$", "white", 13)
arrow([(81, 88.5), (93, 88.5)]); label(87, 90.5, r"$u$", 12)
arrow([(57, 84), (57, 76), (104, 76), (104, 84)])
label(84, 75, r"$\hat{j}$", 12)
arrow([(115, 88.5), (122, 88.5)]); label(118.5, 91, r"$\Delta$", 12)
arrow([(87, 88.5), (87, 81), (132.5, 81), (132.5, 84)])
arrow([(143, 88.5), (153, 88.5)]); label(156, 88.5, r"$\hat{x}$", 17)

label(2, 71.5, "(b)  Stage 1: candidate selection, learned atoms and regularized solve", 15, "left", "bold")
panel(2, 42, 156, 26)
box(5, 53, 25, 10, "Rect-MF peaks\nK = 16; spacing ≥ 6", size=10)
box(35, 53, 27, 10, "Invariant features\nDelay + window heads", "blue", 10)
box(68, 53, 26, 10, "Delayed LFM × mask\nColumn normalization", size=10)
box(101, 53, 30, 10, "Diagonal-precision\nanalytic solve", "coral", 10)
box(138, 53, 16, 10, r"$\hat{j}=s\Phi\hat{c}$", "white", 12)
for x1, x2 in [(30,35),(62,68),(94,101),(131,138)]: arrow([(x1,58),(x2,58)])
label(17.5, 48, r"$\tau_k^{(0)}$", 12)
label(48.5, 48, r"$\tau_k=\tau_k^{(0)}+2\tanh(q_k)$", 11)
label(81, 48, r"$\phi_k=\frac{w_k\odot\ell_{\tau_k}}{\|w_k\odot\ell_{\tau_k}\|_2}$", 13)
label(125, 46.8, r"$\hat{c}=(\Phi^H\Phi+\mathrm{diag}(p))^{-1}\Phi^H(y/s)$", 12)
label(3, 39.5, r"$s=\mathrm{RMS}(y)$;  $p_k>0$ is learned from invariant context.   Correction rank ≤ 16 for each constructed dictionary.", 10, "left", color=C["muted"])

label(2, 35.5, "(c)  Stage 2: complex U-Net with invariant gates", 15, "left", "bold")
panel(2, 3, 156, 29)
box(5, 20, 19, 8, r"$[u/s,\;\hat{j}/s]$"+"\n2 complex channels", "white", 10)
box(29, 20, 16, 8, "Stem + E0\n9", "blue", 10)
box(50, 16, 14, 8, "E1\n13", "blue", 10)
box(69, 12, 14, 8, "E2\n17", "blue", 10)
box(88, 8, 20, 8, "Bottleneck · 21\ndilation 1 / 2 / 4", "blue", 9)
box(113, 12, 11, 8, "D2\n17", "blue", 10)
box(129, 16, 11, 8, "D1\n13", "blue", 10)
box(145, 20, 10, 8, "D0\n9", "blue", 10)
arrow([(24,24),(29,24)])
arrow([(45,24),(47.5,24),(47.5,20),(50,20)])
arrow([(64,20),(66.5,20),(66.5,16),(69,16)])
arrow([(83,16),(85.5,16),(85.5,12),(88,12)])
arrow([(108,12),(110.5,12),(110.5,16),(113,16)])
arrow([(124,16),(126.5,16),(126.5,20),(129,20)])
arrow([(140,20),(142.5,20),(142.5,24),(145,24)])
# Skip concatenations at matching resolutions, with dedicated routing lanes.
arrow([(37,28),(37,31),(150,31),(150,28)], True)
arrow([(57,24),(57,29),(134.5,29),(134.5,24)], True)
arrow([(76,20),(76,26),(118.5,26),(118.5,20)], True)
box(5, 5, 34, 10, "6 invariant features × 3 statistics\n+ 2 physical scalars → MLP 20–4–1", "gold", 9)
arrow([(14.5,20),(14.5,15)])
box(44, 5, 20, 10, r"$\alpha\in[0,1]$"+"\nOne scalar / pulse", "gold", 10)
arrow([(39,10),(44,10)])
box(113, 4, 42, 6, r"$d=\mathrm{CConv}_{1\times1}(D0),\quad \Delta=\alpha s d$", "white", 10)
arrow([(64,10),(66,10),(66,5),(113,5)])
arrow([(150,20),(150,10)])
label(80, .6, r"$F(ay)=aF(y),\ a\in\mathbb{C}^{*}$   |   Effective real parameters: Stage 1  28,948  +  Stage 2  28,577  =  57,525", 11)

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fig2_recovp_hybrid_FINAL.{ext}", dpi=200, facecolor="white")
# Matplotlib adds trailing spaces inside multiline path attributes.
svg_path = OUT / "fig2_recovp_hybrid_FINAL.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
print(OUT)
