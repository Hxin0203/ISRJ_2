# -*- coding: utf-8 -*-
"""ReCoVP-Hybrid 结构图 v2（2026-11-27）。风格对齐第一篇 Fig.2 (fig2_rsdr_net.png)。
相对草案：按 WP9 代码核对（evidence/outputs/paper2_baseline_suite_20261126/WP9/ARCH_FACTS.md）
修正全部标注；符号与草稿正文一致；(c) 画成三层复值 U-Net；以证据门控替代 K/N 预算说明。
用法：python make_fig2_recovp_hybrid_v2.py  -> 同目录输出 fig2_recovp_hybrid_v2.{png,pdf,svg}"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 15})
C = dict(panel="#F2F6F7", pedge="#8FA3A8", teal="#9FCBCB", lteal="#C9E1E1", slate="#8FAFC4",
         salmon="#E3ABA3", sand="#E8D298", line="#22313F", dash="#3A8A8A", grey="#5E6E78", red="#D46A5E")
OUT = os.path.dirname(os.path.abspath(__file__))
fig = plt.figure(figsize=(20, 20.5), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 200); ax.set_ylim(0, 205); ax.axis("off")
OFF = 0.0   # vertical offset applied by every helper (panels a and b are drawn with OFF=40)


def panel(x, y, w, h, ec=None, ls="-"):
    ax.add_patch(Rectangle((x, y + OFF), w, h, fc=C["panel"], ec=ec or C["pedge"], lw=1.3, ls=ls, zorder=0))


def box(x, y, w, h, txt, fc, bold=True, fs=16, sub=None):
    y = y + OFF
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=C["line"], lw=1.8, zorder=2))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=C["line"],
                fontweight="bold" if bold else "normal", zorder=3)
    else:
        ax.text(x + w / 2, y + h * 0.68, txt, ha="center", va="center", fontsize=fs, color=C["line"],
                fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 2, color=C["line"], zorder=3)


def arr(pts, dashed=False, head=True, lw=1.8):
    pts = [(px, py + OFF) for px, py in pts]
    col = C["dash"] if dashed else C["line"]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, ls=(0, (6, 4)) if dashed else "-", zorder=1.5)
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(
            arrowstyle="-|>,head_length=0.8,head_width=0.35", color=col, lw=lw, shrinkA=0, shrinkB=0), zorder=1.6)


def op(x, y, s, r=2.0):
    ax.add_patch(Circle((x, y + OFF), r, fc="white", ec=C["line"], lw=1.8, zorder=3))
    ax.text(x, y + OFF - 0.1, s, ha="center", va="center", fontsize=20, color=C["line"], zorder=4)


def dot(x, y):
    ax.add_patch(Circle((x, y + OFF), 0.55, fc=C["line"], ec="none", zorder=4))


def lab(x, y, s, col=None, fs=16, ha="center", **k):
    ax.text(x, y + OFF, s, ha=ha, va="center", fontsize=fs, color=col or C["line"], **k)


def rect(x, y, w, h, **k):
    ax.add_patch(Rectangle((x, y + OFF), w, h, **k))


def wave_icon(x, y, w, h, kind="in"):
    rect(x, y, w, h, fc="white", ec=C["line"], lw=1.8, zorder=2)
    t = np.linspace(0, 1, 900)
    for i, yy in enumerate([0.78, 0.5, 0.22]):
        base = np.cos(2 * np.pi * (8 * t + 30 * t ** 2))
        if kind == "in":
            env = 0.15 + 0.85 * sum((np.abs(t - c0) < 0.035 + 0.015 * i) for c0 in [0.15, 0.38, 0.61, 0.84])
        else:
            env = np.clip((t - 0.4) * 5, 0, 1) * 0.9 + 0.04
        ax.plot(x + 0.06 * w + 0.88 * w * t, y + OFF + h * (yy + 0.085 * env * base), color="#4F86A6", lw=0.8, zorder=3)


t = np.linspace(0, 1, 1200)


def strip(x0, y0, w, h, sig, col="#4F86A6"):
    rect(x0, y0, w, h, fc="white", ec=C["line"], lw=1.3, zorder=2)
    ax.plot(x0 + w * t, y0 + OFF + h / 2 + sig * h * 0.42, color=col, lw=0.9, zorder=3)


# =====================================================================  (a)
OFF = 40.0
lab(1.5, 161.5, "(a)  Overall architecture of ReCoVP-Hybrid", fs=22, ha="left", fontweight="bold")
yc = 107
wave_icon(1.5, yc - 9, 13, 18, "in")
lab(8, yc - 12, r"$\mathbf{y}\in\mathbb{C}^{T}$", fs=18)
lab(17, 155.5, r"$\bf{Stage\ 1}$   evidence-gated structured undo, closed-form coefficients", ha="left", fs=17)
panel(17, 98, 98, 54)
lab(66, 147.5, r"atoms without matched evidence are shrunk away:  $\lambda_k\uparrow$  as  $\rho_k\downarrow$",
    fs=15, col=C["dash"])
box(21, 128, 22, 13, "Candidate peaks", C["teal"], fs=15, sub=r"top-$K$ of $|\mathbf{x}\star\tilde{\mathbf{p}}|$")
box(50, 128, 26, 13, "Delay & window", C["slate"], fs=15, sub=r"$\tau_k=a_k+\Delta\tau_k,\ \mathbf{m}_k$")
box(83, 128, 27, 13, "Atom dictionary", C["lteal"], fs=15, sub=r"$\mathbf{B}\in\mathbb{C}^{T\times K}$")
arr([(43, 134.5), (50, 134.5)]); lab(46.5, 137.3, r"$\{a_k\}$", col=C["dash"], fs=14)
arr([(76, 134.5), (83, 134.5)]); lab(79.5, 137.3, r"$\{\tau_k,\mathbf{m}_k\}$", col=C["dash"], fs=13)
box(44, 111, 38, 13, "Evidence-gated solve", C["salmon"], fs=15,
    sub=r"$\hat{\mathbf{c}}=(\mathbf{B}^{H}\mathbf{B}+\mathbf{\Lambda}(\rho))^{-1}\mathbf{B}^{H}\mathbf{x}$")
op(100, 117.5, r"$\times$")
dot(32, yc); arr([(32, yc), (32, 128)])
dot(63, yc); arr([(63, yc), (63, 111)])
dot(100, 126); arr([(100, 128), (100, 119.5)])
arr([(100, 126), (86, 126), (86, 121), (82, 121)])
lab(103, 125, r"$\mathbf{B}$", fs=17)
arr([(82, 115), (98, 116.7)]); lab(90, 112.6, r"$\hat{\mathbf{c}}\in\mathbb{C}^{K}$", fs=14)
op(108, yc, r"$-$")
arr([(102, 117.5), (108, 117.5), (108, yc + 2)]); lab(111.3, 112.5, r"$\hat{\mathbf{j}}_1$", fs=17)
arr([(14.5, yc), (106, yc)])
lab(52, yc - 3.2, "identity path: the input is kept unless explained by an atom with evidence", col=C["grey"], fs=13)
box(117.5, yc - 4, 11, 8, r"$\hat{\mathbf{u}}_1$", fc="white", bold=False, fs=19)
arr([(110, yc), (117.5, yc)])
lab(123, yc - 7.5, r"$\hat{\mathbf{u}}_1=\mathbf{y}-\hat{\mathbf{j}}_1$", fs=14)

lab(132, 155.5, r"$\bf{Stage\ 2}$   equivariant refinement, scalar gate", ha="left", fs=17)
panel(132, 98, 56, 54)
box(137, 129, 46, 14, "Complex U-Net refiner", C["slate"], fs=15,
    sub=r"input $(\hat{\mathbf{u}}_1,\hat{\mathbf{j}}_1)/r$,  output $\tilde{\delta}$")
box(138, 111, 24, 12, r"Gate $\alpha$", C["sand"], fs=15, sub=r"$\alpha\in(0,1)$ per pulse")
dot(134, yc); arr([(134, yc), (134, 136), (137, 136)])
lab(135.6, 124, r"$\hat{\mathbf{u}}_1$", fs=14)
arr([(112, 117.5), (112, 146), (135, 146), (135, 140), (137, 140)], dashed=True)
lab(123.5, 148.2, r"$\hat{\mathbf{j}}_1$", col=C["dash"], fs=14)
dot(150, 129); arr([(150, 129), (150, 123)], dashed=True)
op(170, 117, r"$\times$"); op(178, yc, r"$-$")
arr([(170, 129), (170, 119)]); lab(172.6, 125, r"$\tilde{\delta}$", fs=17)
arr([(162, 117), (168, 117)]); lab(165, 119.4, r"$\alpha$", fs=16)
arr([(172, 117), (178, 117), (178, yc + 2)])
arr([(128.5, yc), (176, yc)])
lab(160, 101.5, r"$\hat{\mathbf{u}}=\hat{\mathbf{u}}_1-r\,\alpha\,\tilde{\delta}$", fs=16)
wave_icon(189, yc - 9, 10, 18, "out"); arr([(180, yc), (189, yc)])
lab(194, yc - 12, r"$\hat{\mathbf{u}}$", fs=19)
ax.plot([1.5, 198.5], [92.5 + OFF, 92.5 + OFF], color="#C9D3D6", lw=1.2)

# =====================================================================  (b)
lab(1.5, 88, "(b)  Windowed delayed-LFM atoms and evidence-gated coefficients (Stage 1)", fs=22, ha="left",
    fontweight="bold")
panel(4, 46, 116, 37)
lfm = np.cos(2 * np.pi * (30 * t + 120 * t ** 2)); taus = [0.22, 0.47, 0.73]
obs = 0.22 * np.cos(2 * np.pi * (5 * t + 80 * t ** 2)) + sum((np.abs(t - tk) < 0.05) * lfm for tk in taus) * 0.9
obs = obs / np.max(np.abs(obs))
strip(8, 64, 30, 12, obs)
lab(23, 78.5, r"observation $\mathbf{x}=\mathbf{y}/r$ with ISRJ slices", fs=14)
for tk in taus:
    rect(8 + 30 * (tk - 0.06), 64, 30 * 0.12, 12, fc="none", ec=C["red"], lw=1.4, ls=(0, (4, 3)), zorder=4)
lab(23, 60.5, r"matched-filter peaks $\{a_k\}$, $K=16$", col=C["red"], fs=14)
arr([(38, 70), (46, 70)])
for i, tk in enumerate(taus):
    yy = 74 - i * 9.5
    strip(46, yy, 30, 7.5, np.exp(-0.5 * ((t - tk) / 0.03) ** 2) * lfm, col="#2E6F80")
    lab(78, yy + 3.7, rf"$\mathbf{{b}}_{{{i + 1}}}\propto\mathbf{{m}}_{{{i + 1}}}\odot\mathbf{{t}}(\tau_{{{i + 1}}})$",
        fs=14, ha="left")
lab(61, 49.5, r"$\vdots$  up to $K=16$ unit-norm atoms: delayed LFM copies under learned windows", fs=13)
arr([(99, 64.5), (104, 64.5)])
box(104, 57, 8, 15, r"$\mathbf{B}$", C["lteal"], fs=20)
panel(124, 46, 74, 37)
lab(161, 80.5, r"$\bf{Evidence\ gate}$", fs=17)
lab(161, 75, r"$\rho_k=|\mathbf{b}_k^{H}\mathbf{x}|^2/\hat{\sigma}^2$   (matched energy vs. robust noise floor)", fs=14)
lab(161, 69.5, r"$\lambda_k=\Lambda(v_k+2\ln[9/(1+\rho_k)])$,   $\mathbf{\Lambda}=\mathrm{diag}(\lambda_k)$", fs=14)
lab(161, 64, r"$\hat{\mathbf{c}}=\arg\min_{\mathbf{c}}\ \|\mathbf{x}-\mathbf{B}\mathbf{c}\|^2+\mathbf{c}^{H}\mathbf{\Lambda}\mathbf{c}$", fs=14)
lab(161, 58.5, r"noise only:  small $\rho_k$ $\Rightarrow$ huge $\lambda_k$ $\Rightarrow$ $\hat{c}_k\approx0$ (background kept)",
    fs=14, col=C["dash"])
lab(161, 53.5, r"ISRJ slice:  large $\rho_k$ $\Rightarrow$ small $\lambda_k$ $\Rightarrow$ near-exact undo", fs=14,
    col=C["dash"])
lab(161, 48.5, r"$\|\hat{\mathbf{j}}_1\|^2/\|\mathbf{y}\|^2\leq K\hat{\sigma}^2\,(\sum\rho_k)/(T\lambda_{\min}^2)$"
               "   (no independence assumed)", fs=13, col=C["grey"])
arr([(112, 64.5), (124, 64.5)])

# =====================================================================  (c)
OFF = 0.0
lab(1.5, 79, "(c)  Phase-equivariant complex U-Net refiner and scalar gate (Stage 2)", fs=22, ha="left",
    fontweight="bold")
panel(4, 1, 194, 74, ec=C["red"], ls=(0, (6, 4)))
ytop, y1, y2 = 56, 38, 20
box(7, ytop, 14, 10, r"$(\hat{\mathbf{u}}_1,\hat{\mathbf{j}}_1)/r$", fc="white", bold=False, fs=15)
lab(26, ytop - 2.6, "2 complex ch.", col=C["grey"], fs=12)
box(25, ytop, 13, 10, "Stem", C["slate"], fs=15, sub="9 ch")
box(42, ytop, 16, 10, "Enc 0", C["slate"], fs=15, sub="9 ch, res")
box(42, y1, 16, 10, "Enc 1", C["slate"], fs=15, sub="13 ch")
box(42, y2, 16, 10, "Enc 2", C["slate"], fs=15, sub="17 ch")
box(64, 8, 40, 9, r"Bottleneck $\times3$", C["slate"], fs=15, sub="21 ch, dilation 1/2/4, res")
box(110, y2, 16, 10, "Dec 2", C["teal"], fs=15, sub="17 ch")
box(110, y1, 16, 10, "Dec 1", C["teal"], fs=15, sub="13 ch")
box(110, ytop, 16, 10, "Dec 0", C["teal"], fs=15, sub="9 ch")
box(130, ytop, 12, 10, "Out", C["lteal"], fs=15, sub=r"$1\times1$ CConv")
arr([(21, ytop + 5), (25, ytop + 5)]); arr([(38, ytop + 5), (42, ytop + 5)])
arr([(50, ytop), (50, y1 + 10)]); lab(53.5, (ytop + y1 + 10) / 2, r"$\downarrow2$", fs=15)
arr([(50, y1), (50, y2 + 10)]); lab(53.5, (y1 + y2 + 10) / 2, r"$\downarrow2$", fs=15)
arr([(50, y2), (50, 12.5), (64, 12.5)])
arr([(104, 12.5), (118, 12.5), (118, y2)]); lab(121.5, 16.3, r"$\uparrow2$", fs=15)
arr([(118, y2 + 10), (118, y1)]); lab(121.5, (y2 + 10 + y1) / 2, r"$\uparrow2$", fs=15)
arr([(118, y1 + 10), (118, ytop)]); lab(121.5, (y1 + 10 + ytop) / 2, r"$\uparrow2$", fs=15)
for yy in (ytop, y1, y2):
    arr([(58, yy + 5), (110, yy + 5)])
    lab(84, yy + 7, "skip (concat)", col=C["grey"], fs=12)
arr([(126, ytop + 5), (130, ytop + 5)])
box(128, 30, 20, 13, r"Gate $\alpha$", C["sand"], fs=15, sub=r"$\sigma(\mathrm{MLP}_{20\to4\to1})$")
dot(14, ytop); arr([(14, ytop), (14, 3), (138, 3), (138, 30)], dashed=True)
lab(76, 5.4, r"invariant statistics (mean, std, p95 of 6 features) + energy ratio + MF peak-to-mean",
    col=C["dash"], fs=12)
op(148, ytop + 5, r"$\times$"); op(158, ytop + 5, r"$-$")
arr([(142, ytop + 5), (146, ytop + 5)]); lab(144, ytop + 7.6, r"$\tilde{\delta}$", fs=15)
arr([(138, 43), (138, 50), (148, 50), (148, ytop + 3)]); lab(150.5, 47.5, r"$\alpha$", fs=15)
arr([(150, ytop + 5), (156, ytop + 5)])
dot(14, ytop + 10); arr([(14, ytop + 10), (14, 71), (158, 71), (158, ytop + 7)])
lab(86, 72.8, r"$\hat{\mathbf{u}}_1$  (identity path)", col=C["grey"], fs=13)
arr([(160, ytop + 5), (166, ytop + 5)]); lab(170, ytop + 5, r"$\hat{\mathbf{u}}$", fs=19)
lab(175, 40, r"each block:  $\mathbf{z}'=\sigma(\mathrm{Conv}(\log|\mathbf{W}\mathbf{z}|,\ \mathrm{inv}))\odot\mathbf{W}\mathbf{z}$",
    fs=13)
lab(175, 34, r"$\mathbf{W}$: bias-free complex convolution", fs=13, col=C["grey"])
lab(175, 26, r"$\mathcal{R}(c\,\mathbf{z})=c\,\mathcal{R}(\mathbf{z})$  for any $c\in\mathbb{C}\setminus\{0\}$",
    fs=14)
lab(175, 18, r"$\hat{\mathbf{u}}=\hat{\mathbf{u}}_1-r\,\alpha\,\tilde{\delta}$,  one scalar $\alpha$ per pulse", fs=14)

for ext in ("png", "pdf", "svg"):
    fig.savefig(os.path.join(OUT, f"fig2_recovp_hybrid_v2.{ext}"), dpi=200 if ext == "png" else None,
                facecolor="white")
print("saved to", OUT)
