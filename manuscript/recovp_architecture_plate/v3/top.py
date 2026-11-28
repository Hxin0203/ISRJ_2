"""Typeset the two overview panels in millimetres at final publication size."""
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch, FancyBboxPatch

INK = "#111827"
SIGNAL = "#1E3A5F"
REAL = "#2A7F83"
MODULE = {
    "analytic": ("white", "#334155"),
    "solve": ("#FDF6E3", "#8A6D1D"),
    "complex": ("#DCE8F2", "#3B5B78"),
    "real": ("#D8ECE9", "#2B6B68"),
    "gate": ("#F4DCD6", "#9A4B3E"),
}


def frame(ax, width):
    ax.set(xlim=(0, width), ylim=(85, 0))
    ax.axis("off")
    group = FancyBboxPatch((.7, 2.3), width-1.4, 79.4,
                           boxstyle="round,pad=0,rounding_size=1.4",
                           lw=.6, ec="#CBD5E1", fc="none", ls=(0,(3,3)), zorder=0)
    ax.add_patch(group)


def label(ax, x, y, text, size=7.5, color=INK, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            family="Arial", linespacing=1.10, zorder=5)


def block(ax, x, y, w, h, lines, kind="analytic"):
    fill, stroke = MODULE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=.88",
                 fc=fill, ec=stroke, lw=.75, zorder=3))
    ys = np.linspace(y+h/2-(len(lines)-1)*2.0, y+h/2+(len(lines)-1)*2.0, len(lines))
    for yc, (text, size) in zip(ys, lines):
        label(ax, x+w/2, yc, text, size)


def route(ax, *points, control=False, arrow=True):
    color, lw, style = (REAL, .9, (0,(3,2))) if control else (SIGNAL, 1, "-")
    x, y = zip(*points)
    ax.plot(x, y, c=color, lw=lw, ls=style, solid_capstyle="butt", zorder=1)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                     mutation_scale=7.2, lw=0, fc=color, ec=color,
                     shrinkA=0, shrinkB=0, zorder=2))


def node(ax, x, y, symbol):
    ax.add_patch(Circle((x,y), 2.25, fc="white", ec=SIGNAL, lw=.9, zorder=3))
    label(ax, x, y, symbol, 10, SIGNAL)


def wave(ax, data, x, y, scale):
    if data is None:
        return
    data = np.asarray(data).real.ravel()
    ax.plot(np.linspace(x, x+16, data.size), y - 3.9*data/scale, color=SIGNAL,
            lw=.30, solid_capstyle="butt", zorder=2)


def join(a, b, y):
    connector = ConnectionPatch((95, y), (0, y), a.transData, b.transData,
                                color=SIGNAL, lw=1, clip_on=False, zorder=1)
    a.figure.add_artist(connector)


def draw(ax_a, ax_b, observed, template, recovered=None):
    a,b = ax_a,ax_b
    frame(a,95)
    frame(b,78)
    peaks = [np.abs(np.real(v)).max() for v in (observed,recovered) if v is not None]
    scale = max(peaks + [np.finfo(float).tiny])
    wave(a, observed, 3.2, 13, scale)
    label(a, 11, 6.3, r"$\mathrm{Re}(\boldsymbol{y})$", 7.8)
    label(a, 11, 20, r"$\boldsymbol{y}\in\mathbb{C}^{1800}$", 7.1)
    route(a,(19.6,13),(85.75,13))
    route(a,(29.5,13),(29.5,22))
    block(a,19.5,22,20,12,[("RMS norm",7.7),(r"$\boldsymbol{x}=\boldsymbol{y}/r$",7.5)])
    route(a,(39.5,28),(49,28))
    block(a,49,21,33,14,[("Variable projection",7.6),
                         ("FP64 solve",6.9),
                         (r"$(\boldsymbol{B}^H\boldsymbol{B}+\boldsymbol{\Lambda})\widehat{\boldsymbol{c}}=\boldsymbol{B}^H\boldsymbol{x}$",7.1)],"solve")
    route(a,(82,28),(88,28),(88,15.25))
    label(a,75,18,r"$\widehat{\boldsymbol{j}}_1=r\boldsymbol{B}\widehat{\boldsymbol{c}}$",7.5)
    node(a,88,13,"−")
    route(a,(90.25,13),(95,13),arrow=False)
    route(a,(88,28),(95,28),arrow=False)
    for edge in (13,28):
        join(a,b,edge)

    if template is not None:
        wave(a,template,3.2,48,max(float(np.abs(np.real(template)).max()),1e-20))
        label(a,10.5,54,"Known template",6.9)
        label(a,10.5,59,r"$\boldsymbol{p}$",8)
    else:
        label(a,10.5,43,"Known template",7.1)
        label(a,10.5,48,r"$\boldsymbol{p}$",9.2)
    route(a,(19.5 if template is not None else 14.2,48),(23,48))
    route(a,(20.2,48),(20.2,79.6),(61,79.6),(61,77))
    route(a,(43.5,28),(43.5,40),(35,40),(35,43))
    block(a,23,43,24,10,[("Invariant features",7.5),(r"$\boldsymbol{\phi}$  (4 ch.)",7.0)])
    route(a,(35,53),(35,63),control=True)
    block(a,21.5,63,27,12,[("Shared 1D U-Net",7.5),(r"$\boldsymbol{h}$",7.4)],"real")
    route(a,(48.5,69),(55,69),control=True)
    block(a,55,56,31,21,[("Windowed dictionary",7.5),
          ("Matched filter, 16 peaks",6.9),
          (r"Shifted templates $\boldsymbol{t}_k$",6.9),
          (r"Learned windows $\boldsymbol{m}_k$",6.9),
          (r"$\boldsymbol{B}\in\mathbb{C}^{T\times16}$",7.4)])
    route(a,(59,56),(59,35))
    label(a,56,47,r"$\boldsymbol{B}$",8)
    block(a,66,41,23,11,[("Evidence gate",7.6),
                           (r"$\rho_k\;\to\;\lambda_k$",7.3)],"gate")
    route(a,(77.5,56),(77.5,52),control=True)
    route(a,(77.5,41),(77.5,35),control=True)

    route(b,(0,13),(48.75,13))
    label(b,6.3,9,r"$\widehat{\boldsymbol{u}}_1$",8.7)
    route(b,(14.5,13),(14.5,23))
    route(b,(0,28),(4,28))
    label(a,92,24,r"$\widehat{\boldsymbol{j}}_1$",7.5)
    block(b,4,23,23,12,[("Normalize",7.6),(r"$(\widehat{\boldsymbol{u}}_1,\widehat{\boldsymbol{j}}_1)/r$",7.5)])
    node(b,51,13,"−")
    route(b,(53.25,13),(58.6,13))
    if recovered is not None:
        wave(b,recovered,60,13,scale)
        label(b,68,6.4,r"$\mathrm{Re}(\widehat{\boldsymbol{u}})$",7.8)
        label(b,68,20,r"$\widehat{\boldsymbol{u}}\in\mathbb{C}^{1800}$",7.1)
    else:
        label(b,67,13,r"$\widehat{\boldsymbol{u}}\in\mathbb{C}^{1800}$",8.5)
    route(b,(27,29),(35,29),(35,47),(40,47))
    block(b,40,41,27,12,[("Complex U-Net",7.8)],"complex")
    route(b,(51,41),(51,32.25))
    label(b,54.9,37.6,r"$\widetilde{\boldsymbol{\delta}}$",8)
    node(b,51,30,"×")
    route(b,(51,27.75),(51,15.25))
    label(b,59.5,24,r"$\boldsymbol{\delta}_2=r\alpha\widetilde{\boldsymbol{\delta}}$",7.4)
    label(b,42.3,23,r"$r$ (RMS)",7.2,REAL)
    route(b,(42.3,25),(42.3,30),(48.75,30),control=True)
    route(b,(15.5,35),(15.5,48))
    block(b,4,48,25,11,[("Invariant features",7.6),(r"$\boldsymbol{\psi}$  (6 ch.)",7.3)])
    route(b,(29,53.5),(32.5,53.5),(32.5,50.5),(40,50.5),control=True)
    route(b,(15.5,59),(15.5,70),(34,70),control=True)
    block(b,34,64,30,12,[("Pulse gate MLP",7.7),("20 statistics",7.0)],"gate")
    route(b,(64,70),(72,70),(72,30),(53.25,30),control=True)
    label(b,68.5,58,r"$\alpha$",8.5,REAL)
    return {"dictionary":(79/95,1-77/85),"unet":(64.8/78,1-53/85)}

