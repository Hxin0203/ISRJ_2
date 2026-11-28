"""Draw the two network insets in a 178 mm architecture plate."""

from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#1E3A5F"
TEAL = "#2A7F83"
INK = "#111827"
COMPLEX = ("#DCE8F2", "#3B5B78")
REAL = ("#D8ECE9", "#2B6B68")
ANALYTIC = ("white", "#334155")
BOTTLENECK = ("#FDF6E3", "#8A6D1D")


def label(ax, x, y, value, size=7.5, **kwargs):
    ax.text(x, y, value, ha="center", va="center", fontsize=size,
            fontfamily="Arial", color=kwargs.pop("color", INK),
            linespacing=1.05, zorder=5, **kwargs)


def module(ax, x, y, w, h, value, palette):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
        facecolor=palette[0], edgecolor=palette[1], lw=.75, zorder=3))
    label(ax, x + w / 2, y + h / 2, value)


def route(ax, *points, control=False, arrow=True, linewidth=.9):
    color = TEAL if control else NAVY
    linestyle = (0, (3, 2)) if control else "solid"
    xs, ys = zip(*points)
    ax.plot(xs, ys, lw=linewidth, color=color, linestyle=linestyle,
            solid_capstyle="butt", solid_joinstyle="round", zorder=2)
    if arrow:
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=6.5,
            lw=0, color=color, shrinkA=0, shrinkB=0, zorder=4))


def residual(ax):
    y = .70
    module(ax, .115, y - .071, .352, .142, "Bias-free\ncomplex conv.", COMPLEX)
    label(ax, .026, y, r"$\boldsymbol{z}$", 8)
    route(ax, (.055, y), (.115, y))
    route(ax, (.467, y), (.61 - .0433, y))
    label(ax, .51, y + .052, r"$\boldsymbol{v}$", 8)
    for x, sign in ((.61, "×"), (.815, "+")):
        ax.add_patch(Ellipse((x, y), 4.5 / 52, 4.5 / 65,
                             facecolor="white", edgecolor=NAVY, lw=.9, zorder=3))
        label(ax, x, y, sign, 9)
    route(ax, (.6533, y), (.815 - .0433, y))
    route(ax, (.8583, y), (.925, y))
    label(ax, .964, y, r"$\boldsymbol{z}^{+}$", 8)
    route(ax, (.084, y), (.084, .89), (.815, .89), (.815, y + 2.25 / 65))

    module(ax, .13, .408, .39, .104,
           r"$\log|\boldsymbol{v}|_{\varepsilon}\oplus\boldsymbol{\psi}$", ANALYTIC)
    module(ax, .13, .174, .39, .135,
           "Real conv.\n" + r"SiLU $\rightarrow\sigma$", REAL)
    route(ax, (.50, y), (.50, .512), control=True)
    route(ax, (.325, .408), (.325, .309), control=True)
    route(ax, (.52, .2415), (.61, .2415), (.61, y - 2.25 / 65), control=True)
    label(ax, .795, .45, r"$g_i\in(0.43,1)$", 7.0, color=TEAL)


def unet(ax):
    left, right = .225, .775
    rows = [.735, .522, .309, .096]
    channels = [9, 13, 17, 21]
    widths = [.27 / (2 ** level) for level in range(4)]
    heights = [.043 * n / 9 for n in channels]
    labels = [r"$T\times9$", r"$T/2\times13$",
              r"$T/4\times17$", r"$T/8\times21$"]

    module(ax, right - .092, .858, .184, .078, r"$1\!\times\!1$", COMPLEX)
    route(ax, (right, rows[0] + heights[0] / 2), (right, .858))
    route(ax, (right + .092, .897), (.928, .897))
    label(ax, .963, .899, r"$\widetilde{\boldsymbol{\delta}}$", 8)
    label(ax, left, .815, "Encoder", 7)
    label(ax, right - .109, .815, "Decoder", 7)
    label(ax, .50, .815, "Skip", 6.8)
    # Outer rails carry invariant conditioning to matching network levels.
    for rail in (.052, .948):
        route(ax, (rail, rows[0]), (rail, rows[2]), control=True, arrow=False)
        label(ax, rail, .792, r"$\boldsymbol{\psi}$", 8, color=TEAL)

    for level, (y, w, h) in enumerate(zip(rows[:3], widths, heights)):
        for x in (left, right):
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor=COMPLEX[0], edgecolor=COMPLEX[1],
                                   lw=.75, zorder=3))
        route(ax, (.052, y), (left - w / 2, y), control=True, linewidth=.72)
        route(ax, (.948, y), (right + w / 2, y), control=True, linewidth=.72)
        route(ax, (left + w / 2, y), (right - w / 2, y), linewidth=.65)
        label(ax, .50, y + .035, labels[level], 6.9)
        if level < 2:
            next_top = rows[level + 1] + heights[level + 1] / 2
            route(ax, (left, y - h / 2), (left, next_top))
            route(ax, (right, next_top), (right, y - h / 2))
    label(ax, left + .125, .627, "AvgPool ↓2", 6.5)
    label(ax, right - .125, .627, "Linear ↑2", 6.5)

    y, w, h = rows[-1], widths[-1], heights[-1]
    ax.add_patch(Rectangle((.5 - w / 2, y - h / 2), w, h,
                           facecolor=BOTTLENECK[0], edgecolor=BOTTLENECK[1],
                           lw=.75, zorder=3))
    route(ax, (left, rows[2] - heights[2] / 2), (left, y), (.5 - w / 2, y))
    route(ax, (.5 + w / 2, y), (right, y), (right, rows[2] - heights[2] / 2))
    label(ax, .64, .186, labels[-1], 6.9)
    label(ax, .434, .20, r"$\boldsymbol{\psi}$", 8, color=TEAL)
    route(ax, (.462, .20), (.50, .20), (.50, y + h / 2), control=True)
    label(ax, .50, .021, r"Dilated $d=1,2,4$", 6.8)


def draw(ax_d, ax_e):
    for ax in (ax_d, ax_e):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
    residual(ax_d)
    unet(ax_e)
