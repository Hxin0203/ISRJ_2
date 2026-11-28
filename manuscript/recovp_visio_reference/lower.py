"""Native Visio artwork for the four training objectives."""
from math import exp, sin


def draw(g):
    border = "#A4B4B7"
    teal = "#478181"
    ink = "#263D4B"
    soft = "#CED7D6"
    g.rect(8, 508, 1184, 138, "#FAF6ED", border)
    g.text(20, 514, 750, 26, "(c)  Physics-informed training objectives",
           size=15, bold=True, align="left")
    g.text(1057, 516, 119, 20, "Training only", size=10)
    g.line([(20, 544), (1179, 544)], stroke=border, lw=0.6, arrow=False)
    for x in (299, 588, 903):
        g.line([(x, 554), (x, 632)], stroke=soft, lw=0.65, arrow=False)

    def mini_chart(x, mode):
        g.rect(x, 566, 80, 49, "#FFFFFF", border, lw=0.65)
        axis = 595 if mode == "wave" else 607
        g.line([(x + 5, axis), (x + 75, axis)], stroke=soft, lw=0.45, arrow=False)
        points = []
        for i in range(45):
            t = (i - 22) / 22
            if mode == "wave":
                v = 14 * exp(-2.6 * t * t) * sin(20 * t)
            elif mode == "pulse":
                v = 27 * exp(-45 * t * t) + 4 * abs(sin(11 * t)) * exp(-3 * t*t)
            else:
                v = 3 + 5 * abs(sin(7 * t)) + 18 * exp(-100 * (t-0.48)**2)
            points.append((x+5+70*i/44, axis-v))
        g.line(points, stroke=teal, lw=1.15, arrow=False)

    mini_chart(27, "wave")
    mini_chart(316, "pulse")
    mini_chart(918, "tail")
    for x, width, heading, subtitle in (
        (115, 173, "Waveform recovery", "Time-domain error"),
        (403, 177, "Pulse compression", "Matched-filter error"),
        (699, 198, "Target damage", "Four paired states"),
        (1006, 179, "Residual tail", "Range–Doppler proxy"),
    ):
        g.text(x, 569, width, 23, heading, size=11.5, bold=True, align="left")
        g.text(x, 594, width, 21, subtitle, size=9.5, align="left")
    for k, (state, fill) in enumerate((("y₀₀", "#FFFFFF"), ("y₁₀", "#D8EAE5"),
                                       ("y₀₁", "#EEDFC0"), ("y₁₁", "#D9E5EB"))):
        x = 606 + 42 * (k % 2)
        y = 565 + 26 * (k // 2)
        g.rect(x, y, 39, 24, fill, border, lw=0.6)
        g.text(x, y, 39, 24, state, size=10.5)
