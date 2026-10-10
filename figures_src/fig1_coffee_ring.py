"""Figure 1 (water.pdf): schematic of the coffee-ring effect.

a  top view of a droplet with particles uniformly distributed
b  the same droplet after evaporation; particles pushed to the rim
c  side view near the contact line: larger particles stop farther from it
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.lines import Line2D

import style as S

# ── Parameters ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = S.FULL_W, 52          # mm
SEED = 7                              # fixes particle positions
# (size class, radius relative to droplet radius, count)
PARTICLES = [("large", 0.13, 4), ("medium", 0.095, 7), ("small", 0.062, 11)]
# radial band each class ends up in after evaporation (fraction of radius);
# smaller particles get closer to the contact line
RIM_BANDS = {"large": (0.55, 0.70), "medium": (0.70, 0.80), "small": (0.83, 0.92)}
CONTACT_ANGLE = 30                    # degrees, drawn in panel c


def place(rng, specs, rmin, rmax_fn, tries=4000):
    """Rejection-sample non-overlapping circles inside an annulus."""
    placed = []
    for name, rad, n in specs:
        lo, hi = rmax_fn(name)
        for _ in range(n):
            for _ in range(tries):
                r = np.sqrt(rng.uniform(lo ** 2, hi ** 2))
                a = rng.uniform(0, 2 * np.pi)
                x, y = r * np.cos(a), r * np.sin(a)
                if r + rad > 0.97:
                    continue
                if all(np.hypot(x - px, y - py) > rad + pr + 0.012 for px, py, pr, _ in placed):
                    placed.append((x, y, rad, name))
                    break
    return placed


def droplet_top(ax, particles):
    ax.add_patch(Circle((0, 0), 1, color=S.COLORS["droplet"], lw=0))
    for x, y, r, name in particles:
        ax.add_patch(Circle((x, y), r, color=S.COLORS[f"particle_{name}"], lw=0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_axis_off()


def side_view(ax):
    """Spherical-cap liquid film over a substrate, contact line on the right."""
    ink = S.COLORS["ink"]
    L, xc = 10.0, 9.6                          # substrate length, contact line x
    th = np.radians(CONTACT_ANGLE)
    R = xc / np.sin(th)                        # cap radius giving that contact angle
    xs = np.linspace(0, xc, 400)
    h = np.sqrt(R ** 2 - xs ** 2) - np.sqrt(R ** 2 - xc ** 2)
    ax.fill_between(xs, 0, h, color=S.COLORS["droplet"], lw=0)
    ax.plot(xs, h, color=ink, lw=0.9)
    ax.plot([0, L], [0, 0], color=ink, lw=2.2, solid_capstyle="butt")
    # each particle stops where the film becomes thinner than its diameter
    for name, d, label in [("large", 1.9, "D$_3$"), ("medium", 1.35, "D$_2$"), ("small", 0.85, "D$_1$")]:
        x = xs[np.argmin(np.abs(h - d * 1.06))] - d / 2
        ax.add_patch(Circle((x, d / 2), d / 2, color=S.COLORS[f"particle_{name}"], lw=0))
        ax.add_patch(FancyArrowPatch((x, 0.06), (x, d - 0.06), arrowstyle="<|-|>",
                                     mutation_scale=5, color=ink, lw=0.7))
        ax.text(x - d / 2 - 0.15, d / 2, label, ha="right", va="center")
    # contact angle between the substrate and the tangent at the contact line
    t = 2.3
    ax.plot([xc, xc - t * np.cos(th)], [0, t * np.sin(th)], color=ink, lw=0.6, ls=(0, (2, 1.5)))
    ax.add_patch(Arc((xc, 0), 2.0, 2.0, theta1=180 - CONTACT_ANGLE, theta2=180, color=ink, lw=0.7))
    ax.text(xc - 1.35 * np.cos(th / 2), 1.35 * np.sin(th / 2), "θ", ha="center", va="center")
    top = h.max()
    ax.text(0, top + 0.9, "Near the contact line", ha="left", va="bottom")
    ax.text(L, top + 0.9, "D$_3$ > D$_2$ > D$_1$", ha="right", va="bottom")
    ax.set_xlim(0, L)
    ax.set_ylim(-0.3, top + 1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    S.apply()
    rng = np.random.default_rng(SEED)
    before = place(rng, PARTICLES, 0, lambda n: (0.0, 0.9))
    after = place(rng, PARTICLES, 0, lambda n: RIM_BANDS[n])

    fig = S.figure(WIDTH, HEIGHT)
    d = 44                                    # droplet panel size, mm
    ax_a = S.axes_mm(fig, 4, 5, d, d)
    ax_b = S.axes_mm(fig, 66, 5, d, d)
    ax_c = S.axes_mm(fig, 121, 5 + (d - 26) / 2, 57, 26)
    droplet_top(ax_a, before)
    droplet_top(ax_b, after)
    side_view(ax_c)

    # evaporation arrow between a and b
    fig.add_artist(FancyArrowPatch((0.29, 0.5), (0.36, 0.5), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=8, color=S.COLORS["ink"], lw=0.8))
    fig.text(0.325, 0.56, "Evaporation", ha="center", va="bottom")

    S.panel_label(ax_a, "a", dx=0)
    S.panel_label(ax_b, "b", dx=0)
    S.panel_label(ax_c, "c", dx=0, dy=0.5 + (d - 26) / 2)     # level with a and b
    S.save(fig, "water")


if __name__ == "__main__":
    main()
