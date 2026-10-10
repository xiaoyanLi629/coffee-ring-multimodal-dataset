"""Figure 2 (zutu.pdf): experimental setup and imaging of one deposit.

a  autosampler drawing (asset) with part labels
b  environmental chamber photograph (asset)
c  original optical image      d  adjusted optical image
e  original SEM image          f  adjusted SEM image with magnified regions
g  workflow: sampling, adding liquid, evaporation, deposition
"""
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D

import style as S

# ── Parameters ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = S.FULL_W, 184                     # mm
CONDITION, DROPLET = "E", 2                       # deposit shown in c–f
OPTICAL_ORIGINAL = S.DATA / "optical" / CONDITION / f"Image_{DROPLET}.jpg"
OPTICAL_ADJUSTED = S.DATA / "corrected_images" / CONDITION / f"Image_{DROPLET}.jpg"
SEM_ORIGINAL = S.DATA / "sem" / CONDITION / f"SEM_{DROPLET}.jpg"
SEM_ADJUSTED = S.DATA / "corrected_images" / CONDITION / f"SEM_{DROPLET}.jpg"
# crops (left, top, right, bottom, fractions removed) giving 4:3 views of the deposit
OPTICAL_CROP = (0.073, 0.10, 0.073, 0.10)
SEM_CROP = (0.0, 0.0, 0.0, 0.0)

AUTOSAMPLER = S.ASSETS / "autosampler_drawing.png"
AUTOSAMPLER_CROP = (0.17, 0.03, 0.18, 0.11)
CHAMBER = S.ASSETS / "environmental_chamber.jpg"
CHAMBER_CROP = (0.089, 0.0, 0.089, 0.0)

# autosampler labels: (text, target x, target y as fractions of the uncropped
# drawing, side of the panel the label sits on)
AUTOSAMPLER_LABELS = [
    ("Z-axis motor", 0.576, 0.087, "right"),
    ("E-axis motor", 0.538, 0.238, "right"),
    ("Y-axis motor", 0.596, 0.529, "right"),
    ("Sample holder", 0.676, 0.691, "right"),
    ("X-axis motor", 0.223, 0.270, "left"),
    ("Sample collector", 0.514, 0.379, "left"),
    ("Collection plate", 0.345, 0.614, "left"),
    ("Sampler base", 0.245, 0.756, "left"),
]

# magnified regions on the adjusted SEM image, in SEM pixels:
# (source centre x, y, source radius, inset centre x, y, inset radius)
MAGNIFIERS = [
    (402, 684, 117, 300, 430, 285),
    (1613, 366, 125, 2255, 430, 285),
    (631, 1302, 111, 300, 1600, 285),
    (1309, 947, 50, 2160, 1590, 285),
]
SCALE_UM_PER_PX = {"optical": None, "sem": None}   # set to draw scale bars
WORKFLOW = ["Sampling", "Adding liquid", "Water evaporation", "Solid deposition"]


# ── Panels ───────────────────────────────────────────────────────────────────
def show(ax, img, tag_left=None, tag_right=None, color="white"):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="none")
    S.image_axes(ax)
    kw = dict(transform=ax.transAxes, color=color, va="top", fontsize=S.FONT_SIZE)
    if tag_left:
        ax.text(0.03, 0.97, tag_left, ha="left", **kw)
    if tag_right:
        ax.text(0.97, 0.97, tag_right, ha="right", **kw)


def autosampler(ax):
    img = S.load_image(AUTOSAMPLER, crop=AUTOSAMPLER_CROP)
    h, w = img.shape[:2]
    full_w, full_h = w / (1 - AUTOSAMPLER_CROP[0] - AUTOSAMPLER_CROP[2]), h / (1 - AUTOSAMPLER_CROP[1] - AUTOSAMPLER_CROP[3])
    ax.imshow(img, interpolation="antialiased", extent=(0, w, h, 0))
    ax.set_xlim(-0.72 * w, 1.62 * w)
    ax.set_aspect("equal", adjustable="datalim")       # keep the panel box, widen limits
    ax.set_ylim(h * 1.02, -0.02 * h)
    ax.set_axis_off()
    left = [l for l in AUTOSAMPLER_LABELS if l[3] == "left"]
    right = [l for l in AUTOSAMPLER_LABELS if l[3] == "right"]
    for group, xtext, ha in [(left, -0.06 * w, "right"), (right, 1.06 * w, "left")]:
        for text, fx, fy, _ in group:
            tx = fx * full_w - AUTOSAMPLER_CROP[0] * full_w
            ty = fy * full_h - AUTOSAMPLER_CROP[1] * full_h
            ax.plot([xtext, tx], [ty, ty], color=S.COLORS["ink"], lw=0.4)
            ax.plot(tx, ty, "o", ms=1.2, color=S.COLORS["ink"])
            ax.text(xtext + (-0.02 if ha == "right" else 0.02) * w, ty, text, ha=ha, va="center",
                    fontsize=S.FONT_SIZE - 1)


def sem_with_magnifiers(ax, img):
    ax.imshow(img, cmap="gray", interpolation="none")
    S.image_axes(ax)
    ring = S.COLORS["highlight"]
    for sx, sy, sr, ix, iy, ir in MAGNIFIERS:
        ax.add_patch(Circle((sx, sy), sr, fill=False, ec=ring, lw=0.8))
        # outer tangent lines between source circle and inset circle
        dx, dy = ix - sx, iy - sy
        dist = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        phi = np.arccos(np.clip((sr - ir) / dist, -1, 1))
        for a in (ang + phi, ang - phi):
            ax.plot([sx + sr * np.cos(a), ix + ir * np.cos(a)],
                    [sy + sr * np.sin(a), iy + ir * np.sin(a)], color=ring, lw=0.6)
        crop = img[int(sy - sr):int(sy + sr), int(sx - sr):int(sx + sr)]
        im = ax.imshow(crop, cmap="gray", interpolation="antialiased",
                       extent=(ix - ir, ix + ir, iy + ir, iy - ir), zorder=3)
        clip = Circle((ix, iy), ir, transform=ax.transData)
        im.set_clip_path(clip)
        ax.add_patch(Circle((ix, iy), ir, fill=False, ec=ring, lw=1.0, zorder=4))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)


def workflow_step(ax, step):
    ink, liquid, vap = S.COLORS["ink"], S.COLORS["liquid"], S.COLORS["vapour"]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.2, 0.2), 9.6, 9.6, fill=False, ec=ink, lw=0.8))

    def syringe(x, tip_y):
        ax.add_patch(Rectangle((x - 0.9, 7.2), 1.8, 2.3, fc="white", ec=ink, lw=0.7))
        ax.add_patch(Rectangle((x - 0.9, 7.2), 1.8, 1.0, fc=liquid, ec="none"))
        ax.add_patch(Rectangle((x - 0.25, 9.5), 0.5, 0.3, fc="white", ec=ink, lw=0.7))
        ax.plot([x, x], [7.2, tip_y], color=ink, lw=0.7)
        ax.plot([1.0, x - 1.3], [8.0, 8.0], color=ink, lw=0.7)
        ax.plot([x + 1.3, 9.0], [8.0, 8.0], color=ink, lw=0.7)

    if step == 0:                                   # sampling from tubes
        for x in (2.3, 5.0, 7.7):
            ax.add_patch(FancyBboxPatch((x - 0.75, 1.3), 1.5, 3.6, boxstyle="round,pad=0,rounding_size=0.4",
                                        fc="white", ec=ink, lw=0.7))
            ax.add_patch(FancyBboxPatch((x - 0.75, 1.3), 1.5, 2.5, boxstyle="round,pad=0,rounding_size=0.4",
                                        fc=liquid, ec="none"))
        ax.plot([0.6, 9.4], [1.0, 1.0], color=ink, lw=0.7)
        syringe(5.0, 2.0)
    elif step == 1:                                 # dispensing onto a slide
        ax.add_patch(Rectangle((2.6, 1.2), 4.8, 1.1, fc="white", ec=ink, lw=0.7))
        ax.add_patch(Ellipse((5.0, 2.3), 1.3, 0.8, fc=liquid, ec="none"))
        syringe(5.0, 3.2)
    elif step == 2:                                 # evaporation with capillary flow
        ax.plot([0.8, 9.2], [2.5, 2.5], color=ink, lw=0.8)
        t = np.linspace(0, np.pi, 100)
        ax.fill_between(5 + 3.2 * np.cos(t), 2.5, 2.5 + 2.3 * np.sin(t), color=liquid, lw=0)
        ax.plot(5 + 3.2 * np.cos(t), 2.5 + 2.3 * np.sin(t), color=vap, lw=0.6)
        rng = np.random.default_rng(3)
        for _ in range(14):
            x = rng.uniform(2.4, 7.6)
            y = rng.uniform(2.7, 2.5 + 2.0 * np.sqrt(max(0, 1 - ((x - 5) / 3.2) ** 2)))
            ax.add_patch(Circle((x, y), rng.choice([0.12, 0.18, 0.24]), color=S.COLORS["particle_small"], lw=0))
        for x0, x1 in [(4.4, 2.6), (5.6, 7.4)]:
            ax.add_patch(FancyArrowPatch((x0, 3.9), (x1, 2.8), connectionstyle="arc3,rad=0.3",
                                         arrowstyle="-|>", mutation_scale=4, color=ink, lw=0.5))
        for a in np.linspace(0.25, np.pi - 0.25, 7):
            x, y = 5 + 3.5 * np.cos(a), 2.5 + 2.6 * np.sin(a)
            ax.add_patch(FancyArrowPatch((x, y), (x, y + 1.2), arrowstyle="-|>", mutation_scale=4, color=vap, lw=0.6))
    else:                                           # dry deposit at the rim
        ax.plot([0.8, 9.2], [2.5, 2.5], color=ink, lw=0.8)
        rng = np.random.default_rng(5)
        for xc in (2.3, 7.7):
            for _ in range(6):
                ax.add_patch(Circle((xc + rng.uniform(-0.5, 0.5), 2.62 + rng.uniform(0, 0.25)),
                                    rng.choice([0.14, 0.2]), color=S.COLORS["particle_small"], lw=0))
        for x in (1.7, 2.3, 2.9, 7.1, 7.7, 8.3):
            ax.add_patch(FancyArrowPatch((x, 3.2), (x, 4.4), arrowstyle="-|>", mutation_scale=4, color=vap, lw=0.6))


def main():
    S.apply()
    fig = S.figure(WIDTH, HEIGHT)
    margin, gap, right_w = 4.0, 2.0, 56.0
    right_h = right_w * 3 / 4                          # 4:3 image panels
    left_w = WIDTH - margin - right_w - gap - 1
    top = 4.0

    # a, b along the top-left; c, d, e down the right column; f below a, b
    ax_a = S.axes_mm(fig, margin, top, 66, right_h)
    ax_b = S.axes_mm(fig, margin + 68, top, left_w - 68, right_h)
    xr = WIDTH - right_w - 1
    ax_c = S.axes_mm(fig, xr, top, right_w, right_h)
    ax_d = S.axes_mm(fig, xr, top + right_h + gap, right_w, right_h)
    ax_e = S.axes_mm(fig, xr, top + 2 * (right_h + gap), right_w, right_h)
    f_h = 2 * right_h + gap
    f_w = f_h * 4 / 3
    ax_f = S.axes_mm(fig, margin + (left_w - f_w) / 2, top + right_h + gap, f_w, f_h)

    autosampler(ax_a)
    show(ax_b, S.load_image(CHAMBER, crop=CHAMBER_CROP))
    ax_b.set_aspect("auto")
    show(ax_c, S.load_image(OPTICAL_ORIGINAL, crop=OPTICAL_CROP), "Original", "Optical")
    show(ax_d, S.load_image(OPTICAL_ADJUSTED, crop=OPTICAL_CROP), "Adjusted", "Optical")
    show(ax_e, S.load_image(SEM_ORIGINAL, crop=SEM_CROP, gray=True), "Original", "SEM")
    sem_with_magnifiers(ax_f, S.load_image(SEM_ADJUSTED, crop=SEM_CROP, gray=True))
    ax_f.text(0.015, 0.985, "Adjusted", transform=ax_f.transAxes, color="white", ha="left", va="top")
    ax_f.text(0.985, 0.985, "SEM", transform=ax_f.transAxes, color="white", ha="right", va="top")

    # g: four workflow boxes across the bottom
    y_g = top + 3 * right_h + 2 * gap + 6
    box = min((WIDTH - margin - 1 - 3 * 4) / 4, HEIGHT - y_g - 6)
    xs = np.linspace(margin, WIDTH - 1 - box, 4)
    ax_g = []
    for i, x in enumerate(xs):
        ax = S.axes_mm(fig, x, y_g, box, box)
        workflow_step(ax, i)
        ax.text(0.5, -0.04, WORKFLOW[i], transform=ax.transAxes, ha="center", va="top")
        ax_g.append(ax)

    for ax, k in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g[0]], "abcdefg"):
        S.panel_label(ax, k, dx=0)
    S.save(fig, "zutu")


if __name__ == "__main__":
    main()
