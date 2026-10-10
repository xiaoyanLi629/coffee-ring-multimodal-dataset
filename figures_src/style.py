"""Shared style, paths and helpers for all manuscript figures.

Every figure script imports this module, so fonts, sizes and colours are
changed in one place. Sizes follow Scientific Data: 180 mm full width,
88 mm single column, 7 pt text, 8 pt bold panel letters.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DATA = ROOT / "data"
ASSETS = SRC / "assets"
OUT = ROOT / "manuscript" / "figures"

# ── Sizes ────────────────────────────────────────────────────────────────────
MM = 1 / 25.4                     # inches per millimetre
FULL_W = 180                      # mm, double-column width
HALF_W = 88                       # mm, single-column width
FONT_SIZE = 7                     # pt, all ordinary text
LABEL_SIZE = 8                    # pt, panel letters a, b, c ...
LINE_W = 0.6                      # pt, axes and drawn lines

# ── Colours ──────────────────────────────────────────────────────────────────
COLORS = {
    # particle size classes in the coffee-ring schematic
    "particle_large": "#2E7D4F",
    "particle_medium": "#E0823D",
    "particle_small": "#7A64A8",
    "droplet": "#E6F2FA",
    "liquid": "#BFE0F5",
    "vapour": "#4FA3D9",
    "ink": "#1A1A1A",
    "grey": "#8C8C8C",
    "highlight": "#FFFFFF",       # magnifier rings on SEM images
}
# nine environmental conditions A–I, used wherever conditions are coloured
CONDITION_CMAP = "viridis"
CONDITIONS = list("ABCDEFGHI")
CONDITION_TEMP = dict(zip(CONDITIONS, ["20–23"] * 3 + ["23–26"] * 3 + ["26–29"] * 3))
CONDITION_RH = dict(zip(CONDITIONS, ["35–40", "40–45", "45–50"] * 3))


FONT_DIRS = ["/usr/share/fonts/truetype/arial", "/System/Library/Fonts/Supplemental",
             "/Library/Fonts"]


def _register_arial():
    """matplotlib's font cache can miss system fonts; register Arial explicitly."""
    from matplotlib import font_manager
    for d in FONT_DIRS:
        for f in Path(d).glob("[Aa]rial*.ttf"):
            font_manager.fontManager.addfont(str(f))


def apply():
    """Set matplotlib defaults shared by every figure."""
    _register_arial()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.linewidth": LINE_W,
        "xtick.major.width": LINE_W,
        "ytick.major.width": LINE_W,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
        "mathtext.sf": "Arial",
        "pdf.fonttype": 42,           # embed TrueType: text stays editable
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def figure(width_mm, height_mm):
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    return fig


def axes_mm(fig, x, y, w, h, **kw):
    """Add axes from a top-left origin in millimetres (easier to lay out)."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def panel_label(ax, letter, dx=-1.5, dy=0.5):
    """Bold lower-case letter just outside the top-left corner (offsets in mm)."""
    fig = ax.figure
    W, H = fig.get_size_inches() / MM
    x0, y1 = ax.get_position().x0, ax.get_position().y1
    fig.text(x0 + dx / W, y1 + dy / H, letter, fontsize=LABEL_SIZE,
             fontweight="bold", ha="right", va="bottom")


def image_axes(ax):
    ax.set_axis_off()
    return ax


def load_image(path, rotate=0, crop=None, gray=False):
    """Read an image as an array.

    crop is (left, top, right, bottom) as fractions removed from each side,
    applied before rotation; rotate is degrees clockwise (0, 90, 180, 270).
    """
    im = Image.open(path)
    im = im.convert("L" if gray else "RGB")
    if crop is not None:
        l, t, r, b = crop
        w, h = im.size
        im = im.crop((round(l * w), round(t * h), round((1 - r) * w), round((1 - b) * h)))
    if rotate:
        im = im.rotate(-rotate, expand=True)
    return np.asarray(im)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"saved {OUT / name}.pdf / .png")
