"""Figure 3 (data_overview.pdf): representative multi-modal images.

Columns are droplets, rows are modalities. Optical and SEM images come from
corrected_images; EDS maps are the unmodified instrument exports.
"""
import numpy as np

import style as S

# ── Parameters ───────────────────────────────────────────────────────────────
WIDTH = S.FULL_W
# (condition, droplet number, optical rotation in degrees clockwise).
# Optical images are rotated so the deposit has the same orientation as in SEM.
COLUMNS = [
    ("A", 1, 90), ("A", 2, 90), ("A", 3, 90),
    ("C", 1, 90), ("C", 2, 90),
    ("E", 1, 270), ("E", 2, 90), ("E", 3, 90),
]
# rows: (label, kind, element file name for EDS)
ROWS = [
    ("Optical", "optical", None),
    ("SEM", "sem", None),
    ("EDS (Cl)", "eds", "Cl Kα1.png"),
    ("EDS (Ca)", "eds", "Ca Kα1.png"),
    ("EDS (O)", "eds", "O Kα1.png"),
]
# crop (left, top, right, bottom) per column and image kind, as fractions;
# taken from the earlier hand-made layout so each tile frames the deposit
CROPS = {
    ("A", 1): {"optical": (0.2692, 0.3089, 0.2938, 0.1981), "sem": (0.1871, 0.0744, 0.2846, 0.1517), "eds": (0.1758, 0.052, 0.2819, 0.1459)},
    ("A", 2): {"optical": (0.2807, 0.2677, 0.2923, 0.2379), "sem": (0.2552, 0.0909, 0.2314, 0.1258), "eds": (0.2384, 0.0804, 0.2322, 0.113)},
    ("A", 3): {"optical": (0.269, 0.2707, 0.2482, 0.2666), "sem": (0.2405, 0.0479, 0.2692, 0.0937), "eds": (0.2416, 0.04, 0.266, 0.0853)},
    ("C", 1): {"optical": (0.2983, 0.2779, 0.2846, 0.2018), "sem": (0.2297, 0.1771, 0.2136, 0.1194), "eds": (0.2086, 0.1611, 0.1899, 0.1125)},
    ("C", 2): {"optical": (0.3223, 0.3014, 0.3237, 0.2783), "sem": (0.2523, 0.1771, 0.2607, 0.1661), "eds": (0.2541, 0.1738, 0.2666, 0.1764)},
    ("E", 1): {"optical": (0.2554, 0.1332, 0.2113, 0.133), "sem": (0.0801, 0.0287, 0.1069, 0.0589), "eds": (0.0739, 0.0, 0.1293, 0.0565)},
    ("E", 2): {"optical": (0.2282, 0.1494, 0.2236, 0.1197), "sem": (0.1098, 0.0157, 0.1356, 0.0351), "eds": (0.092, 0.0, 0.1241, 0.0219)},
    ("E", 3): {"optical": (0.2044, 0.154, 0.2114, 0.1465), "sem": (0.1083, 0.0, 0.1409, 0.0), "eds": (0.0926, 0.0, 0.1308, 0.0)},
}
# Display-only brightness adjustment (data files are not changed):
# SEM tiles are histogram-matched to their common average distribution, so
# every tile has the same grey-level histogram;
# each EDS map is reduced to its signal intensity, lightly smoothed so sparse
# counts stay visible at print size, lifted with a gamma, scaled so its mean
# brightness equals EDS_MEAN, and recoloured with a display colour of similar
# lightness for every element (the instrument's dark blue for Ca would
# otherwise look much dimmer than Cl and O).
EDS_SMOOTH = 0.8                    # Gaussian sigma in EDS pixels (maps are 630 × 472)
EDS_GAMMA = 0.6
EDS_MEAN = 0.24                     # target mean brightness of every EDS tile
EDS_COLORS = {"Cl": "#FFA024", "Ca": "#3FB4FF", "O": "#43D65A"}
TILE_W, TILE_H = 20.2, 22.4        # mm, portrait tiles
GAP_X, GAP_Y = 1.0, 1.4            # mm
LEFT, TOP = 6.0, 5.0               # mm reserved for row / column labels


def path_for(kind, cond, n, element):
    if kind == "optical":
        return S.DATA / "corrected_images" / cond / f"Image_{n}.jpg"
    if kind == "sem":
        return S.DATA / "corrected_images" / cond / f"SEM_{n}.jpg"
    return S.DATA / "eds" / cond / str(n) / element


QUANTILES = np.linspace(0, 1, 257)


def sem_reference(images):
    """Average quantile function of all SEM tiles (the common target histogram)."""
    return np.mean([np.quantile(im, QUANTILES) for im in images], axis=0)


def match_histogram(img, ref):
    q = np.quantile(img, QUANTILES)
    q = np.maximum.accumulate(q + np.arange(len(q)) * 1e-9)    # strictly increasing
    return np.interp(img, q, ref) / 255


def brighten_eds(img, element):
    from matplotlib.colors import to_rgb
    from scipy.ndimage import gaussian_filter
    signal = gaussian_filter(img.astype(float).max(axis=2) / 255, EDS_SMOOTH)
    level = (signal / max(np.percentile(signal, 99.5), 1e-6)).clip(0, 1) ** EDS_GAMMA
    level = np.clip(level * (EDS_MEAN / max(level.mean(), 1e-6)), 0, 1)
    return level[..., None] * np.array(to_rgb(EDS_COLORS[element]))


def main():
    S.apply()
    def read(kind, cond, n, rot, element=None):
        return S.load_image(path_for(kind, cond, n, element), crop=CROPS[(cond, n)][kind],
                            rotate=rot if kind == "optical" else 0, gray=(kind == "sem"))

    sem_ref = sem_reference([read("sem", c, n, r) for c, n, r in COLUMNS])
    height = TOP + len(ROWS) * TILE_H + (len(ROWS) - 1) * GAP_Y + 1
    fig = S.figure(WIDTH, height)
    for j, (cond, n, rot) in enumerate(COLUMNS):
        x = LEFT + j * (TILE_W + GAP_X)
        for i, (label, kind, element) in enumerate(ROWS):
            y = TOP + i * (TILE_H + GAP_Y)
            ax = S.axes_mm(fig, x, y, TILE_W, TILE_H)
            img = read(kind, cond, n, rot, element)
            if kind == "sem":
                ax.imshow(match_histogram(img, sem_ref), cmap="gray", vmin=0, vmax=1,
                          aspect="auto", interpolation="antialiased")
            else:
                ax.imshow(brighten_eds(img, element.split()[0]) if kind == "eds" else img, aspect="auto",
                          interpolation="antialiased")
            S.image_axes(ax)
            if i == 0:
                ax.set_title(f"{cond}, droplet {n}", pad=2, fontweight="bold")
            if j == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontweight="bold")
    S.save(fig, "data_overview")


if __name__ == "__main__":
    main()
