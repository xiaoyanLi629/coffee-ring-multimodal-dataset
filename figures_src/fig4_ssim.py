"""Figure 4 (analysis2.pdf): cross-modal registration and SSIM across conditions.

SSIM values are the registered cross-modal SSIM (experiment/cross_modal_registration.py).
a  mean cross-modal SSIM per temperature × humidity cell
b  example of a registered optical / SEM pair (crystal features overlaid)
c  crystal-map correlation for matched droplets versus mismatched droplets
d  per-condition SSIM distributions (ridge plot)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import sys
import cv2

import style as S

sys.path.insert(0, str(S.ROOT / "experiment"))

# ── Parameters ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = S.FULL_W, 140          # mm
FEATURES = S.DATA / "features" / "comprehensive_computer_vision_features.csv"
TEMPS = ["20–23", "23–26", "26–29"]
HUMIDS = ["35–40", "40–45", "45–50"]
REGISTRATION = S.ROOT / "experiment" / "cross_modal_registration.csv"
NULL = S.ROOT / "experiment" / "registration_null.csv"
EXAMPLE = ("E", 2)                     # pair shown in panel b (same deposit as Fig. 2)
MATCHED_COLOR, SHUFFLED_COLOR = "#2B6CB0", "#A0A7B4"


def registered_overlay(cond, i):
    """SEM (magenta) with the registered optical image (green), cropped to the droplet."""
    from cross_modal_registration import gray, droplet_mask, stretch, WORK_W
    reg = pd.read_csv(REGISTRATION)
    r = reg[(reg.condition == cond) & (reg.image_id == i)].iloc[0]
    M = np.array([[r.M00, r.M01, r.M02], [r.M10, r.M11, r.M12]])
    def work(name):
        g = gray(S.DATA / "corrected_images" / cond / name)
        f = WORK_W / g.shape[1]
        return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sem, opt = work(f"SEM_{i}.jpg"), work(f"Image_{i}.jpg")
    h, w = sem.shape
    opt_r = cv2.warpAffine(opt, M, (w, h))
    m = droplet_mask(sem)
    ys, xs = np.where(m)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    a, b = stretch(sem, m), stretch(opt_r, m)
    return np.dstack([a, b, a])[y0:y1, x0:x1], r.final_ncc


def colorbar(fig, mappable, ax, label=None, offset=0.008):
    p = ax.get_position()
    cax = fig.add_axes([p.x1 + offset, p.y0 + p.height * 0.15, 0.008, p.height * 0.7])
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_linewidth(S.LINE_W)
    if label:
        cb.set_label(label)
    return cb


def main():
    S.apply()
    df = pd.read_csv(FEATURES)
    df["cond"] = df["condition"].str.replace("condition_", "")
    df["temp"] = df["cond"].map(S.CONDITION_TEMP)
    df["hum"] = df["cond"].map(S.CONDITION_RH)
    fig = S.figure(WIDTH, HEIGHT)
    ax_a = S.axes_mm(fig, 18, 6, 58, 52)
    ax_b = S.axes_mm(fig, 104, 6, 52, 52)
    ax_c = S.axes_mm(fig, 18, 86, 66, 42)
    ax_d = S.axes_mm(fig, 104, 76, 70, 54)

    # a: SSIM heatmap
    mat = np.array([[df.loc[(df.temp == t) & (df.hum == h), "cross_modal_ssim"].mean()
                     for h in HUMIDS] for t in TEMPS])
    im = ax_a.pcolormesh(np.arange(4) - 0.5, np.arange(4) - 0.5, mat, cmap=S.CONDITION_CMAP,
                         vmin=mat.min() * 0.85, vmax=mat.max() * 1.10, shading="flat")
    for i in range(3):
        for j in range(3):
            ax_a.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontweight="bold")
    ax_a.set_xticks(range(3), [f"{h}%" for h in HUMIDS])
    ax_a.set_yticks(range(3), [f"{t} °C" for t in TEMPS])
    ax_a.set_xlabel("Relative humidity")
    ax_a.set_ylabel("Temperature")
    ax_a.invert_yaxis()
    colorbar(fig, im, ax_a, "SSIM")

    # b: registered example
    img, ncc_ex = registered_overlay(*EXAMPLE)
    ax_b.imshow(img, interpolation="antialiased")
    ax_b.set_axis_off()
    ax_b.text(0.02, 0.98, "SEM", color="#FF66FF", transform=ax_b.transAxes, ha="left", va="top", fontweight="bold")
    ax_b.text(0.98, 0.98, "Optical (registered)", color="#66FF66", transform=ax_b.transAxes, ha="right", va="top", fontweight="bold")
    ax_b.text(0.02, 0.02, f"Condition {EXAMPLE[0]}, droplet {EXAMPLE[1]}  ·  NCC = {ncc_ex:.2f}", color="white",
              transform=ax_b.transAxes, ha="left", va="bottom")

    # c: matched versus mismatched droplets
    null = pd.read_csv(NULL)
    bins = np.linspace(-0.3, 1.0, 40)
    for col, lab, colr in [("shuffled_ncc", "Different droplet", SHUFFLED_COLOR), ("matched_ncc", "Same droplet", MATCHED_COLOR)]:
        v = null[col].values
        ax_c.hist(v, bins=bins, color=colr, alpha=0.75, label=f"{lab} (median {np.median(v):.2f})")
        ax_c.axvline(np.median(v), color=colr, lw=1.0, ls="--")
    ax_c.set_xlabel("Crystal-map correlation after registration (NCC)")
    ax_c.set_ylabel("Number of pairs")
    ax_c.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.2, columnspacing=1.2)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    # d: ridge plot
    cmap = plt.get_cmap(S.CONDITION_CMAP)
    xr = np.linspace(0, 0.7, 300)
    for i, cond in enumerate(S.CONDITIONS):
        v = df.loc[df.cond == cond, "cross_modal_ssim"].values
        yd = gaussian_kde(v, bw_method=0.4)(xr)
        yd = yd / yd.max() * 0.85
        col = cmap(i / 8 * 0.95)
        ax_d.fill_between(xr, i, i + yd, color=col, alpha=0.75, lw=0, zorder=9 - i)
        ax_d.plot(xr, i + yd, color=col, lw=0.8, zorder=9 - i)
        ax_d.axhline(i, color="#dddddd", lw=0.3, zorder=0)
    ax_d.set_yticks(range(9), S.CONDITIONS)
    ax_d.set_xlabel("SSIM")
    ax_d.set_xlim(0, 0.7)
    ax_d.set_ylim(-0.3, 9.8)
    for side in ("top", "right", "left"):
        ax_d.spines[side].set_visible(False)
    ax_d.tick_params(left=False)

    for ax, k, dx in [(ax_a, "a", -12), (ax_b, "b", -1), (ax_c, "c", -12), (ax_d, "d", -6)]:
        S.panel_label(ax, k, dx=dx)
    S.save(fig, "analysis2")


if __name__ == "__main__":
    main()
