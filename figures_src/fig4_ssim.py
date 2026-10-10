"""Figure 4 (analysis2.pdf): SSIM and feature distributions across conditions.

Calculations are unchanged from experiment/generate_figure3.py; only the
layout and style follow the shared figure style.
a  mean cross-modal SSIM per temperature × humidity cell
b  feature interaction surface    c  3D feature density
d  per-condition SSIM distributions (ridge plot)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

import style as S

# ── Parameters ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = S.FULL_W, 140          # mm
FEATURES = S.DATA / "features" / "comprehensive_computer_vision_features.csv"
TEMPS = ["20–23", "23–26", "26–29"]
HUMIDS = ["35–40", "40–45", "45–50"]
# The two frequency energies are min–max rescaled for display only, as in the
# original script: optical to [0.2, 1.4], SEM to [1, 7].
OPT_RANGE, SEM_RANGE = (0.2, 1.4), (1.0, 7.0)
VIEW_B, VIEW_C = (28, -55), (22, -48)


def rescale(v, lo, hi):
    return (v - v.min()) / (v.max() - v.min()) * (hi - lo) + lo


def clean_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"].update(color="#e6e6e6", linewidth=0.3)
    ax.tick_params(pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))


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
    nf = rescale(df["normal_freq_total_spectral_energy"].values, *OPT_RANGE)
    sf = rescale(df["sem_freq_total_spectral_energy"].values, *SEM_RANGE)
    ss = df["cross_modal_ssim"].values

    fig = S.figure(WIDTH, HEIGHT)
    ax_a = S.axes_mm(fig, 18, 6, 58, 52)
    ax_b = S.axes_mm(fig, 92, 4, 64, 60, projection="3d")
    ax_c = S.axes_mm(fig, 6, 74, 64, 60, projection="3d")
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

    # b: interaction surface
    clean_3d(ax_b)
    xg, yg = np.meshgrid(np.linspace(nf.min(), nf.max(), 40), np.linspace(ss.min(), ss.max(), 40))
    zg = griddata((nf, ss), sf, (xg, yg), method="linear")
    surf = ax_b.plot_surface(xg, yg, zg, cmap=S.CONDITION_CMAP, alpha=0.6, edgecolor="none")
    ax_b.scatter(nf, ss, sf, c="#D9473B", s=2, alpha=0.6, edgecolors="none")
    ax_b.set_xlabel("Optical freq. energy", labelpad=-4)
    ax_b.set_ylabel("SSIM", labelpad=-4)
    ax_b.set_zlabel("SEM freq. energy", labelpad=-2)
    ax_b.view_init(*VIEW_B)
    colorbar(fig, surf, ax_b, offset=0.06)

    # c: density in feature space
    clean_3d(ax_c)
    xyz = np.vstack([nf, ss, sf])
    density = gaussian_kde(xyz, bw_method=0.15)(xyz)
    dn = (density - density.min()) / (density.max() - density.min())
    sc = ax_c.scatter(nf, ss, sf, c=density, cmap=S.CONDITION_CMAP, s=3 + dn * 12, alpha=0.8,
                      edgecolors="#888888", linewidths=0.15)
    ax_c.set_xlabel("Optical freq. energy", labelpad=-4)
    ax_c.set_ylabel("SSIM", labelpad=-4)
    ax_c.set_zlabel("SEM freq. energy", labelpad=-2)
    ax_c.view_init(*VIEW_C)
    colorbar(fig, sc, ax_c, "Density", offset=0.06)

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

    for ax, k, dx in [(ax_a, "a", -12), (ax_b, "b", 2), (ax_c, "c", 2), (ax_d, "d", -6)]:
        S.panel_label(ax, k, dx=dx, dy=-3 if ax in (ax_b, ax_c) else 0.5)
    S.save(fig, "analysis2")


if __name__ == "__main__":
    main()
