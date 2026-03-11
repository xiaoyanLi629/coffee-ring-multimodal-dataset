#!/usr/bin/env python3
"""Generate Figure 3: SSIM and feature distribution analysis for manuscript."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.dpi': 300,
})


def clean_3d(ax):
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('#e0e0e0')
        axis._axinfo['grid'].update(color='#e0e0e0', linewidth=0.4)


# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('comprehensive_computer_vision_features.csv')
for c in 'ABCDEFGHI':
    idx = ord(c) - ord('A')
    t = ['20-23','20-23','20-23','23-26','23-26','23-26','26-29','26-29','26-29'][idx]
    h = ['35-40','40-45','45-50'][idx % 3]
    df.loc[df['condition'] == f'condition_{c}', 'temp'] = t
    df.loc[df['condition'] == f'condition_{c}', 'hum'] = h
df['cond'] = df['condition'].str.replace('condition_', '')

nf = df['normal_freq_total_spectral_energy'].values
sf = df['sem_freq_total_spectral_energy'].values
ss = df['cross_modal_ssim'].values
nf_n = (nf - nf.min()) / (nf.max() - nf.min()) * 1.2 + 0.2
sf_n = (sf - sf.min()) / (sf.max() - sf.min()) * 6 + 1

# ── Precise layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 12))

# Coordinates: [left, bottom, width, height]
# Two rows, two columns. Each "cell" = plot + thin colorbar
# Consistent colorbar: width=0.01, height matched to plot
CB = 0.01       # colorbar width
G = 0.015       # gap plot→cbar
MID = 0.07      # gap between columns
MARGIN_L = 0.07
MARGIN_R = 0.02
MARGIN_B = 0.06
MARGIN_T = 0.03
ROW_GAP = 0.08

# Compute plot widths
# Total = MARGIN_L + PW + G + CB + MID + PW + G + CB + MARGIN_R = 1
# For row with d (no cbar): d_width = PW + G + CB (fills the cbar space)
avail_w = 1 - MARGIN_L - MARGIN_R - MID - 2*(G + CB)
PW = avail_w / 2   # ~0.375

avail_h = 1 - MARGIN_B - MARGIN_T - ROW_GAP
PH = avail_h / 2   # ~0.415

# Column x-starts
x1 = MARGIN_L
x2 = MARGIN_L + PW + G + CB + MID

# Row y-starts
y2 = MARGIN_B                    # bottom row
y1 = MARGIN_B + PH + ROW_GAP    # top row

CB_H = PH * 0.70   # colorbar is 70% of plot height
CB_Y_OFF = (PH - CB_H) / 2

# ═══════════════════════════════════════════════════════════════════════════
# (a) SSIM Heatmap
# ═══════════════════════════════════════════════════════════════════════════
# Shrink heatmap plot area (keep colorbar position same)
A_SHRINK = 0.06
ax_a = fig.add_axes([x1 + A_SHRINK, y1 + A_SHRINK/2, PW - A_SHRINK, PH - A_SHRINK])

temps = ['20-23', '23-26', '26-29']
humids = ['35-40', '40-45', '45-50']
mat = np.zeros((3, 3))
for i, t in enumerate(temps):
    for j, h in enumerate(humids):
        mask = (df['temp'] == t) & (df['hum'] == h)
        mat[i, j] = df.loc[mask, 'cross_modal_ssim'].mean()

im = ax_a.pcolormesh(np.arange(4)-0.5, np.arange(4)-0.5, mat,
                     cmap='viridis', vmin=mat.min()*0.85,
                     vmax=mat.max()*1.10, shading='flat')
for i in range(3):
    for j in range(3):
        ax_a.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center',
                  fontsize=17, fontweight='bold', color='white')

ax_a.set_xticks([0,1,2])
ax_a.set_xticklabels(['35–40%', '40–45%', '45–50%'])
ax_a.set_yticks([0,1,2])
ax_a.set_yticklabels(['20–23°C', '23–26°C', '26–29°C'])
ax_a.set_xlabel('Relative Humidity')
ax_a.set_ylabel('Temperature')
ax_a.set_xlim(-0.5, 2.5); ax_a.set_ylim(-0.5, 2.5)
ax_a.invert_yaxis()

cax_a = fig.add_axes([x1 + PW + G, y1 + CB_Y_OFF, CB, CB_H])
cb_a = plt.colorbar(im, cax=cax_a)
cb_a.set_label('SSIM', fontsize=12)
cb_a.ax.tick_params(labelsize=11)

ax_a.text(-0.06, 1.03, 'a', transform=ax_a.transAxes,
          fontsize=21, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# (b) Feature Interaction Surface
# ═══════════════════════════════════════════════════════════════════════════
ax_b = fig.add_axes([x2, y1, PW, PH], projection='3d')
clean_3d(ax_b)

xi = np.linspace(nf_n.min(), nf_n.max(), 40)
yi = np.linspace(ss.min(), ss.max(), 40)
xg, yg = np.meshgrid(xi, yi)
zg = griddata((nf_n, ss), sf_n, (xg, yg), method='linear')

surf = ax_b.plot_surface(xg, yg, zg, cmap='viridis', alpha=0.6,
                         edgecolor='none', antialiased=True)
ax_b.scatter(nf_n, ss, sf_n, c='#e74c3c', s=10, alpha=0.5, edgecolors='none')

ax_b.set_xlabel('Normal Freq. Energy', fontsize=12, labelpad=8)
ax_b.set_ylabel('SSIM', fontsize=12, labelpad=8)
ax_b.set_zlabel('SEM Freq.\nEnergy', fontsize=12, labelpad=6)
ax_b.view_init(elev=28, azim=-55)
ax_b.tick_params(labelsize=13, pad=2)
ax_b.xaxis.set_major_locator(plt.MaxNLocator(5))
ax_b.yaxis.set_major_locator(plt.MaxNLocator(5))
ax_b.zaxis.set_major_locator(plt.MaxNLocator(4))

cax_b = fig.add_axes([x2 + PW + G, y1 + CB_Y_OFF, CB, CB_H])
cb_b = plt.colorbar(surf, cax=cax_b)
cb_b.ax.tick_params(labelsize=11)

ax_b.text2D(-0.01, 1.01, 'b', transform=ax_b.transAxes,
            fontsize=21, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# (c) 3D Feature Density
# ═══════════════════════════════════════════════════════════════════════════
ax_c = fig.add_axes([x1, y2, PW, PH], projection='3d')
clean_3d(ax_c)

xyz = np.vstack([nf_n, ss, sf_n])
try:
    kde = gaussian_kde(xyz, bw_method=0.15)
    density = kde(xyz)
except:
    density = np.ones(len(nf_n))
d_n = (density - density.min()) / (density.max() - density.min())

sc = ax_c.scatter(nf_n, ss, sf_n, c=density, cmap='viridis',
                  s=25 + d_n*100, alpha=0.75,
                  edgecolors='#888888', linewidths=0.3)

ax_c.set_xlabel('Normal Freq. Energy', fontsize=12, labelpad=8)
ax_c.set_ylabel('SSIM', fontsize=12, labelpad=8)
ax_c.set_zlabel('SEM Freq.\nEnergy', fontsize=12, labelpad=6)
ax_c.view_init(elev=22, azim=-48)
ax_c.tick_params(labelsize=13, pad=2)
ax_c.xaxis.set_major_locator(plt.MaxNLocator(5))
ax_c.yaxis.set_major_locator(plt.MaxNLocator(5))
ax_c.zaxis.set_major_locator(plt.MaxNLocator(4))

cax_c = fig.add_axes([x1 + PW + G, y2 + CB_Y_OFF, CB, CB_H])
cb_c = plt.colorbar(sc, cax=cax_c)
cb_c.set_label('Density', fontsize=12)
cb_c.ax.tick_params(labelsize=11)

ax_c.text2D(-0.01, 1.01, 'c', transform=ax_c.transAxes,
            fontsize=21, fontweight='bold', va='top')

# ═══════════════════════════════════════════════════════════════════════════
# (d) Ridge Plot — wider: fills plot + gap + cbar space
# ═══════════════════════════════════════════════════════════════════════════
# d right edge aligns with b's colorbar right edge: x2 + PW + G + CB
D_SHRINK_H = 0.04
d_right = x2 + PW + G + CB   # same right edge as b's colorbar
d_height = PH - D_SHRINK_H
d_width = d_right - x2 - D_SHRINK_H  # shift right by D_SHRINK_H
ax_d = fig.add_axes([x2 + D_SHRINK_H, y2 + D_SHRINK_H/2,
                     d_right - x2 - D_SHRINK_H, d_height])

conditions = list('ABCDEFGHI')
ridge_colors = [plt.cm.viridis(x) for x in np.linspace(0.0, 0.95, 9)]

for i, cond in enumerate(conditions):
    data = df.loc[df['cond'] == cond, 'cross_modal_ssim'].values
    if len(data) < 3:
        continue
    kde_r = gaussian_kde(data, bw_method=0.4)
    xr = np.linspace(0, 0.7, 300)
    yd = kde_r(xr)
    yd = yd / yd.max() * 0.85
    ax_d.fill_between(xr, i, i + yd, alpha=0.75,
                      color=ridge_colors[i], zorder=9-i)
    ax_d.plot(xr, i + yd, color=ridge_colors[i], lw=1.2, zorder=9-i)
    ax_d.axhline(y=i, color='#dddddd', lw=0.4, zorder=0)

ax_d.set_yticks(range(9))
ax_d.set_yticklabels(conditions, fontsize=13)
ax_d.set_xlabel('SSIM Value')
ax_d.set_xlim(0, 0.7)
ax_d.set_ylim(-0.3, 9.8)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.spines['left'].set_visible(False)
ax_d.tick_params(left=False)

ax_d.text(-0.05, 1.03, 'd', transform=ax_d.transAxes,
          fontsize=21, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
out = '../manuscript/figures/analysis2.pdf'
fig.savefig(out, format='pdf', dpi=300)
fig.savefig(out.replace('.pdf', '.png'), format='png', dpi=300)
print(f'Saved: {out}')
plt.close()
