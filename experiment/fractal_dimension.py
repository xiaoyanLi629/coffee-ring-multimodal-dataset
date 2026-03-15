#!/usr/bin/env python3
"""
Analysis 4: Fractal Dimension Analysis

Computes box-counting fractal dimension of deposit boundaries and
particle spatial distributions.

Outputs:
  - fractal_dimensions.csv
  - figure_fractal_dimension.png / .svg
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import (
    iterate_dataset, get_deposit_mask, detect_particles,
    extract_particle_properties, save_figure,
    CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR
)

setup_professional_style()


def box_counting_fractal(binary_image, box_sizes=None):
    """
    Compute fractal dimension via box-counting method.

    Parameters
    ----------
    binary_image : ndarray (bool)
        Binary image (True = foreground).
    box_sizes : list or None
        Box sizes in pixels. If None, uses powers of 2.

    Returns
    -------
    D : float
        Fractal dimension (slope of log(N) vs log(1/eps)).
    R2 : float
        R² of the linear fit.
    epsilons : ndarray
        Box sizes used.
    counts : ndarray
        Box counts at each size.
    """
    if box_sizes is None:
        max_dim = max(binary_image.shape)
        box_sizes = [2 ** i for i in range(1, int(np.log2(max_dim)))]

    epsilons = []
    counts = []

    for eps in box_sizes:
        h, w = binary_image.shape
        # Count boxes that contain at least one True pixel
        n_count = 0
        for r in range(0, h, eps):
            for c in range(0, w, eps):
                box = binary_image[r:r + eps, c:c + eps]
                if box.any():
                    n_count += 1
        if n_count > 0:
            epsilons.append(eps)
            counts.append(n_count)

    if len(epsilons) < 3:
        return np.nan, np.nan, np.array([]), np.array([])

    epsilons = np.array(epsilons, dtype=float)
    counts = np.array(counts, dtype=float)

    # Linear fit: log(N) = D * log(1/eps) + c
    log_inv_eps = np.log(1.0 / epsilons)
    log_N = np.log(counts)

    coeffs = np.polyfit(log_inv_eps, log_N, 1)
    D = coeffs[0]

    # R² calculation
    y_pred = np.polyval(coeffs, log_inv_eps)
    ss_res = np.sum((log_N - y_pred) ** 2)
    ss_tot = np.sum((log_N - log_N.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return D, R2, epsilons, counts


def compute_boundary_fractal(mask):
    """Compute fractal dimension of the deposit boundary."""
    from skimage import measure as sk_measure
    # Extract boundary
    contours = sk_measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.nan, np.nan, None, None

    contour = max(contours, key=len)
    # Create a thin boundary image
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for pt in contour.astype(int):
        r, c = pt
        if 0 <= r < h and 0 <= c < w:
            boundary[r, c] = True

    # Crop to bounding box with padding
    rows = np.any(boundary, axis=1)
    cols = np.any(boundary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 10
    rmin = max(0, rmin - pad)
    rmax = min(h, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w, cmax + pad)
    cropped = boundary[rmin:rmax, cmin:cmax]

    return box_counting_fractal(cropped)


def compute_particle_fractal(particles, img_shape):
    """Compute fractal dimension of particle centroid point cloud."""
    if len(particles) < 10:
        return np.nan, np.nan, None, None

    # Create binary image from centroids
    point_img = np.zeros(img_shape, dtype=bool)
    for _, p in particles.iterrows():
        r, c = int(p['centroid_y']), int(p['centroid_x'])
        if 0 <= r < img_shape[0] and 0 <= c < img_shape[1]:
            # Use a small disk around each centroid
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < img_shape[0] and 0 <= cc < img_shape[1]:
                        point_img[rr, cc] = True

    # Crop to deposit region
    rows = np.any(point_img, axis=1)
    cols = np.any(point_img, axis=0)
    if not rows.any():
        return np.nan, np.nan, None, None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 10
    rmin = max(0, rmin - pad)
    rmax = min(img_shape[0], rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(img_shape[1], cmax + pad)
    cropped = point_img[rmin:rmax, cmin:cmax]

    return box_counting_fractal(cropped)


# ======================================================================
# Main
# ======================================================================

def run_analysis():
    print("=" * 60)
    print("Analysis 4: Fractal Dimension Analysis")
    print("=" * 60)

    all_results = []

    for cond, img_num, sample_id, sem_gray in tqdm(
        list(iterate_dataset()), desc="Fractal analysis", unit="img"
    ):
        mask, contour, center, equiv_r = get_deposit_mask(sem_gray)
        if equiv_r < 50:
            continue

        # Boundary fractal dimension
        bD, bR2, _, _ = compute_boundary_fractal(mask)

        # Particle fractal dimension
        labeled = detect_particles(sem_gray, mask)
        particles = extract_particle_properties(labeled)
        pD, pR2, _, _ = compute_particle_fractal(particles, sem_gray.shape)

        all_results.append({
            'condition': cond, 'image_num': img_num, 'sample_id': sample_id,
            'boundary_fractal_D': bD,
            'boundary_fractal_R2': bR2,
            'particle_fractal_D': pD,
            'particle_fractal_R2': pR2,
        })

    return pd.DataFrame(all_results)


def plot_fractal(df):
    """Publication figure: Fractal analysis (2×2 layout)."""
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    temp_colors = {'20-23': '#1f77b4', '23-26': '#ff7f0e', '26-29': '#2ca02c'}
    df_valid = df.dropna(subset=['boundary_fractal_D'])

    # (a) Example box-counting (one sample)
    ax = fig.add_subplot(gs[0, 0])
    from particle_analysis_utils import load_sem_gray
    sem = load_sem_gray('E', 13, corrected=True)
    mask_ex, _, _, _ = get_deposit_mask(sem)
    D_ex, R2_ex, eps_ex, counts_ex = compute_boundary_fractal(mask_ex)
    if eps_ex is not None and len(eps_ex) > 0:
        log_inv = np.log(1.0 / eps_ex)
        log_N = np.log(counts_ex)
        ax.scatter(log_inv, log_N, c='#1f77b4', s=40, zorder=5, edgecolors='k', linewidth=0.5)
        coeffs = np.polyfit(log_inv, log_N, 1)
        x_fit = np.linspace(log_inv.min(), log_inv.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', linewidth=1.5,
                label=f'D = {D_ex:.3f}, R² = {R2_ex:.3f}')
        ax.set_xlabel('log(1/ε)')
        ax.set_ylabel('log(N)')
        ax.legend(fontsize=9)
    ax.set_title('(a) Box-Counting Example (Cond. E)')
    style_axis(ax)

    # (b) Boundary fractal D by condition
    ax = fig.add_subplot(gs[0, 1])
    data = [df_valid[df_valid['condition'] == c]['boundary_fractal_D'].values for c in CONDITIONS]
    bp = ax.boxplot(data, labels=CONDITIONS, patch_artist=True, widths=0.6)
    for i, (patch, cond) in enumerate(zip(bp['boxes'], CONDITIONS)):
        patch.set_facecolor(temp_colors[CONDITION_INFO[cond]['temp']])
        patch.set_alpha(0.6)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Boundary Fractal Dimension')
    ax.set_title('(b) Boundary Fractal D')
    style_axis(ax)

    # (c) Scatter: boundary D vs particle D
    ax = fig.add_subplot(gs[1, 0])
    df_both = df.dropna(subset=['boundary_fractal_D', 'particle_fractal_D'])
    for cond in CONDITIONS:
        s = df_both[df_both['condition'] == cond]
        if len(s) > 0:
            ax.scatter(s['boundary_fractal_D'], s['particle_fractal_D'],
                       c=temp_colors[CONDITION_INFO[cond]['temp']],
                       label=cond, s=25, alpha=0.6, edgecolors='k', linewidth=0.2)
    ax.set_xlabel('Boundary Fractal D')
    ax.set_ylabel('Particle Fractal D')
    ax.set_title('(c) Boundary vs Particle D')
    ax.legend(fontsize=7, ncol=3)
    style_axis(ax)

    # (d) Heatmap: 3×3 temperature × humidity of mean boundary D
    ax = fig.add_subplot(gs[1, 1])
    temps = ['20-23', '23-26', '26-29']
    rhs = ['35-40', '40-45', '45-50']
    hm = np.full((3, 3), np.nan)
    for i, t in enumerate(temps):
        for j, r in enumerate(rhs):
            cond = [c for c in CONDITIONS if CONDITION_INFO[c]['temp'] == t and CONDITION_INFO[c]['rh'] == r]
            if cond:
                vals = df_valid[df_valid['condition'] == cond[0]]['boundary_fractal_D']
                if len(vals) > 0:
                    hm[i, j] = vals.mean()
    sns.heatmap(hm, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'{r}%' for r in rhs],
                yticklabels=[f'{t}°C' for t in temps],
                cbar_kws={'label': 'Fractal D'})
    ax.set_xlabel('Relative Humidity')
    ax.set_ylabel('Temperature')
    ax.set_title('(d) Environmental Effect on Fractal D')

    fig.suptitle('Fractal Dimension Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_fractal_dimension")
    plt.close(fig)


if __name__ == '__main__':
    df = run_analysis()
    df.to_csv(EXPERIMENT_DIR / "fractal_dimensions.csv", index=False)
    print(f"\nSaved fractal_dimensions.csv ({len(df)} rows)")

    plot_fractal(df)

    print("\nPer-Condition Fractal Summary:")
    for cond in CONDITIONS:
        s = df[df['condition'] == cond].dropna(subset=['boundary_fractal_D'])
        if len(s) > 0:
            print(f"  {cond}: boundary D={s['boundary_fractal_D'].mean():.3f}±{s['boundary_fractal_D'].std():.3f}, "
                  f"particle D={s['particle_fractal_D'].mean():.3f}±{s['particle_fractal_D'].std():.3f}")
    print("\nDone.")
