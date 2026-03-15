#!/usr/bin/env python3
"""
Analysis 3: Spatial Statistics of Particle Distributions

Computes Ripley's K/L functions, nearest-neighbor distances, Clark-Evans R,
and Voronoi tessellation metrics for particle spatial patterns.

Depends on: particle_analysis_utils.py

Outputs:
  - spatial_statistics.csv
  - ripley_k_profiles.csv
  - figure_spatial_statistics.png / .svg
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree, Voronoi
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import (
    iterate_dataset, get_deposit_mask, detect_particles,
    extract_particle_properties, save_figure, load_sem_gray,
    CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR, PIXEL_SCALE_UM
)

setup_professional_style()
COLORS = get_professional_colors()


def compute_ripley_L(centroids, study_area, max_r=None, n_steps=30):
    """
    Compute Besag's L(r) function: L(r) = sqrt(K(r)/pi) - r.
    L(r) = 0 for CSR, > 0 for clustering, < 0 for dispersion.
    Uses isotropic edge correction.
    """
    n = len(centroids)
    if n < 5 or study_area <= 0:
        return np.array([]), np.array([])

    intensity = n / study_area
    if max_r is None:
        max_r = np.sqrt(study_area / np.pi) / 2

    r_values = np.linspace(0, max_r, n_steps)
    tree = cKDTree(centroids)

    K_values = []
    for r in r_values:
        if r == 0:
            K_values.append(0.0)
            continue
        # Count pairs within distance r
        pairs = tree.count_neighbors(tree, r)
        pairs -= n  # subtract self-pairs
        K_r = study_area * pairs / (n * (n - 1)) if n > 1 else 0
        K_values.append(K_r)

    K_arr = np.array(K_values)
    L_arr = np.sqrt(np.clip(K_arr / np.pi, 0, None)) - r_values

    return r_values, L_arr


def compute_nnd(centroids):
    """Compute nearest-neighbor distances."""
    if len(centroids) < 2:
        return np.array([])
    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=2)
    return dists[:, 1]  # distance to nearest neighbor (k=1 is self)


def compute_clark_evans(centroids, study_area):
    """
    Clark-Evans R ratio. R < 1: clustered, R = 1: random, R > 1: dispersed.
    """
    n = len(centroids)
    if n < 2 or study_area <= 0:
        return np.nan
    nnd = compute_nnd(centroids)
    mean_nnd = nnd.mean()
    expected_nnd = 0.5 * np.sqrt(study_area / n)
    return mean_nnd / expected_nnd if expected_nnd > 0 else np.nan


def compute_voronoi_metrics(centroids, study_area):
    """Compute Voronoi tessellation metrics (CV and entropy of cell areas)."""
    n = len(centroids)
    if n < 4:
        return np.nan, np.nan

    try:
        vor = Voronoi(centroids)
    except Exception:
        return np.nan, np.nan

    # Compute finite cell areas
    areas = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        polygon = vor.vertices[region]
        # Shoelace formula
        x, y = polygon[:, 0], polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if 0 < area < study_area:
            areas.append(area)

    if len(areas) < 3:
        return np.nan, np.nan

    areas = np.array(areas)
    voronoi_cv = areas.std() / areas.mean() if areas.mean() > 0 else np.nan

    # Shannon entropy of normalized area distribution
    probs = areas / areas.sum()
    probs = probs[probs > 0]
    voronoi_entropy = -np.sum(probs * np.log(probs))

    return voronoi_cv, voronoi_entropy


# ======================================================================
# Main
# ======================================================================

def run_analysis():
    print("=" * 60)
    print("Analysis 3: Spatial Statistics")
    print("=" * 60)

    all_stats = []
    all_ripley = []

    for cond, img_num, sample_id, sem_gray in tqdm(
        list(iterate_dataset()), desc="Spatial statistics", unit="img"
    ):
        mask, contour, center, equiv_r = get_deposit_mask(sem_gray)
        if equiv_r < 50:
            continue

        labeled = detect_particles(sem_gray, mask)
        particles = extract_particle_properties(labeled)

        if len(particles) < 5:
            all_stats.append({
                'condition': cond, 'image_num': img_num, 'sample_id': sample_id,
                'particle_count': len(particles),
            })
            continue

        # Convert centroids to μm
        centroids = particles[['centroid_x', 'centroid_y']].values * PIXEL_SCALE_UM
        study_area = mask.sum() * PIXEL_SCALE_UM ** 2

        # Clark-Evans R
        ce_R = compute_clark_evans(centroids, study_area)

        # NND statistics
        nnd = compute_nnd(centroids) * 1  # already in μm after centroid conversion
        nnd_mean = nnd.mean() if len(nnd) > 0 else np.nan
        nnd_median = np.median(nnd) if len(nnd) > 0 else np.nan
        nnd_std = nnd.std() if len(nnd) > 0 else np.nan
        nnd_cv = nnd_std / nnd_mean if nnd_mean > 0 else np.nan

        # Voronoi
        voronoi_cv, voronoi_entropy = compute_voronoi_metrics(centroids, study_area)

        # Ripley's L function
        r_vals, L_vals = compute_ripley_L(centroids, study_area, n_steps=30)
        L_max = np.max(L_vals) if len(L_vals) > 0 else np.nan
        L_max_r = r_vals[np.argmax(L_vals)] if len(L_vals) > 0 else np.nan

        all_stats.append({
            'condition': cond, 'image_num': img_num, 'sample_id': sample_id,
            'particle_count': len(particles),
            'clark_evans_R': ce_R,
            'nnd_mean_um': nnd_mean,
            'nnd_median_um': nnd_median,
            'nnd_std_um': nnd_std,
            'nnd_cv': nnd_cv,
            'voronoi_cv': voronoi_cv,
            'voronoi_entropy': voronoi_entropy,
            'L_max': L_max,
            'L_max_radius_um': L_max_r,
        })

        # Save Ripley profile
        for rv, lv in zip(r_vals, L_vals):
            all_ripley.append({
                'condition': cond, 'image_num': img_num,
                'r_um': rv, 'L_r': lv,
            })

    df_stats = pd.DataFrame(all_stats)
    df_ripley = pd.DataFrame(all_ripley)
    return df_stats, df_ripley


def plot_spatial(df_stats, df_ripley):
    """Publication figure: Spatial statistics (2×2 layout)."""
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    temp_colors = {'20-23': '#1f77b4', '23-26': '#ff7f0e', '26-29': '#2ca02c'}

    # (a) Mean L(r) by condition
    ax = fig.add_subplot(gs[0, 0])
    for cond in CONDITIONS:
        subset = df_ripley[df_ripley['condition'] == cond]
        if len(subset) == 0:
            continue
        grouped = subset.groupby('r_um')['L_r'].agg(['mean', 'std'])
        color = temp_colors[CONDITION_INFO[cond]['temp']]
        ax.plot(grouped.index, grouped['mean'], color=color, alpha=0.6, label=cond)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, label='CSR')
    ax.set_xlabel('r (μm)')
    ax.set_ylabel('L(r)')
    ax.set_title("(a) Ripley's L Function")
    ax.legend(fontsize=7, ncol=4)
    style_axis(ax)

    # (b) Clark-Evans R by condition
    ax = fig.add_subplot(gs[0, 1])
    df_valid = df_stats.dropna(subset=['clark_evans_R'])
    data = [df_valid[df_valid['condition'] == c]['clark_evans_R'].values for c in CONDITIONS]
    bp = ax.boxplot(data, labels=CONDITIONS, patch_artist=True, widths=0.6)
    for i, (patch, cond) in enumerate(zip(bp['boxes'], CONDITIONS)):
        patch.set_facecolor(temp_colors[CONDITION_INFO[cond]['temp']])
        patch.set_alpha(0.6)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.0, label='CSR (R=1)')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Clark-Evans R')
    ax.set_title('(b) Clark-Evans Index')
    ax.legend(fontsize=9)
    style_axis(ax)

    # (c) NND violin plots by condition
    ax = fig.add_subplot(gs[1, 0])
    for i, cond in enumerate(CONDITIONS):
        vals = df_valid[df_valid['condition'] == cond]['nnd_mean_um'].dropna()
        if len(vals) > 1:
            vp = ax.violinplot(vals, positions=[i], showmeans=True, showmedians=True, widths=0.7)
            for pc in vp['bodies']:
                pc.set_facecolor(temp_colors[CONDITION_INFO[cond]['temp']])
                pc.set_alpha(0.6)
    ax.set_xticks(range(9))
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean NND (μm)')
    ax.set_title('(c) Nearest-Neighbor Distances')
    style_axis(ax)

    # (d) Voronoi CV by condition
    ax = fig.add_subplot(gs[1, 1])
    data_v = [df_valid[df_valid['condition'] == c]['voronoi_cv'].dropna().values for c in CONDITIONS]
    bp2 = ax.boxplot(data_v, labels=CONDITIONS, patch_artist=True, widths=0.6)
    for i, (patch, cond) in enumerate(zip(bp2['boxes'], CONDITIONS)):
        patch.set_facecolor(temp_colors[CONDITION_INFO[cond]['temp']])
        patch.set_alpha(0.6)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Voronoi Cell Area CV')
    ax.set_title('(d) Voronoi Tessellation Regularity')
    style_axis(ax)

    fig.suptitle('Spatial Statistics of Particle Distributions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_spatial_statistics")
    plt.close(fig)


if __name__ == '__main__':
    df_stats, df_ripley = run_analysis()

    df_stats.to_csv(EXPERIMENT_DIR / "spatial_statistics.csv", index=False)
    print(f"\nSaved spatial_statistics.csv ({len(df_stats)} rows)")

    df_ripley.to_csv(EXPERIMENT_DIR / "ripley_k_profiles.csv", index=False)
    print(f"Saved ripley_k_profiles.csv ({len(df_ripley)} rows)")

    plot_spatial(df_stats, df_ripley)

    print("\nPer-Condition Spatial Summary:")
    for cond in CONDITIONS:
        s = df_stats[df_stats['condition'] == cond].dropna(subset=['clark_evans_R'])
        if len(s) > 0:
            print(f"  {cond}: R={s['clark_evans_R'].mean():.3f}, "
                  f"NND={s['nnd_mean_um'].mean():.1f}μm, "
                  f"Voronoi CV={s['voronoi_cv'].mean():.3f}")
    print("\nDone.")
