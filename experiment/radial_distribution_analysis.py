#!/usr/bin/env python3
"""
Analysis 2: Radial Distribution Analysis of Coffee Ring Deposits

Computes radial intensity profiles and particle density as a function
of normalized distance from deposit center to edge. Extracts ring metrics
(peak position, width, center-to-edge ratio).

Depends on: particle_summary_features.csv (from particle_morphometry.py)

Outputs:
  - radial_profiles.csv
  - radial_metrics.csv
  - figure_radial_distribution.png / .svg
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import (
    iterate_dataset, get_deposit_mask, detect_particles,
    extract_particle_properties, save_figure, load_sem_gray,
    CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR, PIXEL_SCALE_UM
)

setup_professional_style()
COLORS = get_professional_colors()
N_BINS = 50


def compute_radial_profile(sem_gray, mask, center, equiv_radius, n_bins=N_BINS):
    """Compute radial intensity profile within the deposit."""
    cy, cx = center
    h, w = sem_gray.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Normalize distance by equivalent radius
    norm_dist = dist / max(equiv_radius, 1)

    profiles = []
    for i in range(n_bins):
        r_lo = i / n_bins
        r_hi = (i + 1) / n_bins
        ring_mask = mask & (norm_dist >= r_lo) & (norm_dist < r_hi)
        n_px = ring_mask.sum()
        if n_px > 0:
            mean_intensity = sem_gray[ring_mask].mean()
        else:
            mean_intensity = np.nan
        profiles.append({
            'radius_bin': i,
            'r_center': (r_lo + r_hi) / 2,
            'mean_intensity': mean_intensity,
            'pixel_count': n_px,
        })

    return pd.DataFrame(profiles)


def compute_radial_particle_density(particle_df, center, equiv_radius, mask, n_bins=N_BINS):
    """Compute radial particle density within annular rings."""
    cy, cx = center
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    norm_dist = dist / max(equiv_radius, 1)

    densities = []
    for i in range(n_bins):
        r_lo = i / n_bins
        r_hi = (i + 1) / n_bins
        ring_mask = mask & (norm_dist >= r_lo) & (norm_dist < r_hi)
        ring_area = ring_mask.sum() * PIXEL_SCALE_UM ** 2  # μm²

        if len(particle_df) > 0 and ring_area > 0:
            # Count particles whose centroid falls in this ring
            p_dist = np.sqrt((particle_df['centroid_y'] - cy) ** 2 +
                             (particle_df['centroid_x'] - cx) ** 2)
            p_norm = p_dist / max(equiv_radius, 1)
            n_particles = ((p_norm >= r_lo) & (p_norm < r_hi)).sum()
            density = n_particles / (ring_area * 1e-6)  # particles/mm²
        else:
            n_particles = 0
            density = 0.0

        densities.append({
            'radius_bin': i,
            'r_center': (r_lo + r_hi) / 2,
            'particle_density': density,
            'n_particles': n_particles,
            'ring_area_um2': ring_area,
        })

    return pd.DataFrame(densities)


def compute_ring_metrics(profile_df, particle_density_df):
    """Extract ring-characterizing metrics from radial profiles."""
    # Smooth the intensity profile
    intensities = profile_df['mean_intensity'].values
    valid = ~np.isnan(intensities)
    if valid.sum() < 5:
        return {}

    r = profile_df['r_center'].values
    smooth = np.copy(intensities)
    smooth[valid] = savgol_filter(intensities[valid], min(11, valid.sum() // 2 * 2 + 1), 2)

    # Ring peak position (where intensity is maximum)
    peak_idx = np.nanargmax(smooth)
    ring_peak_position = r[peak_idx]

    # Center-to-edge ratio
    center_mask = r < 0.3
    edge_mask = r > 0.7
    center_mean = np.nanmean(intensities[center_mask]) if center_mask.any() else np.nan
    edge_mean = np.nanmean(intensities[edge_mask]) if edge_mask.any() else np.nan
    center_to_edge = center_mean / edge_mean if edge_mean > 0 else np.nan

    # Ring width (FWHM of the peak)
    half_max = (np.nanmax(smooth) + np.nanmin(smooth[valid])) / 2
    above_half = smooth > half_max
    transitions = np.diff(above_half.astype(int))
    rises = np.where(transitions == 1)[0]
    falls = np.where(transitions == -1)[0]
    if len(rises) > 0 and len(falls) > 0:
        ring_width = r[falls[-1]] - r[rises[0]]
    else:
        ring_width = np.nan

    # Ring uniformity: std of intensity along the peak radius band
    peak_band = (r > ring_peak_position - 0.05) & (r < ring_peak_position + 0.05)
    if peak_band.any():
        ring_uniformity_cv = np.nanstd(intensities[peak_band]) / (np.nanmean(intensities[peak_band]) + 1e-10)
    else:
        ring_uniformity_cv = np.nan

    # Particle density gradient
    pd_vals = particle_density_df['particle_density'].values
    if len(pd_vals) > 10:
        edge_density = np.mean(pd_vals[-10:])
        center_density = np.mean(pd_vals[:10])
        density_gradient = edge_density - center_density
    else:
        density_gradient = np.nan

    return {
        'ring_peak_position': ring_peak_position,
        'ring_width': ring_width,
        'center_to_edge_ratio': center_to_edge,
        'ring_uniformity_cv': ring_uniformity_cv,
        'density_gradient': density_gradient,
    }


# ======================================================================
# Main
# ======================================================================

def run_analysis():
    print("=" * 60)
    print("Analysis 2: Radial Distribution Analysis")
    print("=" * 60)

    all_profiles = []
    all_metrics = []

    for cond, img_num, sample_id, sem_gray in tqdm(
        list(iterate_dataset()), desc="Radial analysis", unit="img"
    ):
        mask, contour, center, equiv_r = get_deposit_mask(sem_gray)
        if equiv_r < 50:
            continue

        # Radial intensity profile
        profile = compute_radial_profile(sem_gray, mask, center, equiv_r)
        profile['condition'] = cond
        profile['image_num'] = img_num

        # Particle detection for density
        labeled = detect_particles(sem_gray, mask)
        particles = extract_particle_properties(labeled)

        # Radial particle density
        density = compute_radial_particle_density(particles, center, equiv_r, mask)
        profile = profile.merge(
            density[['radius_bin', 'particle_density', 'n_particles']],
            on='radius_bin', how='left'
        )
        all_profiles.append(profile)

        # Ring metrics
        metrics = compute_ring_metrics(profile, density)
        metrics.update({
            'condition': cond, 'image_num': img_num, 'sample_id': sample_id,
            'deposit_area_fraction': mask.sum() / mask.size,
        })
        all_metrics.append(metrics)

    df_profiles = pd.concat(all_profiles, ignore_index=True)
    df_metrics = pd.DataFrame(all_metrics)

    return df_profiles, df_metrics


def plot_radial(df_profiles, df_metrics):
    """Publication figure: Radial distribution (2×2 layout)."""
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    temp_colors = {'20-23': '#1f77b4', '23-26': '#ff7f0e', '26-29': '#2ca02c'}

    # (a) Mean radial intensity profiles by condition
    ax = fig.add_subplot(gs[0, 0])
    for cond in CONDITIONS:
        subset = df_profiles[df_profiles['condition'] == cond]
        if len(subset) == 0:
            continue
        grouped = subset.groupby('r_center')['mean_intensity'].agg(['mean', 'std'])
        color = temp_colors[CONDITION_INFO[cond]['temp']]
        ax.plot(grouped.index, grouped['mean'], color=color, alpha=0.6, linewidth=1.0, label=cond)
    ax.set_xlabel('Normalized Radius')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('(a) Radial Intensity Profiles')
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlim(0, 1.2)
    style_axis(ax)

    # (b) Mean radial particle density
    ax = fig.add_subplot(gs[0, 1])
    for cond in CONDITIONS:
        subset = df_profiles[df_profiles['condition'] == cond]
        if len(subset) == 0:
            continue
        grouped = subset.groupby('r_center')['particle_density'].agg(['mean'])
        color = temp_colors[CONDITION_INFO[cond]['temp']]
        ax.plot(grouped.index, grouped['mean'], color=color, alpha=0.6, linewidth=1.0, label=cond)
    ax.set_xlabel('Normalized Radius')
    ax.set_ylabel('Particle Density (particles/mm²)')
    ax.set_title('(b) Radial Particle Density')
    ax.legend(fontsize=7, ncol=3)
    ax.set_xlim(0, 1.2)
    style_axis(ax)

    # (c) Box plots: center-to-edge ratio by condition
    ax = fig.add_subplot(gs[1, 0])
    data_by_cond = []
    for cond in CONDITIONS:
        vals = df_metrics[df_metrics['condition'] == cond]['center_to_edge_ratio'].dropna()
        data_by_cond.append(vals.values if len(vals) > 0 else [])
    bp = ax.boxplot(data_by_cond, labels=CONDITIONS, patch_artist=True, widths=0.6)
    for i, (patch, cond) in enumerate(zip(bp['boxes'], CONDITIONS)):
        patch.set_facecolor(temp_colors[CONDITION_INFO[cond]['temp']])
        patch.set_alpha(0.6)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Center / Edge Intensity Ratio')
    ax.set_title('(c) Coffee Ring Strength')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    style_axis(ax)

    # (d) Scatter: ring_peak_position vs humidity
    ax = fig.add_subplot(gs[1, 1])
    rh_map = {'35-40': 37.5, '40-45': 42.5, '45-50': 47.5}
    for cond in CONDITIONS:
        subset = df_metrics[df_metrics['condition'] == cond]
        if len(subset) == 0:
            continue
        rh_val = rh_map[CONDITION_INFO[cond]['rh']]
        color = temp_colors[CONDITION_INFO[cond]['temp']]
        ax.scatter(
            [rh_val] * len(subset) + np.random.normal(0, 0.3, len(subset)),
            subset['ring_peak_position'],
            c=color, alpha=0.4, s=20, edgecolors='k', linewidth=0.2
        )
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f'{t}°C') for t, c in temp_colors.items()]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Ring Peak Position (normalized)')
    ax.set_title('(d) Ring Peak vs Environment')
    style_axis(ax)

    fig.suptitle('Radial Distribution Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_radial_distribution")
    plt.close(fig)


if __name__ == '__main__':
    df_profiles, df_metrics = run_analysis()

    df_profiles.to_csv(EXPERIMENT_DIR / "radial_profiles.csv", index=False)
    print(f"\nSaved radial_profiles.csv ({len(df_profiles)} rows)")

    df_metrics.to_csv(EXPERIMENT_DIR / "radial_metrics.csv", index=False)
    print(f"Saved radial_metrics.csv ({len(df_metrics)} rows)")

    print("\nGenerating figures...")
    plot_radial(df_profiles, df_metrics)

    # Summary
    print("\nPer-Condition Ring Metrics:")
    for cond in CONDITIONS:
        s = df_metrics[df_metrics['condition'] == cond]
        if len(s) > 0:
            print(f"  {cond}: peak={s['ring_peak_position'].mean():.3f}, "
                  f"C/E={s['center_to_edge_ratio'].mean():.3f}, "
                  f"width={s['ring_width'].mean():.3f}")

    print("\nDone.")
