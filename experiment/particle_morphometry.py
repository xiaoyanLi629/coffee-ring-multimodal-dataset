#!/usr/bin/env python3
"""
Analysis 1: Single-Particle Morphometry from SEM Images

Extracts per-particle morphological features (area, perimeter, equivalent
diameter, Feret diameter, circularity, solidity, aspect ratio, eccentricity)
from 225 SEM images and produces summary statistics per image and per condition.

Outputs:
  - particle_level_features.csv   (one row per particle)
  - particle_summary_features.csv (one row per image, 225 rows)
  - figure_particle_morphometry.png / .svg
  - figure_particle_segmentation_examples.png / .svg
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import (
    iterate_dataset, get_deposit_mask, detect_particles,
    extract_particle_properties, save_figure, load_sem_gray,
    CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR, PIXEL_SCALE_UM,
    get_sample_id
)

setup_professional_style()
COLORS = get_professional_colors()


# ======================================================================
# 1. Extract particles from all 225 SEM images
# ======================================================================

def run_extraction():
    """Extract particles from all SEM images, return DataFrames."""
    all_particles = []
    all_summaries = []

    print("=" * 60)
    print("Analysis 1: Single-Particle Morphometry")
    print("=" * 60)

    for cond, img_num, sample_id, sem_gray in tqdm(
        list(iterate_dataset()), desc="Processing SEM images", unit="img"
    ):
        # 1. Deposit segmentation
        mask, contour, center, equiv_r = get_deposit_mask(sem_gray)
        if equiv_r < 50:
            # Skip images where no deposit is detected
            print(f"  [WARN] {cond}/SEM_{img_num}: deposit too small, skipping")
            continue

        # 2. Particle detection
        labeled = detect_particles(sem_gray, mask)
        n_particles = labeled.max()

        # 3. Per-particle properties
        df_particles = extract_particle_properties(labeled)

        if len(df_particles) == 0:
            # Record zero-particle summary
            all_summaries.append({
                'condition': cond, 'image_num': img_num,
                'sample_id': sample_id, 'particle_count': 0,
            })
            continue

        # Tag with metadata
        df_particles['condition'] = cond
        df_particles['image_num'] = img_num
        df_particles['sample_id'] = sample_id
        all_particles.append(df_particles)

        # 4. Per-image summary statistics
        summary = {
            'condition': cond,
            'image_num': img_num,
            'sample_id': sample_id,
            'particle_count': len(df_particles),
            # Size
            'area_mean': df_particles['area_um2'].mean(),
            'area_median': df_particles['area_um2'].median(),
            'area_std': df_particles['area_um2'].std(),
            'area_skew': df_particles['area_um2'].skew(),
            'equiv_diameter_mean': df_particles['equiv_diameter_um'].mean(),
            'equiv_diameter_median': df_particles['equiv_diameter_um'].median(),
            'feret_diameter_mean': df_particles['feret_diameter_um'].mean(),
            'feret_diameter_median': df_particles['feret_diameter_um'].median(),
            # Shape
            'circularity_mean': df_particles['circularity'].mean(),
            'circularity_std': df_particles['circularity'].std(),
            'solidity_mean': df_particles['solidity'].mean(),
            'solidity_std': df_particles['solidity'].std(),
            'aspect_ratio_mean': df_particles['aspect_ratio'].mean(),
            'aspect_ratio_std': df_particles['aspect_ratio'].std(),
            'eccentricity_mean': df_particles['eccentricity'].mean(),
            'eccentricity_std': df_particles['eccentricity'].std(),
            # Deposit info
            'deposit_area_px': int(mask.sum()),
            'deposit_equiv_radius': equiv_r,
            'deposit_center_y': center[0],
            'deposit_center_x': center[1],
        }
        all_summaries.append(summary)

    df_all = pd.concat(all_particles, ignore_index=True) if all_particles else pd.DataFrame()
    df_summary = pd.DataFrame(all_summaries)

    return df_all, df_summary


# ======================================================================
# 2. Generate segmentation example figure
# ======================================================================

def plot_segmentation_examples(df_summary):
    """
    Show segmentation results for representative samples from
    conditions A, E, I (spanning the factorial design).
    """
    examples = [('A', 3), ('E', 13), ('I', 23)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, (cond, img_num) in enumerate(examples):
        sem = load_sem_gray(cond, img_num, corrected=True)
        mask, contour, center, equiv_r = get_deposit_mask(sem)
        labeled = detect_particles(sem, mask)
        n = labeled.max()

        # Top row: original SEM
        ax = axes[0, col]
        ax.imshow(sem, cmap='gray')
        ax.set_title(f"Condition {cond}, Image {img_num}", fontsize=11)
        ax.axis('off')

        # Bottom row: segmentation overlay
        ax = axes[1, col]
        # Create RGB overlay
        rgb = np.stack([sem, sem, sem], axis=-1).astype(float) / 255
        # Color particles
        overlay = rgb.copy()
        for lbl in range(1, labeled.max() + 1):
            color = plt.cm.Set3(lbl % 12)[:3]
            overlay[labeled == lbl] = color
        blended = 0.5 * rgb + 0.5 * overlay
        # Draw deposit contour
        if len(contour) > 0:
            ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.0, alpha=0.7)
        ax.imshow(blended)
        ax.set_title(f"N = {n} particles", fontsize=11)
        ax.axis('off')

    fig.suptitle("Particle Segmentation Examples", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, "figure_particle_segmentation_examples")
    plt.close(fig)


# ======================================================================
# 3. Main morphometry figure (2×3 layout)
# ======================================================================

def plot_morphometry(df_all, df_summary):
    """
    Publication figure: particle morphometry overview.
    (a) Violin: particle count by condition
    (b) Box: area distribution by water sample
    (c) Scatter: circularity vs solidity by condition
    (d) Histogram: log(area) by temperature group
    (e) Heatmap: condition × shape metrics
    (f) Bar: Feret/equiv diameter ratio by condition
    """
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    temp_colors = {'20-23': '#1f77b4', '23-26': '#ff7f0e', '26-29': '#2ca02c'}
    cond_colors = [plt.cm.Set3(i / 9) for i in range(9)]

    df_s = df_summary[df_summary['particle_count'] > 0].copy()

    # (a) Violin: particle count by condition
    ax = fig.add_subplot(gs[0, 0])
    for i, cond in enumerate(CONDITIONS):
        data = df_s[df_s['condition'] == cond]['particle_count']
        if len(data) > 0:
            vp = ax.violinplot(data, positions=[i], showmeans=True, showmedians=True, widths=0.7)
            for pc in vp['bodies']:
                tc = temp_colors[CONDITION_INFO[cond]['temp']]
                pc.set_facecolor(tc)
                pc.set_alpha(0.6)
    ax.set_xticks(range(9))
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel('Condition')
    ax.set_ylabel('Particle Count')
    ax.set_title('(a) Particle Count by Condition')
    style_axis(ax)

    # (b) Box: area by water sample
    ax = fig.add_subplot(gs[0, 1])
    sample_data = [df_s[df_s['sample_id'] == s]['area_mean'].dropna()
                   for s in range(1, 6)]
    bp = ax.boxplot(sample_data, labels=[f'S{i}' for i in range(1, 6)],
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], plt.cm.Pastel1(np.linspace(0, 1, 5))):
        patch.set_facecolor(color)
    ax.set_xlabel('Water Sample')
    ax.set_ylabel('Mean Particle Area (μm²)')
    ax.set_title('(b) Particle Area by Water Sample')
    style_axis(ax)

    # (c) Scatter: circularity vs solidity
    ax = fig.add_subplot(gs[0, 2])
    for i, cond in enumerate(CONDITIONS):
        subset = df_s[df_s['condition'] == cond]
        ax.scatter(subset['circularity_mean'], subset['solidity_mean'],
                   c=[cond_colors[i]], label=cond, s=30, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Mean Circularity')
    ax.set_ylabel('Mean Solidity')
    ax.set_title('(c) Circularity vs Solidity')
    ax.legend(fontsize=7, ncol=3, loc='lower left')
    style_axis(ax)

    # (d) Histogram: log(area) by temperature group
    ax = fig.add_subplot(gs[1, 0])
    for temp_label, color in temp_colors.items():
        conds_in_temp = [c for c in CONDITIONS if CONDITION_INFO[c]['temp'] == temp_label]
        areas = df_all[df_all['condition'].isin(conds_in_temp)]['area_um2']
        if len(areas) > 0:
            log_areas = np.log10(areas.clip(lower=1))
            ax.hist(log_areas, bins=40, alpha=0.5, color=color,
                    label=f'{temp_label}°C', density=True, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('log₁₀(Area / μm²)')
    ax.set_ylabel('Density')
    ax.set_title('(d) Particle Size Distribution')
    ax.legend(fontsize=9)
    style_axis(ax)

    # (e) Heatmap: condition × shape metrics
    ax = fig.add_subplot(gs[1, 1])
    metrics = ['circularity_mean', 'solidity_mean', 'aspect_ratio_mean',
               'eccentricity_mean', 'equiv_diameter_mean', 'feret_diameter_mean']
    metric_labels = ['Circularity', 'Solidity', 'Aspect Ratio',
                     'Eccentricity', 'D_eq (μm)', 'D_Feret (μm)']
    hm_data = []
    for cond in CONDITIONS:
        row = []
        for m in metrics:
            vals = df_s[df_s['condition'] == cond][m]
            row.append(vals.mean() if len(vals) > 0 else np.nan)
        hm_data.append(row)
    hm_df = pd.DataFrame(hm_data, index=CONDITIONS, columns=metric_labels)
    # Normalize each column to [0,1] for visualization
    hm_norm = (hm_df - hm_df.min()) / (hm_df.max() - hm_df.min() + 1e-10)
    sns.heatmap(hm_norm, ax=ax, cmap='YlOrRd', annot=hm_df.round(2),
                fmt='', linewidths=0.5, cbar_kws={'label': 'Normalized', 'shrink': 0.8})
    ax.set_title('(e) Shape Metrics Heatmap')
    ax.set_ylabel('Condition')

    # (f) Bar: Feret/equiv diameter ratio (measures elongation)
    ax = fig.add_subplot(gs[1, 2])
    ratios = []
    for cond in CONDITIONS:
        subset = df_s[df_s['condition'] == cond]
        if len(subset) > 0 and subset['equiv_diameter_mean'].mean() > 0:
            r = subset['feret_diameter_mean'].mean() / subset['equiv_diameter_mean'].mean()
        else:
            r = np.nan
        ratios.append(r)
    bars = ax.bar(range(9), ratios, color=[temp_colors[CONDITION_INFO[c]['temp']] for c in CONDITIONS],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(9))
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel('Condition')
    ax.set_ylabel('D_Feret / D_eq')
    ax.set_title('(f) Elongation Ratio')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    # Legend for temperature colors
    handles = [Patch(facecolor=c, label=f'{t}°C') for t, c in temp_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    style_axis(ax)

    fig.suptitle('Particle Morphometry Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_particle_morphometry")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    # Extract all particle data
    df_all, df_summary = run_extraction()

    # Save CSVs
    out_particles = EXPERIMENT_DIR / "particle_level_features.csv"
    out_summary = EXPERIMENT_DIR / "particle_summary_features.csv"

    if len(df_all) > 0:
        df_all.to_csv(out_particles, index=False)
        print(f"\nSaved particle-level features: {out_particles}")
        print(f"  Total particles: {len(df_all)}")
    else:
        print("\n[WARN] No particles extracted.")

    df_summary.to_csv(out_summary, index=False)
    print(f"Saved per-image summaries: {out_summary}")
    print(f"  Images processed: {len(df_summary)}")

    # Generate figures
    if len(df_summary[df_summary.get('particle_count', 0) > 0]) > 0:
        print("\nGenerating figures...")
        plot_segmentation_examples(df_summary)
        plot_morphometry(df_all, df_summary)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Per-Condition Summary")
    print("=" * 60)
    for cond in CONDITIONS:
        s = df_summary[df_summary['condition'] == cond]
        if len(s) > 0 and 'particle_count' in s.columns:
            sc = s[s['particle_count'] > 0]
            if len(sc) > 0:
                print(f"  {cond} ({CONDITION_INFO[cond]['temp']}°C, "
                      f"{CONDITION_INFO[cond]['rh']}% RH): "
                      f"N={sc['particle_count'].mean():.0f}±{sc['particle_count'].std():.0f}, "
                      f"D_eq={sc['equiv_diameter_mean'].mean():.1f}μm, "
                      f"Circ={sc['circularity_mean'].mean():.3f}, "
                      f"Solid={sc['solidity_mean'].mean():.3f}")
            else:
                print(f"  {cond}: no particles detected")

    print("\nDone.")
