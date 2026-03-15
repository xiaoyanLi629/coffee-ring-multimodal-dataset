#!/usr/bin/env python3
"""
Analysis 6: Information-Theoretic EDS Analysis

Computes Jensen-Shannon and Kullback-Leibler divergences between EDS
elemental composition vectors. Performs MDS dimensionality reduction
on the divergence distance matrix.

Outputs:
  - eds_composition_vectors.csv
  - eds_divergence_summary.csv
  - figure_eds_divergence.png / .svg
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import (
    CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR,
    EDS_DIR, EDS_ELEMENTS, EDS_ELEMENT_NAMES,
    IMAGES_PER_CONDITION, get_sample_id, save_figure
)

setup_professional_style()


def load_eds_coverage(condition, image_num):
    """Load all 7 EDS maps and compute signal coverage fraction for each element."""
    coverages = {}
    for elem_file, elem_name in zip(EDS_ELEMENTS, EDS_ELEMENT_NAMES):
        path = EDS_DIR / condition / str(image_num) / f"{elem_file}.png"
        if not path.exists():
            coverages[elem_name] = 0.0
            continue
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            coverages[elem_name] = 0.0
            continue
        # Convert to grayscale signal
        if len(img.shape) == 3:
            gray = np.max(img[:, :, :3], axis=2)  # max of RGB channels
        else:
            gray = img
        # Threshold: signal > 5 pixel value
        signal = (gray > 5).sum()
        total = gray.size
        coverages[elem_name] = signal / total

    return coverages


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize to probability distributions
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return np.nan
    p = p / p_sum
    q = q / q_sum

    m = 0.5 * (p + q)

    # KL divergence with safety for log(0)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ======================================================================
# Main
# ======================================================================

def run_analysis():
    print("=" * 60)
    print("Analysis 6: EDS Information-Theoretic Analysis")
    print("=" * 60)

    # 1. Compute composition vectors for all 225 samples
    all_records = []
    print("Loading EDS maps...")
    for cond in tqdm(CONDITIONS, desc="Conditions"):
        for img_num in range(1, IMAGES_PER_CONDITION + 1):
            coverages = load_eds_coverage(cond, img_num)
            record = {
                'condition': cond,
                'image_num': img_num,
                'sample_id': get_sample_id(img_num),
            }
            record.update({f'{e}_coverage': coverages[e] for e in EDS_ELEMENT_NAMES})
            all_records.append(record)

    df_comp = pd.DataFrame(all_records)
    n = len(df_comp)
    print(f"Loaded {n} composition vectors")

    # 2. Compute pairwise JS divergence matrix
    print("Computing JS divergence matrix...")
    elem_cols = [f'{e}_coverage' for e in EDS_ELEMENT_NAMES]
    vectors = df_comp[elem_cols].values

    js_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="JS divergence"):
        for j in range(i + 1, n):
            d = js_divergence(vectors[i], vectors[j])
            js_matrix[i, j] = d
            js_matrix[j, i] = d

    # 3. MDS on divergence matrix
    print("Running MDS...")
    # Replace NaN with max divergence
    max_d = np.nanmax(js_matrix)
    js_clean = np.nan_to_num(js_matrix, nan=max_d)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(js_clean)
    df_comp['mds_x'] = coords[:, 0]
    df_comp['mds_y'] = coords[:, 1]

    # 4. Within- vs between-group divergences
    print("Computing group divergences...")
    conditions_arr = df_comp['condition'].values
    samples_arr = df_comp['sample_id'].values

    within_cond, between_cond = [], []
    within_sample, between_sample = [], []

    for i in range(n):
        for j in range(i + 1, n):
            d = js_matrix[i, j]
            if np.isnan(d):
                continue
            if conditions_arr[i] == conditions_arr[j]:
                within_cond.append(d)
            else:
                between_cond.append(d)
            if samples_arr[i] == samples_arr[j]:
                within_sample.append(d)
            else:
                between_sample.append(d)

    df_summary = pd.DataFrame([{
        'within_condition_mean_JS': np.mean(within_cond),
        'within_condition_std_JS': np.std(within_cond),
        'between_condition_mean_JS': np.mean(between_cond),
        'between_condition_std_JS': np.std(between_cond),
        'within_sample_mean_JS': np.mean(within_sample),
        'within_sample_std_JS': np.std(within_sample),
        'between_sample_mean_JS': np.mean(between_sample),
        'between_sample_std_JS': np.std(between_sample),
    }])

    return df_comp, js_matrix, df_summary


def plot_eds_divergence(df_comp, js_matrix, df_summary):
    """Publication figure: EDS divergence analysis (2×2 layout)."""
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    cond_colors = {c: plt.cm.Set3(i / 9) for i, c in enumerate(CONDITIONS)}
    sample_markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}

    # (a) MDS scatter plot
    ax = fig.add_subplot(gs[0, 0])
    for cond in CONDITIONS:
        subset = df_comp[df_comp['condition'] == cond]
        ax.scatter(subset['mds_x'], subset['mds_y'], c=[cond_colors[cond]],
                   label=cond, s=25, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title('(a) MDS of EDS Composition (by Condition)')
    ax.legend(fontsize=7, ncol=3)
    style_axis(ax)

    # (b) Mean pairwise JS divergence between conditions (9×9 heatmap)
    ax = fig.add_subplot(gs[0, 1])
    cond_mean_js = np.zeros((9, 9))
    conditions_arr = df_comp['condition'].values
    for i, ci in enumerate(CONDITIONS):
        for j, cj in enumerate(CONDITIONS):
            mask_i = conditions_arr == ci
            mask_j = conditions_arr == cj
            sub_matrix = js_matrix[np.ix_(mask_i, mask_j)]
            cond_mean_js[i, j] = np.nanmean(sub_matrix)
    sns.heatmap(cond_mean_js, ax=ax, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=CONDITIONS, yticklabels=CONDITIONS,
                cbar_kws={'label': 'Mean JS Divergence', 'shrink': 0.8})
    ax.set_title('(b) Inter-Condition JS Divergence')

    # (c) Within vs between group JS divergence
    ax = fig.add_subplot(gs[1, 0])
    box_data = {
        'Within\nCondition': [js_matrix[i, j] for i in range(len(df_comp))
                              for j in range(i+1, len(df_comp))
                              if conditions_arr[i] == conditions_arr[j]
                              and not np.isnan(js_matrix[i, j])],
        'Between\nCondition': np.random.choice(
            [js_matrix[i, j] for i in range(len(df_comp))
             for j in range(i+1, len(df_comp))
             if conditions_arr[i] != conditions_arr[j]
             and not np.isnan(js_matrix[i, j])],
            size=min(2000, len(df_comp) * (len(df_comp) - 1) // 2), replace=False
        ).tolist() if len(df_comp) > 1 else [],
    }
    bp = ax.boxplot([box_data['Within\nCondition'], box_data['Between\nCondition']],
                    labels=['Within\nCondition', 'Between\nCondition'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#1f77b4')
    bp['boxes'][1].set_facecolor('#ff7f0e')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('JS Divergence')
    ax.set_title('(c) Within vs Between Group Divergence')
    style_axis(ax)

    # (d) Stacked bar: elemental composition by water sample
    ax = fig.add_subplot(gs[1, 1])
    elem_cols = [f'{e}_coverage' for e in EDS_ELEMENT_NAMES]
    sample_means = df_comp.groupby('sample_id')[elem_cols].mean()
    # Normalize to sum=1 per sample
    sample_norm = sample_means.div(sample_means.sum(axis=1), axis=0)
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    bottom = np.zeros(5)
    for idx, (col, elem_name) in enumerate(zip(elem_cols, EDS_ELEMENT_NAMES)):
        vals = sample_norm[col].values
        ax.bar(range(1, 6), vals, bottom=bottom, color=colors[idx],
               label=elem_name, edgecolor='white', linewidth=0.5)
        bottom += vals
    ax.set_xlabel('Water Sample')
    ax.set_ylabel('Relative Coverage')
    ax.set_title('(d) Elemental Composition by Sample')
    ax.legend(fontsize=8, ncol=4, loc='upper right')
    ax.set_xticks(range(1, 6))
    style_axis(ax)

    fig.suptitle('EDS Information-Theoretic Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_eds_divergence")
    plt.close(fig)


if __name__ == '__main__':
    df_comp, js_matrix, df_summary = run_analysis()

    df_comp.to_csv(EXPERIMENT_DIR / "eds_composition_vectors.csv", index=False)
    print(f"\nSaved eds_composition_vectors.csv ({len(df_comp)} rows)")

    df_summary.to_csv(EXPERIMENT_DIR / "eds_divergence_summary.csv", index=False)
    print(f"Saved eds_divergence_summary.csv")

    print(f"\nWithin-condition JS: {df_summary['within_condition_mean_JS'].values[0]:.4f}")
    print(f"Between-condition JS: {df_summary['between_condition_mean_JS'].values[0]:.4f}")

    print("\nGenerating figures...")
    plot_eds_divergence(df_comp, js_matrix, df_summary)

    print("\nDone.")
