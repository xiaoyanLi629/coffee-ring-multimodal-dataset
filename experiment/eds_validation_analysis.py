#!/usr/bin/env python3
"""EDS validation analysis:
1. Confirm all 7 expected elements are detected across all 225 sample sets
2. Quantify within-condition reproducibility (CV%) of EDS signals
3. Examine how S signal varies with Na2SO4 concentration (highest variation across samples)
4. Generate summary figure and statistics for manuscript
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'figure.dpi': 300,
})

data_root = Path('/Volumes/XiaoyanSSD/Data/chapter_3_data/data/eds')
conditions = list('ABCDEFGHI')

element_files = {
    'C':  'C Kα1_2.png',
    'Ca': 'Ca Kα1.png',
    'Cl': 'Cl Kα1.png',
    'Mg': 'Mg Kα1_2.png',
    'Na': 'Na Kα1_2.png',
    'O':  'O Kα1.png',
    'S':  'S Kα1.png',
}

def get_sample_id(img_num):
    return (img_num - 1) // 5 + 1

def compute_eds_signal(fpath):
    """Return fraction of foreground pixels (element coverage) and mean signal."""
    img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    total_px = gray.size
    fg_mask = gray > 5
    coverage = fg_mask.sum() / total_px  # fraction of pixels with signal
    mean_sig = float(gray[fg_mask].mean()) if fg_mask.sum() > 50 else 0.0
    return coverage, mean_sig

# ── 1. Detection rate: confirm all elements are detected ─────────────────────
print("=== 1. Element Detection ===")
detection_records = []
for cond in conditions:
    for img_num in range(1, 26):
        sample_id = get_sample_id(img_num)
        eds_dir = data_root / cond / str(img_num)
        for elem, fname in element_files.items():
            fpath = eds_dir / fname
            coverage, mean_sig = compute_eds_signal(fpath)
            if coverage is not None:
                detection_records.append({
                    'condition': cond, 'image_num': img_num,
                    'sample_id': sample_id, 'element': elem,
                    'coverage': coverage, 'mean_signal': mean_sig,
                    'detected': coverage > 0.01  # >1% pixels active
                })

det_df = pd.DataFrame(detection_records)
det_rate = det_df.groupby('element')['detected'].mean() * 100
print("Detection rate per element (% of 225 sample sets):")
print(det_rate.round(1).to_string())
print(f"\nOverall: {det_df['detected'].mean()*100:.1f}% of element maps show positive signal")

# ── 2. Within-condition reproducibility ─────────────────────────────────────
print("\n=== 2. Within-condition Reproducibility (CV%) ===")
# For each condition and element, compute CV across 25 samples
cv_records = []
for cond in conditions:
    cond_data = det_df[det_df['condition'] == cond]
    for elem in element_files:
        vals = cond_data[cond_data['element'] == elem]['coverage'].values
        if len(vals) > 1 and vals.mean() > 0:
            cv = vals.std() / vals.mean() * 100
            cv_records.append({'condition': cond, 'element': elem, 'cv_pct': cv})

cv_df = pd.DataFrame(cv_records)
mean_cv_per_elem = cv_df.groupby('element')['cv_pct'].mean()
print("Mean within-condition CV% per element:")
print(mean_cv_per_elem.round(1).to_string())
overall_cv = cv_df['cv_pct'].mean()
print(f"\nOverall mean CV across all elements and conditions: {overall_cv:.1f}%")

# ── 3. S signal vs Na2SO4 concentration ─────────────────────────────────────
print("\n=== 3. S Signal vs Na2SO4 Concentration ===")
# Na2SO4 concentrations per sample
na2so4 = {1: 0.35, 2: 0.35, 3: 0.35, 4: 1.35, 5: 2.35}  # mM

s_by_sample = det_df[det_df['element'] == 'S'].groupby('sample_id')['coverage'].mean()
conc_vals = np.array([na2so4[s] for s in range(1, 6)])
cov_vals = s_by_sample.values

r, p = stats.pearsonr(conc_vals, cov_vals)
print(f"S coverage vs Na2SO4: r={r:.4f}, p={p:.4f}")
print(f"S coverage per sample: {dict(zip(range(1,6), cov_vals.round(4)))}")
print(f"Na2SO4 per sample: {na2so4}")

# Also check Ca vs CaCl2
ca2_conc = {1: 1.50, 2: 1.00, 3: 0.50, 4: 1.00, 5: 1.00}
ca_by_sample = det_df[det_df['element'] == 'Ca'].groupby('sample_id')['coverage'].mean()
r_ca, p_ca = stats.pearsonr(
    [ca2_conc[s] for s in range(1,6)], ca_by_sample.values)
print(f"\nCa coverage vs CaCl2: r={r_ca:.4f}, p={p_ca:.4f}")

# Mg vs MgCl2
mg_conc = {1: 0.50, 2: 0.35, 3: 0.20, 4: 1.00, 5: 0.50}
mg_by_sample = det_df[det_df['element'] == 'Mg'].groupby('sample_id')['coverage'].mean()
r_mg, p_mg = stats.pearsonr(
    [mg_conc[s] for s in range(1,6)], mg_by_sample.values)
print(f"Mg coverage vs MgCl2: r={r_mg:.4f}, p={p_mg:.4f}")

# ── 4. Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: element coverage heatmap (condition × element)
cov_matrix = det_df.groupby(['condition', 'element'])['coverage'].mean().unstack()
cov_matrix = cov_matrix[list(element_files.keys())]  # reorder columns

im = axes[0].imshow(cov_matrix.values, cmap='YlOrRd', aspect='auto',
                    vmin=0, vmax=cov_matrix.values.max())
axes[0].set_xticks(range(len(element_files)))
axes[0].set_xticklabels(list(element_files.keys()), fontweight='bold')
axes[0].set_yticks(range(9))
axes[0].set_yticklabels(list('ABCDEFGHI'))
axes[0].set_xlabel('Element')
axes[0].set_ylabel('Condition')
axes[0].set_title('Mean EDS Signal Coverage\n(fraction of active pixels)', pad=8)
for i in range(9):
    for j in range(len(element_files)):
        val = cov_matrix.values[i, j]
        axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=8.5, color='black' if val < 0.5 else 'white')
plt.colorbar(im, ax=axes[0], label='Coverage fraction')

# Right: CV% by element
cv_summary = cv_df.groupby('element')['cv_pct'].agg(['mean', 'std']).reset_index()
cv_summary = cv_summary.set_index('element').reindex(list(element_files.keys())).reset_index()
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cv_summary)))
bars = axes[1].bar(cv_summary['element'], cv_summary['mean'],
                   color=colors, edgecolor='gray', linewidth=0.5)
axes[1].errorbar(range(len(cv_summary)), cv_summary['mean'],
                 yerr=cv_summary['std'], fmt='none', color='black',
                 capsize=4, linewidth=1.2)
axes[1].set_xlabel('Element')
axes[1].set_ylabel('Within-condition CV (%)')
axes[1].set_title(f'EDS Reproducibility\n(mean CV = {overall_cv:.1f}%)', pad=8)
axes[1].axhline(y=overall_cv, color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {overall_cv:.1f}%')
axes[1].legend()
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_xticks(range(len(cv_summary)))
axes[1].set_xticklabels(cv_summary['element'], fontweight='bold')

plt.tight_layout()
fig.savefig('eds_validation.png', dpi=300, bbox_inches='tight')
fig.savefig('eds_validation.svg', format='svg', bbox_inches='tight')
print('\nFigure saved: eds_validation.png')

# Save results
det_rate.reset_index(name='detection_rate_pct').to_csv(
    'eds_detection_rates.csv', index=False)
cv_df.to_csv('eds_reproducibility.csv', index=False)
print("Results saved: eds_detection_rates.csv, eds_reproducibility.csv")
print(f"\n=== SUMMARY FOR MANUSCRIPT ===")
print(f"Detection rate: {det_df['detected'].mean()*100:.0f}% of element maps show positive signal")
print(f"Overall mean within-condition CV: {overall_cv:.1f}%")
print(f"S vs Na2SO4: r={r:.4f}, p={p:.4f}")
print(f"Ca vs CaCl2: r={r_ca:.4f}, p={p_ca:.4f}")
print(f"Mg vs MgCl2: r={r_mg:.4f}, p={p_mg:.4f}")
