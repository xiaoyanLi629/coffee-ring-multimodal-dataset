#!/usr/bin/env python3
"""Compute per-condition validation statistics for manuscript Technical Validation section.
Outputs:
  - validation_statistics.csv: per-condition SSIM + key feature stats
  - feature_correlation_summary.csv: cross-modal correlations by domain
  - anova_results.csv: ANOVA effect sizes for temperature and humidity
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('comprehensive_computer_vision_features.csv')

# ── Condition metadata ────────────────────────────────────────────────────────
cond_info = {
    'condition_A': ('20-23', '35-40'),
    'condition_B': ('20-23', '40-45'),
    'condition_C': ('20-23', '45-50'),
    'condition_D': ('23-26', '35-40'),
    'condition_E': ('23-26', '40-45'),
    'condition_F': ('23-26', '45-50'),
    'condition_G': ('26-29', '35-40'),
    'condition_H': ('26-29', '40-45'),
    'condition_I': ('26-29', '45-50'),
}
df['temp'] = df['condition'].map(lambda c: cond_info[c][0])
df['hum'] = df['condition'].map(lambda c: cond_info[c][1])
df['cond'] = df['condition'].str.replace('condition_', '')

# ── 1. Per-condition SSIM + key feature statistics ────────────────────────────
rows = []
for cond in sorted(df['condition'].unique()):
    sub = df[df['condition'] == cond]
    label = cond.replace('condition_', '')
    temp, hum = cond_info[cond]

    ssim = sub['cross_modal_ssim']
    morph_corr = sub['normal_morph_num_objects'].corr(sub['sem_morph_num_objects'])
    texture_corr = sub[['normal_texture_contrast_d1_a0',
                         'normal_texture_energy_d1_a0',
                         'normal_texture_homogeneity_d1_a0']].corrwith(
        sub[['sem_texture_contrast_d1_a0',
             'sem_texture_energy_d1_a0',
             'sem_texture_homogeneity_d1_a0']]).mean()

    rows.append({
        'condition': label,
        'temperature_C': temp,
        'humidity_pct': hum,
        'ssim_mean': round(ssim.mean(), 4),
        'ssim_std': round(ssim.std(), 4),
        'ssim_median': round(ssim.median(), 4),
        'ssim_min': round(ssim.min(), 4),
        'ssim_max': round(ssim.max(), 4),
        'morphology_corr': round(morph_corr, 4),
        'texture_corr_mean': round(texture_corr, 4),
        'n_samples': len(sub)
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv('validation_statistics.csv', index=False)
print("=== Per-condition SSIM Statistics ===")
print(stats_df[['condition', 'temperature_C', 'humidity_pct',
                 'ssim_mean', 'ssim_std', 'ssim_median',
                 'ssim_min', 'ssim_max']].to_string(index=False))

# ── 2. Cross-modal feature correlations ──────────────────────────────────────
print("\n=== Cross-modal Feature Correlations ===")

# Morphological
morph_r = df['normal_morph_num_objects'].corr(df['sem_morph_num_objects'])
morph_area_r = df['normal_morph_largest_object_area'].corr(df['sem_morph_largest_object_area'])

# Texture (average across features)
texture_features = [c for c in df.columns if 'normal_texture' in c and 'contrast' in c]
sem_texture = [c.replace('normal_', 'sem_') for c in texture_features]
texture_corrs = [df[n].corr(df[s]) for n, s in zip(texture_features, sem_texture) if s in df.columns]
texture_mean_r = np.mean(texture_corrs)
texture_std_r = np.std(texture_corrs)

# Frequency
nf = df['normal_freq_total_spectral_energy']
sf = df['sem_freq_total_spectral_energy']
freq_r2 = nf.corr(sf) ** 2
freq_r = nf.corr(sf)

corr_summary = pd.DataFrame([
    {'domain': 'Morphological (object count)', 'metric': 'Pearson r', 'value': round(morph_r, 4)},
    {'domain': 'Morphological (largest area)', 'metric': 'Pearson r', 'value': round(morph_area_r, 4)},
    {'domain': 'Texture (mean across features)', 'metric': 'Pearson r (mean±SD)',
     'value': f"{texture_mean_r:.4f} ± {texture_std_r:.4f}"},
    {'domain': 'Frequency (total spectral energy)', 'metric': 'Pearson r', 'value': round(freq_r, 4)},
    {'domain': 'Frequency (total spectral energy)', 'metric': 'R²', 'value': round(freq_r2, 4)},
])
corr_summary.to_csv('feature_correlation_summary.csv', index=False)
print(corr_summary.to_string(index=False))

# ── 3. ANOVA: temperature and humidity effects ────────────────────────────────
print("\n=== ANOVA Effect Sizes ===")

temp_groups_texture = [df[df['temp'] == t]['normal_texture_contrast_d1_a0'].values
                       for t in ['20-23', '23-26', '26-29']]
hum_groups_freq = [df[df['hum'] == h]['normal_freq_total_spectral_energy'].values
                   for h in ['35-40', '40-45', '45-50']]

f_temp, p_temp = f_oneway(*temp_groups_texture)
f_hum, p_hum = f_oneway(*hum_groups_freq)

# Eta-squared
def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((v - grand_mean)**2 for v in all_vals)
    return ss_between / ss_total

eta2_temp = eta_squared(temp_groups_texture)
eta2_hum = eta_squared(hum_groups_freq)

anova_df = pd.DataFrame([
    {'factor': 'Temperature', 'feature_domain': 'Texture (contrast)',
     'F': round(f_temp, 3), 'p': f'<0.001' if p_temp < 0.001 else round(p_temp, 4),
     'eta_squared': round(eta2_temp, 4)},
    {'factor': 'Humidity', 'feature_domain': 'Frequency (spectral energy)',
     'F': round(f_hum, 3), 'p': f'<0.001' if p_hum < 0.001 else round(p_hum, 4),
     'eta_squared': round(eta2_hum, 4)},
])
anova_df.to_csv('anova_results.csv', index=False)
print(anova_df.to_string(index=False))

# ── 4. Replicate consistency (within-condition CV) ───────────────────────────
print("\n=== Within-condition Reproducibility (CV%) ===")
cv_ssim = (df.groupby('condition')['cross_modal_ssim'].std() /
           df.groupby('condition')['cross_modal_ssim'].mean() * 100).round(2)
print(cv_ssim.rename('CV%').to_string())
cv_mean = cv_ssim.mean()
print(f"\nMean CV across conditions: {cv_mean:.2f}%")

print("\nAll statistics saved to:")
print("  validation_statistics.csv")
print("  feature_correlation_summary.csv")
print("  anova_results.csv")
