#!/usr/bin/env python3
"""
Analysis 5: Advanced Multivariate Statistics

Performs PERMANOVA, MANOVA, and LDA/QDA classification on the combined
feature matrix from all previous analyses.

Depends on: particle_summary_features.csv, radial_metrics.csv,
            spatial_statistics.csv, fractal_dimensions.csv,
            comprehensive_computer_vision_features.csv

Outputs:
  - permanova_results.csv
  - manova_results.csv
  - discriminant_analysis_results.csv
  - figure_multivariate_statistics.png / .svg
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from professional_plot_style import setup_professional_style, style_axis, get_professional_colors
from particle_analysis_utils import CONDITIONS, CONDITION_INFO, EXPERIMENT_DIR, get_sample_id, save_figure

setup_professional_style()


def load_and_merge_features():
    """Load all feature CSVs and merge into a single matrix."""
    base = EXPERIMENT_DIR

    # Original 91 features
    df_cv = pd.read_csv(base / "comprehensive_computer_vision_features.csv")
    df_cv = df_cv.rename(columns={'image_id': 'image_num'})
    # Extract condition letter from "condition_A" format
    if df_cv['condition'].str.startswith('condition_').any():
        df_cv['condition'] = df_cv['condition'].str.replace('condition_', '')

    # Particle summary
    df_part = pd.read_csv(base / "particle_summary_features.csv")

    # Radial metrics
    df_rad = pd.read_csv(base / "radial_metrics.csv")

    # Spatial statistics
    df_spat = pd.read_csv(base / "spatial_statistics.csv")

    # Fractal dimensions
    df_frac = pd.read_csv(base / "fractal_dimensions.csv")

    # Merge all on (condition, image_num)
    df = df_cv.copy()
    for other in [df_part, df_rad, df_spat, df_frac]:
        if len(other) > 0:
            merge_cols = ['condition', 'image_num']
            other_cols = [c for c in other.columns if c not in df.columns or c in merge_cols]
            if len(other_cols) > len(merge_cols):
                df = df.merge(other[other_cols], on=merge_cols, how='left')

    # Add sample_id if not present
    if 'sample_id' not in df.columns:
        df['sample_id'] = df['image_num'].apply(get_sample_id)

    # Add temperature and humidity groups
    df['temp_group'] = df['condition'].map(lambda c: CONDITION_INFO.get(c, {}).get('temp', ''))
    df['rh_group'] = df['condition'].map(lambda c: CONDITION_INFO.get(c, {}).get('rh', ''))

    return df


def get_feature_sets(df):
    """Define different feature subsets for comparison."""
    # Original 91 CV features
    cv_cols = [c for c in df.columns if c.startswith(('normal_', 'sem_')) or c == 'cross_modal_ssim']

    # New particle-level features
    particle_cols = [c for c in df.columns if c.startswith(('particle_count', 'area_', 'equiv_',
                     'feret_', 'circularity_', 'solidity_', 'aspect_ratio_', 'eccentricity_',
                     'deposit_area', 'deposit_equiv'))]

    # Radial + spatial features
    radial_spatial_cols = [c for c in df.columns if c.startswith(('ring_', 'center_to_edge',
                          'density_gradient', 'clark_evans', 'nnd_', 'voronoi_', 'L_max',
                          'boundary_fractal', 'particle_fractal'))]

    # Combined
    all_feature_cols = cv_cols + particle_cols + radial_spatial_cols

    return {
        'Original (91 CV)': cv_cols,
        'Particle features': particle_cols,
        'Radial + Spatial': radial_spatial_cols,
        'Combined (all)': all_feature_cols,
    }


# ── PERMANOVA ──────────────────────────────────────────────────────────

def permanova(X, groups, n_perms=999):
    """
    Permutational MANOVA (Anderson, 2001).
    Returns pseudo-F, p-value, R².
    """
    X = np.array(X, dtype=float)
    groups = np.array(groups)
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    # Total sum of squares (from distance matrix)
    D = squareform(pdist(X, 'euclidean'))
    D2 = D ** 2

    SS_total = D2.sum() / (2 * n)

    # Within-group SS
    SS_within = 0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g > 1:
            SS_within += D2[np.ix_(mask, mask)].sum() / (2 * n_g)

    SS_between = SS_total - SS_within
    pseudo_F = (SS_between / (k - 1)) / (SS_within / (n - k)) if (n - k) > 0 and SS_within > 0 else 0
    R2 = SS_between / SS_total if SS_total > 0 else 0

    # Permutation test
    count = 1  # include observed
    for _ in range(n_perms):
        perm_groups = np.random.permutation(groups)
        SS_w_perm = 0
        for g in unique_groups:
            mask = perm_groups == g
            n_g = mask.sum()
            if n_g > 1:
                SS_w_perm += D2[np.ix_(mask, mask)].sum() / (2 * n_g)
        SS_b_perm = SS_total - SS_w_perm
        F_perm = (SS_b_perm / (k - 1)) / (SS_w_perm / (n - k)) if (n - k) > 0 and SS_w_perm > 0 else 0
        if F_perm >= pseudo_F:
            count += 1

    p_value = count / (n_perms + 1)

    return pseudo_F, p_value, R2


# ── MANOVA (via Wilks' Lambda approximation) ──────────────────────────

def manova_wilks(X, groups):
    """Compute Wilks' Lambda for MANOVA."""
    X = np.array(X, dtype=float)
    groups = np.array(groups)
    n, p = X.shape
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    grand_mean = X.mean(axis=0)

    # Within-group scatter matrix
    W = np.zeros((p, p))
    for g in unique_groups:
        Xg = X[groups == g]
        Xg_centered = Xg - Xg.mean(axis=0)
        W += Xg_centered.T @ Xg_centered

    # Total scatter matrix
    X_centered = X - grand_mean
    T = X_centered.T @ X_centered

    # Wilks' Lambda
    det_W = np.linalg.det(W)
    det_T = np.linalg.det(T)
    wilks_lambda = det_W / det_T if det_T != 0 else np.nan

    # F-approximation (Rao's approximation)
    if np.isnan(wilks_lambda) or wilks_lambda <= 0:
        return np.nan, np.nan, np.nan

    df1 = p * (k - 1)
    df2 = n - k

    # Simplified F approximation
    F_approx = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1) if wilks_lambda < 1 else 0

    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.f.cdf(F_approx, df1, df2) if F_approx > 0 else 1.0

    return wilks_lambda, F_approx, p_value


# ── LDA Classification ────────────────────────────────────────────────

def run_lda_classification(X, y, label=""):
    """Run LDA with leave-one-out cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle NaN
    nan_mask = np.isnan(X_scaled).any(axis=1)
    X_clean = X_scaled[~nan_mask]
    y_clean = y[~nan_mask]

    if len(np.unique(y_clean)) < 2 or len(X_clean) < 10:
        return np.nan

    # Reduce dimensions if needed (n_components <= min(n_classes-1, n_features))
    n_classes = len(np.unique(y_clean))
    n_components = min(n_classes - 1, X_clean.shape[1], X_clean.shape[0] - 1)

    if X_clean.shape[1] > X_clean.shape[0]:
        # Too many features — PCA first
        pca = PCA(n_components=min(X_clean.shape[0] - 2, 30))
        X_clean = pca.fit_transform(X_clean)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    scores = cross_val_score(lda, X_clean, y_clean, cv=LeaveOneOut(), scoring='accuracy')
    return scores.mean()


# ======================================================================
# Main
# ======================================================================

def run_analysis():
    print("=" * 60)
    print("Analysis 5: Multivariate Statistics")
    print("=" * 60)

    df = load_and_merge_features()
    print(f"Merged feature matrix: {df.shape[0]} samples × {df.shape[1]} columns")

    feature_sets = get_feature_sets(df)
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)} features")

    # ── PERMANOVA ──
    print("\nRunning PERMANOVA (999 permutations)...")
    permanova_results = []
    # Use combined feature set
    all_cols = feature_sets['Combined (all)']
    X_all = df[all_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    for factor_name, factor_col in [
        ('Condition (A-I)', 'condition'),
        ('Temperature', 'temp_group'),
        ('Humidity', 'rh_group'),
        ('Water sample', 'sample_id'),
    ]:
        groups = df[factor_col].values
        F, p, R2 = permanova(X_scaled, groups)
        permanova_results.append({
            'factor': factor_name, 'test': 'PERMANOVA',
            'statistic_name': 'pseudo-F', 'statistic_value': F,
            'p_value': p, 'R2': R2, 'n_permutations': 999,
        })
        print(f"  {factor_name}: F={F:.3f}, p={p:.4f}, R²={R2:.4f}")

    df_permanova = pd.DataFrame(permanova_results)

    # ── MANOVA ──
    print("\nRunning MANOVA...")
    manova_results = []
    # Use top 10 features by variance for MANOVA (avoids singular matrices)
    variances = np.var(X_scaled, axis=0)
    top_idx = np.argsort(variances)[-10:]
    X_manova = X_scaled[:, top_idx]

    for factor_name, factor_col in [
        ('Condition (A-I)', 'condition'),
        ('Temperature', 'temp_group'),
        ('Humidity', 'rh_group'),
        ('Water sample', 'sample_id'),
    ]:
        groups = df[factor_col].values
        wl, F_approx, p = manova_wilks(X_manova, groups)
        manova_results.append({
            'factor': factor_name, 'test': 'MANOVA',
            'statistic_name': "Wilks' Lambda", 'statistic_value': wl,
            'F_approx': F_approx, 'p_value': p,
        })
        print(f"  {factor_name}: Wilks'Λ={wl:.4f}, F={F_approx:.3f}, p={p:.4f}")

    df_manova = pd.DataFrame(manova_results)

    # ── LDA Classification ──
    print("\nRunning LDA classification (LOO-CV)...")
    lda_results = []
    targets = [
        ('Condition', 'condition'),
        ('Temperature', 'temp_group'),
        ('Humidity', 'rh_group'),
        ('Water sample', 'sample_id'),
    ]

    for fs_name, fs_cols in feature_sets.items():
        X_fs = df[fs_cols].fillna(0).values
        for target_name, target_col in targets:
            y = df[target_col].values
            acc = run_lda_classification(X_fs, y, label=f"{fs_name}/{target_name}")
            lda_results.append({
                'feature_set': fs_name,
                'target': target_name,
                'method': 'LDA',
                'accuracy': acc,
            })
            print(f"  {fs_name} → {target_name}: {acc:.3f}" if not np.isnan(acc) else f"  {fs_name} → {target_name}: N/A")

    df_lda = pd.DataFrame(lda_results)

    return df, df_permanova, df_manova, df_lda, feature_sets


def plot_multivariate(df, df_permanova, df_manova, df_lda, feature_sets):
    """Publication figure: Multivariate analysis (2×3 layout)."""
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # (a) LDA projection by condition
    ax = fig.add_subplot(gs[0, 0])
    all_cols = feature_sets['Combined (all)']
    X = df[all_cols].fillna(0).values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if X_s.shape[1] > X_s.shape[0]:
        pca = PCA(n_components=min(X_s.shape[0] - 2, 30))
        X_s = pca.fit_transform(X_s)
    y_cond = df['condition'].values
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_s, y_cond)
    cond_colors = {c: plt.cm.Set3(i / 9) for i, c in enumerate(CONDITIONS)}
    for cond in CONDITIONS:
        mask = y_cond == cond
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=[cond_colors[cond]],
                   label=cond, s=25, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_title('(a) LDA Projection (Condition)')
    ax.legend(fontsize=6, ncol=3, loc='best')
    style_axis(ax)

    # (b) Confusion matrix
    ax = fig.add_subplot(gs[0, 1])
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X_s, y_cond, cv=LeaveOneOut())
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_cond, y_pred, labels=CONDITIONS)
    sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONDITIONS, yticklabels=CONDITIONS,
                cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('(b) Confusion Matrix (LDA)')

    # (c) Classification accuracy comparison
    ax = fig.add_subplot(gs[0, 2])
    pivot = df_lda.pivot(index='target', columns='feature_set', values='accuracy')
    col_order = ['Original (91 CV)', 'Particle features', 'Radial + Spatial', 'Combined (all)']
    col_order = [c for c in col_order if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.plot(kind='bar', ax=ax, width=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('LOO-CV Accuracy')
    ax.set_title('(c) Classification Accuracy')
    ax.legend(fontsize=7, title='Feature Set', title_fontsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    style_axis(ax)

    # (d) PERMANOVA R² effect sizes
    ax = fig.add_subplot(gs[1, 0])
    factors = df_permanova['factor'].values
    R2_vals = df_permanova['R2'].values
    p_vals = df_permanova['p_value'].values
    bars = ax.barh(range(len(factors)), R2_vals, color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors)
    ax.set_xlabel('R² (Effect Size)')
    ax.set_title('(d) PERMANOVA Effect Sizes')
    for i, (r2, p) in enumerate(zip(R2_vals, p_vals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(r2 + 0.005, i, f'{r2:.3f} {sig}', va='center', fontsize=9)
    style_axis(ax)

    # (e) LDA projection by water sample
    ax = fig.add_subplot(gs[1, 1])
    y_sample = df['sample_id'].astype(str).values
    lda2 = LinearDiscriminantAnalysis(n_components=2)
    X_lda2 = lda2.fit_transform(X_s, y_sample)
    sample_colors = plt.cm.Set1(np.linspace(0, 1, 5))
    for s in range(1, 6):
        mask = y_sample == str(s)
        ax.scatter(X_lda2[mask, 0], X_lda2[mask, 1], c=[sample_colors[s - 1]],
                   label=f'Sample {s}', s=25, alpha=0.7, edgecolors='k', linewidth=0.3)
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_title('(e) LDA Projection (Water Sample)')
    ax.legend(fontsize=8)
    style_axis(ax)

    # (f) Summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    table_data = []
    for _, row in df_permanova.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
        table_data.append([row['factor'], f"{row['statistic_value']:.2f}", f"{row['R2']:.3f}", sig])
    table = ax.table(cellText=table_data,
                     colLabels=['Factor', 'pseudo-F', 'R²', 'Sig.'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax.set_title('(f) PERMANOVA Summary', fontsize=12, fontweight='bold', pad=20)

    fig.suptitle('Multivariate Statistical Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "figure_multivariate_statistics")
    plt.close(fig)


if __name__ == '__main__':
    df, df_perm, df_manova, df_lda, feature_sets = run_analysis()

    df_perm.to_csv(EXPERIMENT_DIR / "permanova_results.csv", index=False)
    df_manova.to_csv(EXPERIMENT_DIR / "manova_results.csv", index=False)
    df_lda.to_csv(EXPERIMENT_DIR / "discriminant_analysis_results.csv", index=False)
    print(f"\nSaved: permanova_results.csv, manova_results.csv, discriminant_analysis_results.csv")

    print("\nGenerating figures...")
    plot_multivariate(df, df_perm, df_manova, df_lda, feature_sets)

    print("\nDone.")
