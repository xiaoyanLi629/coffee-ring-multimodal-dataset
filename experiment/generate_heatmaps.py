#!/usr/bin/env python3
"""
Generate heatmaps for comprehensive computer vision features and image characteristics summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# Import professional plotting style
from professional_plot_style import setup_professional_style, style_axis, get_professional_colors

# Set up professional style
setup_professional_style()
colors = get_professional_colors()

def format_feature_label(feature_name):
    """Format feature name for display by removing underscores and capitalizing words"""
    if isinstance(feature_name, str):
        # Replace underscores with spaces
        formatted = feature_name.replace('_', ' ')
        # Capitalize first letter of each word
        formatted = ' '.join(word.capitalize() for word in formatted.split())
        # Handle common abbreviations
        formatted = formatted.replace('Ssim', 'SSIM')
        formatted = formatted.replace('Sem ', 'SEM ')
        formatted = formatted.replace('Freq ', 'Frequency ')
        formatted = formatted.replace('Morph ', 'Morphological ')
        formatted = formatted.replace('D1', 'Distance 1')
        formatted = formatted.replace('D2', 'Distance 2')
        formatted = formatted.replace('A0', 'Angle 0°')
        formatted = formatted.replace('A45', 'Angle 45°')
        formatted = formatted.replace('A90', 'Angle 90°')
        formatted = formatted.replace('A135', 'Angle 135°')
        return formatted
    return str(feature_name)

def generate_features_heatmap():
    """Generate heatmap for comprehensive computer vision features"""
    print("Loading computer vision features...")
    
    # Load the CSV file
    df = pd.read_csv('comprehensive_computer_vision_features.csv')
    
    # Select key features for visualization
    key_features = [
        'cross_modal_ssim',
        'normal_texture_energy_d1_a0',
        'normal_texture_contrast_d1_a0',
        'normal_texture_homogeneity_d1_a0',
        'sem_texture_energy_d1_a0',
        'sem_texture_contrast_d1_a0',
        'sem_texture_homogeneity_d1_a0',
        'normal_freq_total_spectral_energy',
        'sem_freq_total_spectral_energy',
        'normal_freq_low_freq_energy_ratio',
        'sem_freq_low_freq_energy_ratio',
        'normal_morph_object_area_ratio',
        'sem_morph_object_area_ratio',
        'normal_morph_num_objects',
        'sem_morph_num_objects'
    ]
    
    # Filter available features
    available_features = [f for f in key_features if f in df.columns]
    
    # Create correlation matrix
    correlation_matrix = df[available_features].corr()
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Format feature labels
    formatted_labels = [format_feature_label(f) for f in available_features]
    
    # Create heatmap with dendrogram
    # Calculate linkage for hierarchical clustering
    linkage_matrix = linkage(pdist(correlation_matrix, metric='euclidean'), method='complete')
    
    # Create dendrogram
    dendro = dendrogram(linkage_matrix, no_plot=True)
    cluster_order = dendro['leaves']
    
    # Reorder correlation matrix
    clustered_corr = correlation_matrix.iloc[cluster_order, cluster_order]
    clustered_labels = [formatted_labels[i] for i in cluster_order]
    
    # Create heatmap
    mask = np.triu(np.ones_like(clustered_corr, dtype=bool), k=1)
    sns.heatmap(clustered_corr, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                xticklabels=clustered_labels,
                yticklabels=clustered_labels)
    
    plt.title('Computer Vision Features Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('heatmap_computer_vision_features.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('heatmap_computer_vision_features.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Computer vision features heatmap saved!")
    
    # Generate a second heatmap showing feature values by condition
    plt.figure(figsize=(16, 10))
    
    # Aggregate features by condition
    condition_features = df.groupby('condition')[available_features].mean()
    
    # Normalize features for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(condition_features.T).T,
        index=condition_features.index,
        columns=condition_features.columns
    )
    
    # Format condition labels
    formatted_conditions = [f"Condition {cond.split('_')[1]}" for cond in normalized_features.index]
    
    # Create heatmap
    sns.heatmap(normalized_features.T,
                cmap='RdBu_r',
                center=0,
                cbar_kws={"shrink": 0.8, "label": "Standardized Value"},
                xticklabels=formatted_conditions,
                yticklabels=formatted_labels,
                linewidths=0.5)
    
    plt.title('Feature Values by Environmental Condition (Standardized)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Environmental Condition', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('heatmap_features_by_condition.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('heatmap_features_by_condition.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Features by condition heatmap saved!")

def generate_summary_heatmap():
    """Generate heatmap for IMAGE CHARACTERISTICS SUMMARY"""
    print("Creating image characteristics summary heatmap...")
    
    # Parse the summary data from the report
    summary_data = {
        'image_id': [225.000, 13.000, 7.227, 1.000, 7.000, 13.000, 19.000, 25.000],
        'normal_brightness': [225.000, 41.101, 11.084, 19.557, 33.388, 40.377, 45.885, 80.539],
        'normal_contrast': [225.000, 30.780, 6.447, 9.155, 27.583, 31.466, 34.771, 45.643],
        'normal_mean_r': [225.000, 40.992, 11.167, 19.189, 33.093, 40.351, 45.869, 80.486],
        'normal_mean_g': [225.000, 41.162, 11.085, 19.699, 33.896, 40.347, 45.962, 80.658],
        'normal_mean_b': [225.000, 41.149, 11.011, 19.784, 33.419, 40.359, 46.115, 80.489],
        'normal_std_r': [225.000, 29.491, 6.360, 8.514, 26.626, 29.995, 33.304, 45.065],
        'normal_std_g': [225.000, 32.054, 6.613, 9.697, 28.671, 32.677, 36.578, 46.431],
        'normal_std_b': [225.000, 30.723, 6.437, 9.204, 27.658, 31.201, 34.750, 45.419],
        'sem_brightness': [225.000, 123.958, 20.870, 13.433, 119.439, 123.609, 130.710, 172.098],
        'sem_contrast': [225.000, 18.856, 8.129, 7.505, 13.699, 16.671, 22.064, 61.420],
        'sem_entropy': [225.000, 3.991, 0.502, 1.517, 3.760, 3.950, 4.267, 5.422],
        'brightness_ratio': [225.000, 3.263, 1.079, 0.251, 2.665, 3.144, 3.928, 6.567],
        'contrast_ratio': [225.000, 0.674, 0.554, 0.220, 0.430, 0.560, 0.733, 5.957]
    }
    
    # Statistics names
    stat_names = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary_data, index=stat_names)
    
    # Drop 'Count' row for visualization (all values are 225)
    df_viz = df_summary.drop('Count')
    
    # Format column names
    formatted_columns = [format_feature_label(col) for col in df_viz.columns]
    
    # Normalize data for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(df_viz.T).T,
        index=df_viz.index,
        columns=df_viz.columns
    )
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Create heatmap
    sns.heatmap(normalized_data,
                annot=df_viz.round(2),
                fmt='g',
                cmap='YlOrRd',
                cbar_kws={"shrink": 0.8, "label": "Standardized Value"},
                xticklabels=formatted_columns,
                yticklabels=df_viz.index,
                linewidths=0.5)
    
    plt.title('Image Characteristics Summary Statistics Heatmap', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Image Characteristic', fontsize=14)
    plt.ylabel('Statistic', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('heatmap_image_characteristics_summary.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('heatmap_image_characteristics_summary.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Image characteristics summary heatmap saved!")
    
    # Create a second visualization showing distribution patterns
    plt.figure(figsize=(14, 10))
    
    # Create subplots for distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Box plot style visualization
    ax1 = axes[0, 0]
    box_data = df_viz[['normal_brightness', 'normal_contrast', 'sem_brightness', 'sem_contrast', 'sem_entropy']]
    box_stats = box_data.loc[['Min', '25%', '50%', '75%', 'Max']]
    
    x_pos = np.arange(len(box_stats.columns))
    for i, col in enumerate(box_stats.columns):
        values = box_stats[col].values
        ax1.plot([i, i], [values[0], values[4]], 'k-', linewidth=1)
        ax1.fill_between([i-0.2, i+0.2], [values[1], values[1]], [values[3], values[3]], 
                        color=colors['primary'], alpha=0.3)
        ax1.plot([i-0.2, i+0.2], [values[2], values[2]], 'r-', linewidth=2)
        ax1.scatter(i, df_viz.loc['Mean', col], color='blue', s=100, zorder=5)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([format_feature_label(col) for col in box_stats.columns], rotation=45, ha='right')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Distribution of Key Image Characteristics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    style_axis(ax1)
    
    # Plot 2: Coefficient of Variation
    ax2 = axes[0, 1]
    cv_data = (df_viz.loc['Std Dev'] / df_viz.loc['Mean']).sort_values(ascending=False)
    bars = ax2.barh(range(len(cv_data)), cv_data.values, color=colors['secondary'])
    ax2.set_yticks(range(len(cv_data)))
    ax2.set_yticklabels([format_feature_label(col) for col in cv_data.index])
    ax2.set_xlabel('Coefficient of Variation', fontsize=12)
    ax2.set_title('Variability of Image Characteristics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    style_axis(ax2)
    
    # Plot 3: Range analysis
    ax3 = axes[1, 0]
    range_data = df_viz.loc['Max'] - df_viz.loc['Min']
    range_data = range_data.sort_values(ascending=False)
    bars = ax3.bar(range(len(range_data)), range_data.values, color=colors['tertiary'])
    ax3.set_xticks(range(len(range_data)))
    ax3.set_xticklabels([format_feature_label(col) for col in range_data.index], rotation=45, ha='right')
    ax3.set_ylabel('Range (Max - Min)', fontsize=12)
    ax3.set_title('Dynamic Range of Image Characteristics', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    style_axis(ax3)
    
    # Plot 4: Ratio analysis
    ax4 = axes[1, 1]
    ratio_features = ['brightness_ratio', 'contrast_ratio']
    ratio_data = df_viz[ratio_features]
    
    x = np.arange(len(ratio_data.index))
    width = 0.35
    
    for i, feature in enumerate(ratio_features):
        ax4.bar(x + i*width, ratio_data[feature], width, 
               label=format_feature_label(feature), alpha=0.8)
    
    ax4.set_xlabel('Statistic', fontsize=12)
    ax4.set_ylabel('Ratio Value', fontsize=12)
    ax4.set_title('Cross-modal Ratios Statistics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(ratio_data.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    style_axis(ax4)
    
    plt.suptitle('Comprehensive Image Characteristics Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('image_characteristics_analysis.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.savefig('image_characteristics_analysis.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Image characteristics analysis plots saved!")

def main():
    """Main function to generate all heatmaps"""
    print("Starting heatmap generation...")
    
    # Generate heatmaps
    generate_features_heatmap()
    generate_summary_heatmap()
    
    print("\nAll heatmaps generated successfully!")
    print("Generated files:")
    print("- heatmap_computer_vision_features.svg/png")
    print("- heatmap_features_by_condition.svg/png")
    print("- heatmap_image_characteristics_summary.svg/png")
    print("- image_characteristics_analysis.svg/png")

if __name__ == "__main__":
    main()