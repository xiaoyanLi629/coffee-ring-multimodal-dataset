#!/usr/bin/env python3
"""
Advanced Visualizations for Research Paper
Including 3D plots, network analysis, and interactive visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import networkx as nx
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# Import professional plotting style
from professional_plot_style import setup_professional_style, style_axis, get_professional_colors

# Set up professional style
setup_professional_style()
colors = get_professional_colors()

def clean_label(text):
    """Enhanced function to clean all types of labels"""
    if not isinstance(text, str):
        return str(text)
    
    # First, replace all underscores with spaces
    text = text.replace('_', ' ')
    
    # Special handling for condition labels
    if text.lower().startswith('condition '):
        return text.replace('condition ', 'Condition ').replace('condition', 'Condition')
    
    # Remove prefixes but keep important context
    text_lower = text.lower()
    
    # Handle specific patterns
    if 'normal images:' in text_lower:
        text = text.replace('Normal Images:', 'Normal:').replace('normal images:', 'Normal:')
    elif 'sem images:' in text_lower:
        text = text.replace('SEM Images:', 'SEM:').replace('sem images:', 'SEM:')
    
    # Remove technical suffixes
    text = text.replace(' d1 a0', '').replace(' d1 a45', '').replace(' d1 a90', '').replace(' d1 a135', '')
    
    # Special replacements
    replacements = {
        'cross modal ssim': 'SSIM',
        'ssim': 'SSIM',
        'sem ': 'SEM ',
        'freq ': 'Frequency ',
        'num objects': 'Number of Objects',
        'pc1': 'PC1',
        'pc2': 'PC2',
        't-sne': 't-SNE',
        'tsne': 't-SNE'
    }
    
    # Apply replacements
    for old, new in replacements.items():
        if old in text.lower():
            # Case-insensitive replacement
            import re
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    
    # Clean up and capitalize
    text = ' '.join(text.split())
    if text and text[0].islower() and not text.startswith('t-'):
        text = text[0].upper() + text[1:]
    
    return text

def shorten_label_for_3d(text):
    """Shorten labels specifically for 3D plots to avoid overlap"""
    # First clean the label
    clean = clean_label(text)
    
    # Apply specific shortenings for 3D plots
    replacements = {
        'Normal Frequency total spectral energy': 'Normal Freq. Energy',
        'SEM Frequency total spectral energy': 'SEM Freq. Energy',
        'Total Spectral Energy': 'Spectral Energy',
        'Frequency Total Spectral Energy': 'Freq. Energy'
    }
    
    for old, new in replacements.items():
        if old in clean:
            clean = clean.replace(old, new)
    
    return clean

class AdvancedVisualizer:
    def __init__(self):
        self.conditions = ['condition_A', 'condition_B', 'condition_C', 'condition_D', 
                          'condition_E', 'condition_F', 'condition_G', 'condition_H', 'condition_I']
        
    def create_3d_feature_space(self, features_df):
        """Create 3D visualization of feature space"""
        fig = plt.figure(figsize=(24, 18))
        
        # Select key features
        feature_cols = ['cross_modal_ssim', 'normal_freq_total_spectral_energy', 
                       'sem_freq_total_spectral_energy']
        
        # Check if features exist
        available_features = [col for col in feature_cols if col in features_df.columns]
        
        if len(available_features) >= 3:
            # 1. 3D scatter plot
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            
            # Prepare data
            X = features_df[available_features[:3]].values
            conditions = features_df['condition'].values
            
            # Create color map
            unique_conditions = np.unique(conditions)
            colors_map = plt.cm.tab10(np.linspace(0, 1, len(unique_conditions)))
            
            for i, cond in enumerate(unique_conditions):
                mask = conditions == cond
                ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                           c=[colors_map[i]], label=cond, s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
            
            ax1.set_xlabel(shorten_label_for_3d(available_features[0]), labelpad=20, fontsize=12)
            ax1.set_ylabel(shorten_label_for_3d(available_features[1]), labelpad=20, fontsize=12)
            ax1.set_zlabel(shorten_label_for_3d(available_features[2]), labelpad=15, fontsize=12)
            ax1.set_title('3D Feature Space Visualization', fontsize=16, pad=20)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            
            # Adjust viewing angle and tick parameters
            ax1.view_init(elev=25, azim=45)
            ax1.tick_params(axis='x', pad=10)
            ax1.tick_params(axis='y', pad=10)
            ax1.tick_params(axis='z', pad=8)
            
            # 2. 3D surface plot of feature interactions
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            
            # Create meshgrid for surface
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
            y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            
            # Interpolate Z values
            from scipy.interpolate import griddata
            try:
                Z_grid = griddata((X[:, 0], X[:, 1]), X[:, 2], 
                                 (X_grid, Y_grid), method='linear')
            except:
                # If interpolation fails, use nearest neighbor
                Z_grid = griddata((X[:, 0], X[:, 1]), X[:, 2], 
                                 (X_grid, Y_grid), method='nearest')
            
            # Create surface plot
            surf = ax2.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis',
                                   alpha=0.7, antialiased=True, linewidth=0)
            
            # Add scatter points
            ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c='red', s=20, alpha=0.5)
            
            ax2.set_xlabel(shorten_label_for_3d(available_features[0]), labelpad=20, fontsize=12)
            ax2.set_ylabel(shorten_label_for_3d(available_features[1]), labelpad=20, fontsize=12)
            ax2.set_zlabel(shorten_label_for_3d(available_features[2]), labelpad=15, fontsize=12)
            ax2.set_title('Feature Interaction Surface', fontsize=16, pad=20)
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, pad=0.15)
            cbar.set_label('Feature Value', fontsize=10)
            
            # Adjust viewing angle and tick parameters
            ax2.view_init(elev=25, azim=-45)
            ax2.tick_params(axis='x', pad=10)
            ax2.tick_params(axis='y', pad=10)
            ax2.tick_params(axis='z', pad=8)
            
            # 3. 3D trajectory plot by condition
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            
            # Plot trajectories for each sample type
            for sample in features_df['sample'].unique():
                sample_data = features_df[features_df['sample'] == sample]
                if len(sample_data) > 0:
                    # Sort by condition for trajectory
                    sample_data = sample_data.sort_values('condition')
                    X_sample = sample_data[available_features[:3]].values
                    
                    # Plot trajectory
                    ax3.plot(X_sample[:, 0], X_sample[:, 1], X_sample[:, 2],
                            'o-', label=sample, markersize=6, linewidth=2, alpha=0.8)
            
            ax3.set_xlabel(shorten_label_for_3d(available_features[0]), labelpad=20, fontsize=12)
            ax3.set_ylabel(shorten_label_for_3d(available_features[1]), labelpad=20, fontsize=12)
            ax3.set_zlabel(shorten_label_for_3d(available_features[2]), labelpad=15, fontsize=12)
            ax3.set_title('Sample Trajectories in Feature Space', fontsize=16, pad=20)
            ax3.legend()
            
            # Adjust viewing angle and tick parameters
            ax3.view_init(elev=25, azim=135)
            ax3.tick_params(axis='x', pad=10)
            ax3.tick_params(axis='y', pad=10)
            ax3.tick_params(axis='z', pad=8)
            
            # 4. 3D density plot
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            
            # Create 3D histogram
            hist, edges = np.histogramdd(X, bins=10)
            
            # Get bin centers
            x_centers = (edges[0][:-1] + edges[0][1:]) / 2
            y_centers = (edges[1][:-1] + edges[1][1:]) / 2
            z_centers = (edges[2][:-1] + edges[2][1:]) / 2
            
            # Create meshgrid
            x_mesh, y_mesh, z_mesh = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
            
            # Plot only non-zero bins
            mask = hist > 0
            scatter = ax4.scatter(x_mesh[mask], y_mesh[mask], z_mesh[mask],
                                 c=hist[mask], s=hist[mask]*50, cmap='hot',
                                 alpha=0.6, edgecolors='black', linewidth=0.5)
            
            ax4.set_xlabel(shorten_label_for_3d(available_features[0]), labelpad=20, fontsize=12)
            ax4.set_ylabel(shorten_label_for_3d(available_features[1]), labelpad=20, fontsize=12)
            ax4.set_zlabel(shorten_label_for_3d(available_features[2]), labelpad=15, fontsize=12)
            ax4.set_title('3D Feature Density Distribution', fontsize=16, pad=20)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax4, shrink=0.5, aspect=5, pad=0.15)
            cbar.set_label('Sample Density', fontsize=10)
            
            # Adjust viewing angle and tick parameters
            ax4.view_init(elev=25, azim=-135)
            ax4.tick_params(axis='x', pad=10)
            ax4.tick_params(axis='y', pad=10)
            ax4.tick_params(axis='z', pad=8)
            
        else:
            # If not enough features, create placeholder
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Insufficient features for 3D visualization\nRequired: 3 numerical features',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('3D Feature Space Visualization')
            ax.axis('off')
        
        plt.suptitle('Figure 4: 3D Feature Space Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=3.0, w_pad=3.0)
        plt.savefig('figure_4_3d_feature_space.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_4_3d_feature_space.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()

    def create_network_analysis(self, features_df):
        """Create network-based visualizations"""
        fig = plt.figure(figsize=(24, 18))
        
        # Calculate feature correlations
        feature_cols = [col for col in features_df.columns if 
                       ('normal_' in col or 'sem_' in col or col == 'cross_modal_ssim') and 
                       features_df[col].dtype in ['float64', 'int64']]
        
        if len(feature_cols) > 5:
            # 1. Feature correlation network
            ax1 = plt.subplot(2, 2, 1)
            
            # Calculate correlation matrix
            corr_matrix = features_df[feature_cols[:20]].corr()  # Limit to top 20 features
            
            # Create network from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for feature in corr_matrix.columns:
                G.add_node(feature)
            
            # Add edges for significant correlations
            threshold = 0.6  # Increased threshold to reduce edge clutter
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], 
                                  weight=abs(corr_val), correlation=corr_val)
            
            # Calculate layout with more spacing
            # Try different layout algorithms based on network density
            num_edges = len(G.edges())
            num_nodes = len(G.nodes())
            
            if num_nodes > 0:
                density = 2 * num_edges / (num_nodes * (num_nodes - 1))
                
                # Use Kamada-Kawai for dense networks, spring for sparse
                if density > 0.3 and num_nodes < 30:
                    pos = nx.kamada_kawai_layout(G, scale=3)
                else:
                    # k parameter controls optimal distance between nodes
                    k_value = 3 * np.sqrt(1 / density) if density > 0 else 5
                    pos = nx.spring_layout(G, k=k_value, iterations=200, seed=42, scale=3)
            
            # Draw network
            # Draw edges with varying width
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            correlations = [G[u][v]['correlation'] for u, v in edges]
            
            # Color edges by positive/negative correlation
            edge_colors = ['red' if c < 0 else 'blue' for c in correlations]
            
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights],
                                  edge_color=edge_colors, alpha=0.6, ax=ax1)
            
            # Draw nodes
            node_colors = ['lightcoral' if 'sem_' in node else 'lightblue' 
                          if 'normal_' in node else 'lightgreen' for node in G.nodes()]
            
            # Adjust node size based on number of nodes
            node_size = max(400, 1200 - len(G.nodes()) * 20)
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=node_size, alpha=0.9, ax=ax1)
            
            # Add labels with clean formatting
            labels = {}
            for node in G.nodes():
                # Clean and shorten label
                clean = clean_label(node)
                # More aggressive shortening for network display
                if len(clean) > 8:
                    # Try to get meaningful abbreviation
                    parts = clean.split()
                    if len(parts) > 1:
                        labels[node] = parts[0][:4] + '...'
                    else:
                        labels[node] = clean[:6] + '...'
                else:
                    labels[node] = clean
            
            # Adjust font size based on number of nodes
            font_size = max(7, 11 - len(G.nodes()) // 5)
            nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax1,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax1.set_title('Feature Correlation Network (|r| > 0.6)', fontsize=18, pad=20)
            ax1.axis('off')
            # Expand axis limits for better spacing
            ax1.set_xlim(-4, 4)
            ax1.set_ylim(-4, 4)
            
            # Add legend
            red_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Negative')
            blue_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Positive')
            ax1.legend(handles=[red_line, blue_line], loc='upper right')
            
            # 2. Sample similarity network
            ax2 = plt.subplot(2, 2, 2)
            
            # Calculate sample distances
            sample_features = features_df[feature_cols].fillna(features_df[feature_cols].mean())
            scaler = StandardScaler()
            sample_features_scaled = scaler.fit_transform(sample_features)
            
            # Calculate distance matrix
            distances = pdist(sample_features_scaled[:50], metric='euclidean')  # Limit samples
            distance_matrix = squareform(distances)
            
            # Create similarity matrix
            similarity_matrix = 1 / (1 + distance_matrix)
            
            # Create network
            G_samples = nx.Graph()
            
            # Add nodes with attributes
            for idx in range(min(50, len(features_df))):
                G_samples.add_node(idx, 
                                  condition=features_df.iloc[idx]['condition'],
                                  sample=features_df.iloc[idx]['sample'])
            
            # Add edges for similar samples
            threshold = np.percentile(similarity_matrix[similarity_matrix > 0], 80)
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > threshold:
                        G_samples.add_edge(i, j, weight=similarity_matrix[i, j])
            
            # Layout
            pos_samples = nx.spring_layout(G_samples, k=2, iterations=50, seed=42)
            
            # Draw edges
            edges = G_samples.edges()
            weights = [G_samples[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(G_samples, pos_samples, 
                                  width=[w*2 for w in weights],
                                  alpha=0.3, ax=ax2)
            
            # Draw nodes colored by condition
            node_colors = [hash(G_samples.nodes[node]['condition']) % 10 
                          for node in G_samples.nodes()]
            
            nx.draw_networkx_nodes(G_samples, pos_samples, 
                                  node_color=node_colors,
                                  node_size=300, cmap='tab10', 
                                  alpha=0.9, ax=ax2)
            
            ax2.set_title('Sample Similarity Network', fontsize=16)
            ax2.axis('off')
            
            # 3. Hierarchical feature clustering
            ax3 = plt.subplot(2, 2, 3)
            
            # Perform hierarchical clustering on features
            from scipy.cluster.hierarchy import dendrogram, linkage
            
            # Select subset of features
            feature_subset = feature_cols[:15]
            feature_corr = features_df[feature_subset].corr()
            
            # Calculate linkage
            linkage_matrix = linkage(1 - feature_corr, method='ward')
            
            # Create dendrogram
            clean_labels = [clean_label(f) for f in feature_subset]
            dendro = dendrogram(linkage_matrix, labels=clean_labels,
                               ax=ax3, leaf_rotation=90, leaf_font_size=10)
            
            ax3.set_title('Feature Hierarchical Clustering', fontsize=16)
            ax3.set_xlabel('Features')
            ax3.set_ylabel('Distance')
            
            # 4. Condition transition network
            ax4 = plt.subplot(2, 2, 4)
            
            # Create directed graph for condition transitions
            G_conditions = nx.DiGraph()
            
            # Define temperature and humidity levels
            temp_levels = {'A': 0, 'B': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 1, 'G': 2, 'H': 2, 'I': 2}
            humidity_levels = {'A': 0, 'B': 1, 'C': 2, 'D': 0, 'E': 1, 'F': 2, 'G': 0, 'H': 1, 'I': 2}
            
            # Add nodes
            for cond in self.conditions:
                letter = cond.split('_')[1]
                G_conditions.add_node(letter, 
                                     temp=temp_levels[letter],
                                     humidity=humidity_levels[letter])
            
            # Add edges for transitions
            for cond1 in G_conditions.nodes():
                for cond2 in G_conditions.nodes():
                    if cond1 != cond2:
                        # Connect if differ by one level in temp or humidity
                        temp_diff = abs(temp_levels[cond1] - temp_levels[cond2])
                        humidity_diff = abs(humidity_levels[cond1] - humidity_levels[cond2])
                        
                        if (temp_diff == 1 and humidity_diff == 0) or \
                           (temp_diff == 0 and humidity_diff == 1):
                            G_conditions.add_edge(cond1, cond2)
            
            # Layout in grid
            pos_conditions = {}
            for node in G_conditions.nodes():
                x = humidity_levels[node]
                y = temp_levels[node]
                pos_conditions[node] = (x, y)
            
            # Draw network
            nx.draw_networkx_edges(G_conditions, pos_conditions, 
                                  edge_color='gray', arrows=True,
                                  arrowsize=20, alpha=0.5, ax=ax4)
            
            # Draw nodes with size based on average SSIM
            if 'cross_modal_ssim' in features_df.columns:
                node_sizes = []
                for node in G_conditions.nodes():
                    cond_data = features_df[features_df['condition'] == f'condition_{node}']
                    avg_ssim = cond_data['cross_modal_ssim'].mean() if len(cond_data) > 0 else 0.5
                    node_sizes.append(avg_ssim * 1000)
            else:
                node_sizes = [500] * len(G_conditions.nodes())
            
            nx.draw_networkx_nodes(G_conditions, pos_conditions,
                                  node_size=node_sizes,
                                  node_color='lightblue',
                                  alpha=0.9, ax=ax4)
            
            nx.draw_networkx_labels(G_conditions, pos_conditions, 
                                   font_size=12, font_weight='bold', ax=ax4)
            
            ax4.set_xlim(-0.5, 2.5)
            ax4.set_ylim(-0.5, 2.5)
            ax4.set_xlabel('Humidity Level', fontsize=12)
            ax4.set_ylabel('Temperature Level', fontsize=12)
            ax4.set_title('Environmental Condition Transition Network', fontsize=16)
            ax4.grid(True, alpha=0.3)
            
        else:
            # Placeholder if not enough features
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Insufficient features for network analysis',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Network Analysis')
            ax.axis('off')
        
        plt.suptitle('Figure 5: Network-based Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure_5_network_analysis.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_5_network_analysis.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def create_advanced_statistical_plots(self, features_df):
        """Create advanced statistical visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Ridge plot for distribution comparison
        ax1 = plt.subplot(3, 2, 1)
        
        if 'cross_modal_ssim' in features_df.columns:
            # Create ridge plot
            conditions = features_df['condition'].unique()
            n_conditions = len(conditions)
            
            # Set up the plot
            overlap = 0.7
            
            for i, condition in enumerate(sorted(conditions)):
                data = features_df[features_df['condition'] == condition]['cross_modal_ssim'].dropna()
                
                if len(data) > 0:
                    # Calculate KDE
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(0, 1, 200)
                    y_kde = kde(x_range)
                    
                    # Normalize and offset
                    y_kde = y_kde / y_kde.max() * overlap
                    y_offset = i * 0.8
                    
                    # Plot
                    ax1.fill_between(x_range, y_offset, y_kde + y_offset,
                                    alpha=0.7, color=plt.cm.viridis(i/n_conditions))
                    ax1.plot(x_range, y_kde + y_offset, color='black', linewidth=1)
                    
                    # Add label
                    ax1.text(-0.05, y_offset + 0.3, condition.split('_')[1], 
                            fontsize=10, ha='right')
            
            ax1.set_xlim(0, 1)
            ax1.set_xlabel('SSIM Value')
            ax1.set_title('Ridge Plot: SSIM Distribution by Condition', fontsize=14)
            ax1.set_yticks([])
            style_axis(ax1)
        else:
            ax1.text(0.5, 0.5, 'SSIM data not available for ridge plot',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Ridge Plot')
            style_axis(ax1)
        
        # 2. Hexbin plot for dense data
        ax2 = plt.subplot(3, 2, 2)
        
        if 'normal_freq_total_spectral_energy' in features_df.columns and \
           'sem_freq_total_spectral_energy' in features_df.columns:
            # Create hexbin plot
            x = features_df['normal_freq_total_spectral_energy']
            y = features_df['sem_freq_total_spectral_energy']
            
            # Remove outliers
            mask = (x < x.quantile(0.99)) & (y < y.quantile(0.99))
            
            hb = ax2.hexbin(x[mask], y[mask], gridsize=20, cmap='YlOrRd', mincnt=1)
            cb = plt.colorbar(hb, ax=ax2)
            cb.set_label('Sample Count')
            
            ax2.set_xlabel(clean_label('Normal Images: Spectral Energy'))
            ax2.set_ylabel(clean_label('SEM Images: Spectral Energy'))
            ax2.set_title('Hexbin: Spectral Energy Density', fontsize=14)
            style_axis(ax2)
        else:
            ax2.text(0.5, 0.5, 'Spectral energy data not available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hexbin Plot')
            style_axis(ax2)
        
        # 3. Quantile-Quantile plot
        ax3 = plt.subplot(3, 2, 3)
        
        if 'cross_modal_ssim' in features_df.columns:
            # Q-Q plot against normal distribution
            from scipy import stats
            
            ssim_data = features_df['cross_modal_ssim'].dropna()
            
            # Calculate theoretical quantiles
            stats.probplot(ssim_data, dist="norm", plot=ax3)
            
            ax3.set_title('Q-Q Plot: SSIM vs Normal Distribution', fontsize=14)
            ax3.set_xlabel('Theoretical Quantiles')
            ax3.set_ylabel('Sample Quantiles')
            ax3.grid(True, alpha=0.3)
            
            # Add R² value
            (osm, osr), (slope, intercept, r) = stats.probplot(ssim_data, dist="norm", plot=None)
            ax3.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            style_axis(ax3)
        else:
            ax3.text(0.5, 0.5, 'SSIM data not available for Q-Q plot',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Q-Q Plot')
            style_axis(ax3)
        
        # 4. Joint plot with marginal distributions
        ax4 = plt.subplot(3, 2, 4)
        ax4_divider = ax4.get_position()
        ax4.set_position([ax4_divider.x0, ax4_divider.y0, 
                         ax4_divider.width * 0.8, ax4_divider.height * 0.8])
        
        # Create marginal axes
        ax4_margx = plt.axes([ax4_divider.x0, ax4_divider.y0 + ax4_divider.height * 0.8,
                             ax4_divider.width * 0.8, ax4_divider.height * 0.15])
        ax4_margy = plt.axes([ax4_divider.x0 + ax4_divider.width * 0.8, ax4_divider.y0,
                             ax4_divider.width * 0.15, ax4_divider.height * 0.8])
        
        if 'normal_texture_contrast_d1_a0' in features_df.columns and \
           'sem_texture_contrast_d1_a0' in features_df.columns:
            x = features_df['normal_texture_contrast_d1_a0']
            y = features_df['sem_texture_contrast_d1_a0']
            
            # Main scatter plot
            ax4.scatter(x, y, alpha=0.5, s=30)
            ax4.set_xlabel(clean_label('Normal: Texture Contrast'))
            ax4.set_ylabel(clean_label('SEM: Texture Contrast'))
            
            # Marginal histograms
            ax4_margx.hist(x, bins=30, alpha=0.7, color=colors['primary'])
            ax4_margy.hist(y, bins=30, alpha=0.7, color=colors['secondary'], 
                          orientation='horizontal')
            
            # Remove labels from marginal plots
            ax4_margx.set_xticklabels([])
            ax4_margy.set_yticklabels([])
            ax4_margx.set_ylabel('Count')
            ax4_margy.set_xlabel('Count')
            
            # Set title on marginal x
            ax4_margx.set_title('Joint Distribution: Texture Contrast', fontsize=14)
            
            style_axis(ax4)
            style_axis(ax4_margx)
            style_axis(ax4_margy)
        else:
            ax4.text(0.5, 0.5, 'Texture contrast data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Joint Distribution')
            ax4_margx.remove()
            ax4_margy.remove()
            style_axis(ax4)
        
        # 5. Parallel categories plot
        ax5 = plt.subplot(3, 2, 5)
        
        # Create categorical data based on quantiles
        if 'cross_modal_ssim' in features_df.columns:
            # Categorize SSIM
            features_df['ssim_category'] = pd.qcut(features_df['cross_modal_ssim'], 
                                                   q=3, labels=['Low', 'Medium', 'High'])
            
            # Extract condition categories
            features_df['temp_category'] = features_df['condition'].str[-1].map({
                'A': 'Low', 'B': 'Low', 'C': 'Low',
                'D': 'Med', 'E': 'Med', 'F': 'Med',
                'G': 'High', 'H': 'High', 'I': 'High'
            })
            
            features_df['humidity_category'] = features_df['condition'].str[-1].map({
                'A': 'Low', 'B': 'Med', 'C': 'High',
                'D': 'Low', 'E': 'Med', 'F': 'High',
                'G': 'Low', 'H': 'Med', 'I': 'High'
            })
            
            # Create alluvial diagram data
            categories = ['temp_category', 'humidity_category', 'ssim_category']
            
            # Count flows
            flow_data = features_df.groupby(categories).size().reset_index(name='count')
            
            # Simple visualization as stacked bars
            category_positions = {cat: i for i, cat in enumerate(categories)}
            
            for _, row in flow_data.iterrows():
                for i in range(len(categories) - 1):
                    cat1 = categories[i]
                    cat2 = categories[i + 1]
                    
                    # Define positions
                    x1 = category_positions[cat1]
                    x2 = category_positions[cat2]
                    
                    # Map category values to y positions
                    y_map = {'Low': 0, 'Med': 1, 'High': 2}
                    y1 = y_map.get(row[cat1], 0)
                    y2 = y_map.get(row[cat2], 0)
                    
                    # Draw connection
                    ax5.plot([x1, x2], [y1, y2], alpha=0.3, 
                            linewidth=row['count']/5, color='gray')
            
            # Add category labels
            for cat, pos in category_positions.items():
                ax5.text(pos, -0.5, cat.replace('_', '\n'), 
                        ha='center', fontsize=10)
                
                # Add value labels
                for i, val in enumerate(['Low', 'Med', 'High']):
                    ax5.text(pos, i, val, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            ax5.set_xlim(-0.5, len(categories) - 0.5)
            ax5.set_ylim(-1, 3)
            ax5.set_title('Categorical Flow: Environmental → SSIM', fontsize=14)
            ax5.axis('off')
        else:
            ax5.text(0.5, 0.5, 'Data not available for categorical flow',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Categorical Flow')
            ax5.axis('off')
        
        # 6. Bootstrap confidence intervals
        ax6 = plt.subplot(3, 2, 6)
        
        if 'cross_modal_ssim' in features_df.columns:
            # Calculate bootstrap CIs for each condition
            conditions = sorted(features_df['condition'].unique())
            means = []
            ci_lower = []
            ci_upper = []
            
            for cond in conditions:
                data = features_df[features_df['condition'] == cond]['cross_modal_ssim'].dropna()
                
                if len(data) > 5:
                    # Bootstrap
                    n_bootstrap = 1000
                    bootstrap_means = []
                    
                    for _ in range(n_bootstrap):
                        sample = np.random.choice(data, size=len(data), replace=True)
                        bootstrap_means.append(np.mean(sample))
                    
                    # Calculate CI
                    means.append(np.mean(data))
                    ci_lower.append(np.percentile(bootstrap_means, 2.5))
                    ci_upper.append(np.percentile(bootstrap_means, 97.5))
                else:
                    means.append(np.nan)
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)
            
            # Plot
            x_pos = np.arange(len(conditions))
            
            # Plot means with error bars
            ax6.errorbar(x_pos, means, 
                        yerr=[np.array(means) - np.array(ci_lower),
                              np.array(ci_upper) - np.array(means)],
                        fmt='o', capsize=5, capthick=2, 
                        color=colors['primary'], markersize=8)
            
            # Add reference line
            ax6.axhline(np.nanmean(means), color='red', linestyle='--', 
                       alpha=0.5, label='Overall mean')
            
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels([clean_label(c) for c in conditions])
            ax6.set_xlabel('Condition')
            ax6.set_ylabel('SSIM')
            ax6.set_title('Bootstrap 95% Confidence Intervals', fontsize=14)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            style_axis(ax6)
        else:
            ax6.text(0.5, 0.5, 'SSIM data not available for bootstrap analysis',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Bootstrap Analysis')
            style_axis(ax6)
        
        plt.suptitle('Figure 6: Advanced Statistical Visualizations', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure_6_advanced_statistics.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_6_advanced_statistics.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def create_comprehensive_summary(self, features_df):
        """Create a comprehensive summary visualization"""
        fig = plt.figure(figsize=(24, 16))
        
        # Create a dashboard-style layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall metrics summary (top-left, 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        if 'cross_modal_ssim' in features_df.columns:
            # Calculate key metrics
            metrics = {
                'Total Samples': len(features_df),
                'Avg SSIM': f"{features_df['cross_modal_ssim'].mean():.3f}",
                'Std SSIM': f"{features_df['cross_modal_ssim'].std():.3f}",
                'Conditions': len(features_df['condition'].unique()),
                'Sample Types': len(features_df['sample'].unique())
            }
            
            # Create text summary
            y_pos = 0.9
            for key, value in metrics.items():
                ax1.text(0.1, y_pos, f'{key}:', fontsize=14, weight='bold')
                ax1.text(0.6, y_pos, str(value), fontsize=14)
                y_pos -= 0.15
            
            # Add dataset title
            ax1.text(0.5, 1.05, 'Dataset Overview', fontsize=18, weight='bold',
                    ha='center', transform=ax1.transAxes)
            
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            
            # Add border
            rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, 
                                 boxstyle="round,pad=0.02",
                                 facecolor='lightgray', alpha=0.2,
                                 edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
        else:
            ax1.text(0.5, 0.5, 'Summary data not available',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
        
        # 2. Performance radar chart (top-right)
        ax2 = fig.add_subplot(gs[0:2, 2:4], projection='polar')
        
        if any(col in features_df.columns for col in ['cross_modal_ssim', 
                                                       'normal_freq_total_spectral_energy']):
            # Prepare radar chart data
            categories = []
            values = []
            
            if 'cross_modal_ssim' in features_df.columns:
                categories.append('SSIM')
                values.append(features_df['cross_modal_ssim'].mean())
            
            # Add normalized frequency energy
            if 'normal_freq_total_spectral_energy' in features_df.columns:
                categories.append('Normal\nFreq Energy')
                norm_energy = features_df['normal_freq_total_spectral_energy'].mean()
                norm_energy_scaled = norm_energy / features_df['normal_freq_total_spectral_energy'].max()
                values.append(norm_energy_scaled)
            
            if 'sem_freq_total_spectral_energy' in features_df.columns:
                categories.append('SEM\nFreq Energy')
                sem_energy = features_df['sem_freq_total_spectral_energy'].mean()
                sem_energy_scaled = sem_energy / features_df['sem_freq_total_spectral_energy'].max()
                values.append(sem_energy_scaled)
            
            # Add texture metrics
            if 'normal_texture_contrast_d1_a0' in features_df.columns:
                categories.append('Normal\nTexture')
                norm_texture = features_df['normal_texture_contrast_d1_a0'].mean()
                norm_texture_scaled = norm_texture / features_df['normal_texture_contrast_d1_a0'].max()
                values.append(norm_texture_scaled)
            
            if 'sem_texture_contrast_d1_a0' in features_df.columns:
                categories.append('SEM\nTexture')
                sem_texture = features_df['sem_texture_contrast_d1_a0'].mean()
                sem_texture_scaled = sem_texture / features_df['sem_texture_contrast_d1_a0'].max()
                values.append(sem_texture_scaled)
            
            # Complete the circle
            values = values + values[:1]
            
            # Calculate angles
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax2.plot(angles, values, 'o-', linewidth=2, color=colors['primary'])
            ax2.fill(angles, values, alpha=0.25, color=colors['primary'])
            
            # Set labels with padding to avoid overlap
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories, fontsize=10)
            ax2.set_ylim(0, 1.2)  # Increase ylim to give more space
            ax2.set_title('Overall Performance Metrics', fontsize=16, pad=30)  # Increase padding
            ax2.grid(True)
            
            # Move labels outward to avoid overlap
            for label in ax2.get_xticklabels():
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')
        else:
            ax2.text(0, 0, 'Performance data not available',
                    ha='center', va='center')
            ax2.set_title('Performance Metrics')
        
        # 3. Condition matrix heatmap (bottom-left)
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        
        if 'cross_modal_ssim' in features_df.columns:
            # Create condition matrix
            temp_map = {'A': '20-23°C', 'B': '20-23°C', 'C': '20-23°C',
                       'D': '23-26°C', 'E': '23-26°C', 'F': '23-26°C',
                       'G': '26-29°C', 'H': '26-29°C', 'I': '26-29°C'}
            humidity_map = {'A': '35-40%', 'B': '40-45%', 'C': '45-50%',
                           'D': '35-40%', 'E': '40-45%', 'F': '45-50%',
                           'G': '35-40%', 'H': '40-45%', 'I': '45-50%'}
            
            # Create matrix
            matrix_data = np.zeros((3, 3))
            temp_labels = ['20-23°C', '23-26°C', '26-29°C']
            humidity_labels = ['35-40%', '40-45%', '45-50%']
            
            for i, temp in enumerate(temp_labels):
                for j, humidity in enumerate(humidity_labels):
                    # Find matching condition
                    for cond in self.conditions:
                        letter = cond.split('_')[1]
                        if temp_map[letter] == temp and humidity_map[letter] == humidity:
                            cond_data = features_df[features_df['condition'] == cond]
                            if len(cond_data) > 0:
                                matrix_data[i, j] = cond_data['cross_modal_ssim'].mean()
            
            # Plot heatmap
            im = ax3.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax3.set_xticks(range(3))
            ax3.set_yticks(range(3))
            ax3.set_xticklabels(humidity_labels)
            ax3.set_yticklabels(temp_labels)
            ax3.set_xlabel('Relative Humidity', fontsize=12)
            ax3.set_ylabel('Temperature', fontsize=12)
            ax3.set_title('Average SSIM by Environmental Conditions', fontsize=14)
            
            # Add text annotations
            for i in range(3):
                for j in range(3):
                    text = ax3.text(j, i, f'{matrix_data[i, j]:.3f}',
                                   ha='center', va='center', 
                                   color='white' if matrix_data[i, j] < 0.5 else 'black',
                                   fontsize=12, weight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3)
            cbar.set_label('SSIM', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Condition matrix data not available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Environmental Condition Matrix')
            style_axis(ax3)
        
        # 4. Sample comparison (bottom-right)
        ax4 = fig.add_subplot(gs[2:4, 2:4])
        
        if 'cross_modal_ssim' in features_df.columns:
            # Compare samples
            sample_stats = features_df.groupby('sample')['cross_modal_ssim'].agg(['mean', 'std', 'count'])
            sample_stats = sample_stats.sort_values('mean', ascending=False)
            
            # Create bar plot with error bars
            x_pos = np.arange(len(sample_stats))
            bars = ax4.bar(x_pos, sample_stats['mean'], 
                          yerr=sample_stats['std'], capsize=5,
                          color=[colors['primary'], colors['secondary'], colors['tertiary'], 
                                colors['quaternary'], colors['quinary']][:len(sample_stats)],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add sample sizes on bars
            for i, (idx, row) in enumerate(sample_stats.iterrows()):
                ax4.text(i, row['mean'] + row['std'] + 0.02, 
                        f"n={row['count']}", ha='center', fontsize=9)
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([clean_label(s) for s in sample_stats.index])
            ax4.set_xlabel('Sample Type', fontsize=12)
            ax4.set_ylabel('Average SSIM ± SD', fontsize=12)
            ax4.set_title('Performance Comparison by Sample Type', fontsize=14)
            ax4.set_ylim(0, 1.2)
            ax4.grid(True, axis='y', alpha=0.3)
            style_axis(ax4)
        else:
            ax4.text(0.5, 0.5, 'Sample comparison data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sample Comparison')
            style_axis(ax4)
        
        plt.suptitle('Figure 7: Comprehensive Analysis Summary Dashboard', 
                    fontsize=24, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figure_7_summary_dashboard.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_7_summary_dashboard.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def run_advanced_analysis(self):
        """Run all advanced visualizations"""
        print("Loading feature data...")
        features_df = pd.read_csv('comprehensive_computer_vision_features.csv')
        
        # Create sample column if it doesn't exist
        if 'sample' not in features_df.columns and 'image_id' in features_df.columns:
            sample_mapping = {1: 'Sample A', 2: 'Sample B', 3: 'Sample C', 4: 'Sample D', 5: 'Sample E'}
            features_df['sample'] = features_df['image_id'].map(sample_mapping)
        elif 'sample' not in features_df.columns:
            features_df['sample'] = 'Sample A'
        
        print("Creating 3D feature space visualization...")
        self.create_3d_feature_space(features_df)
        
        print("Creating network analysis...")
        self.create_network_analysis(features_df)
        
        print("Creating advanced statistical plots...")
        self.create_advanced_statistical_plots(features_df)
        
        print("Creating comprehensive summary dashboard...")
        self.create_comprehensive_summary(features_df)
        
        print("\nAdvanced visualizations complete! All figures saved in SVG format.")

if __name__ == "__main__":
    visualizer = AdvancedVisualizer()
    visualizer.run_advanced_analysis()