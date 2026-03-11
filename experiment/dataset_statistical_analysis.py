#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of Multi-modal Environmental Water Sample Dataset
This script analyzes the correlation between environmental conditions and water sample characteristics
using both normal cell phone images and SEM images.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import professional plotting style
from professional_plot_style import setup_professional_style, style_axis, get_professional_colors

# Set up professional style
setup_professional_style()
colors = get_professional_colors()

class WaterSampleDatasetAnalyzer:
    def __init__(self, base_path='./'):
        self.base_path = base_path
        self.conditions = ['condition_A', 'condition_B', 'condition_C', 'condition_D', 
                          'condition_E', 'condition_F', 'condition_G', 'condition_H', 'condition_I']
        self.water_samples = ['Sample A', 'Sample B', 'Sample C', 'Sample D', 'Sample E']
        self.replicates = 5
        self.images_per_condition = 25
        
        # Environmental parameters
        self.temp_ranges = {
            'A': (20, 23), 'B': (20, 23), 'C': (20, 23),
            'D': (23, 26), 'E': (23, 26), 'F': (23, 26),
            'G': (26, 29), 'H': (26, 29), 'I': (26, 29)
        }
        
        self.humidity_ranges = {
            'A': (35, 40), 'B': (40, 45), 'C': (45, 50),
            'D': (35, 40), 'E': (40, 45), 'F': (45, 50),
            'G': (35, 40), 'H': (40, 45), 'I': (45, 50)
        }
        
        # Initialize data containers
        self.image_statistics = {}
        self.environmental_data = pd.DataFrame()
        self.water_composition_data = pd.DataFrame()
        
    def load_water_composition_data(self):
        """Load and process water composition data from Michigan water reports"""
        try:
            # Load Michigan water data
            # Try current directory first, then base_path
            csv_path = 'sample_data_sheet.csv'
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.base_path, 'sample_data_sheet.csv')
            water_data = pd.read_csv(csv_path)
            
            # Create water composition matrix
            compositions = {
                'Sample A': {'NaHCO3': 0.1, 'CaCl2': 1.5, 'MgCl2': 0.5, 'Na2SO4': 0.35, 
                           'NaH2PO4': 0.033, 'KF': 0.4, 'Fe(NO3)3': 0.005, 'CuSO4': 0.00024},
                'Sample B': {'NaHCO3': 0.2, 'CaCl2': 1.0, 'MgCl2': 0.35, 'Na2SO4': 0.35, 
                           'NaH2PO4': 0.033, 'KF': 0.4, 'Fe(NO3)3': 0.005, 'CuSO4': 0.00024},
                'Sample C': {'NaHCO3': 0.1, 'CaCl2': 0.5, 'MgCl2': 0.2, 'Na2SO4': 0.35, 
                           'NaH2PO4': 0.033, 'KF': 0.4, 'Fe(NO3)3': 0.005, 'CuSO4': 0.00024},
                'Sample D': {'NaHCO3': 0.0, 'CaCl2': 1.0, 'MgCl2': 1.0, 'Na2SO4': 1.35, 
                           'NaH2PO4': 0.033, 'KF': 0.4, 'Fe(NO3)3': 0.005, 'CuSO4': 0.00024},
                'Sample E': {'NaHCO3': 0.0, 'CaCl2': 1.0, 'MgCl2': 0.5, 'Na2SO4': 2.35, 
                           'NaH2PO4': 0.033, 'KF': 0.4, 'Fe(NO3)3': 0.005, 'CuSO4': 0.00024}
            }
            
            self.water_composition_data = pd.DataFrame(compositions).T
            print("Water composition data loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading water composition data: {e}")
            return False
    
    def create_environmental_matrix(self):
        """Create comprehensive environmental conditions matrix"""
        env_data = []
        
        for condition in self.conditions:
            condition_letter = condition.split('_')[1]
            temp_range = self.temp_ranges[condition_letter]
            humidity_range = self.humidity_ranges[condition_letter]
            
            for sample_idx in range(len(self.water_samples)):
                for replicate in range(self.replicates):
                    env_data.append({
                        'condition': condition,
                        'condition_letter': condition_letter,
                        'water_sample': self.water_samples[sample_idx],
                        'replicate': replicate + 1,
                        'temp_min': temp_range[0],
                        'temp_max': temp_range[1],
                        'temp_mean': np.mean(temp_range),
                        'humidity_min': humidity_range[0],
                        'humidity_max': humidity_range[1],
                        'humidity_mean': np.mean(humidity_range),
                        'image_id': sample_idx * self.replicates + replicate + 1
                    })
        
        self.environmental_data = pd.DataFrame(env_data)
        print(f"Environmental matrix created: {len(self.environmental_data)} records")
        
    def analyze_image_basic_statistics(self):
        """Analyze basic image statistics for both normal and SEM images"""
        image_stats = []
        
        for condition in self.conditions:
            condition_path = os.path.join(self.base_path, condition)
            if not os.path.exists(condition_path):
                print(f"Warning: {condition_path} does not exist")
                continue
                
            # Analyze normal images
            for i in range(1, self.images_per_condition + 1):
                normal_image_path = os.path.join(condition_path, f'Image_{i}.jpg')
                sem_image_path = os.path.join(condition_path, f'SEM_{i}.jpg')
                
                if os.path.exists(normal_image_path) and os.path.exists(sem_image_path):
                    try:
                        # Normal image analysis
                        with Image.open(normal_image_path) as img:
                            img_array = np.array(img)
                            if len(img_array.shape) == 3:
                                # RGB image
                                mean_rgb = np.mean(img_array, axis=(0, 1))
                                std_rgb = np.std(img_array, axis=(0, 1))
                                brightness = np.mean(img_array)
                                contrast = np.std(img_array)
                            else:
                                # Grayscale
                                mean_rgb = [np.mean(img_array)] * 3
                                std_rgb = [np.std(img_array)] * 3
                                brightness = np.mean(img_array)
                                contrast = np.std(img_array)
                        
                        # SEM image analysis
                        with Image.open(sem_image_path) as sem_img:
                            sem_array = np.array(sem_img)
                            if len(sem_array.shape) == 3:
                                sem_array = np.mean(sem_array, axis=2)  # Convert to grayscale
                            
                            sem_brightness = np.mean(sem_array)
                            sem_contrast = np.std(sem_array)
                            sem_entropy = stats.entropy(np.histogram(sem_array, bins=256)[0] + 1e-10)
                        
                        image_stats.append({
                            'condition': condition,
                            'image_id': i,
                            'normal_brightness': brightness,
                            'normal_contrast': contrast,
                            'normal_mean_r': mean_rgb[0],
                            'normal_mean_g': mean_rgb[1],
                            'normal_mean_b': mean_rgb[2],
                            'normal_std_r': std_rgb[0],
                            'normal_std_g': std_rgb[1],
                            'normal_std_b': std_rgb[2],
                            'sem_brightness': sem_brightness,
                            'sem_contrast': sem_contrast,
                            'sem_entropy': sem_entropy,
                            'brightness_ratio': sem_brightness / brightness if brightness > 0 else 0,
                            'contrast_ratio': sem_contrast / contrast if contrast > 0 else 0
                        })
                        
                    except Exception as e:
                        print(f"Error processing {normal_image_path} or {sem_image_path}: {e}")
        
        self.image_statistics = pd.DataFrame(image_stats)
        print(f"Image statistics analyzed: {len(self.image_statistics)} image pairs")
        
    def perform_environmental_correlation_analysis(self):
        """Perform correlation analysis between environmental conditions and image characteristics"""
        if self.image_statistics.empty or self.environmental_data.empty:
            print("Error: Missing image statistics or environmental data")
            return
        
        # Merge datasets
        merged_data = self.image_statistics.merge(
            self.environmental_data, 
            left_on=['condition', 'image_id'], 
            right_on=['condition', 'image_id'],
            how='inner'
        )
        
        # Correlation analysis
        numeric_columns = ['temp_mean', 'humidity_mean', 'normal_brightness', 'normal_contrast',
                          'sem_brightness', 'sem_contrast', 'sem_entropy', 'brightness_ratio']
        
        correlation_matrix = merged_data[numeric_columns].corr()
        
        # Create clean labels without underscores for display
        clean_labels = {
            'temp_mean': 'temp mean',
            'humidity_mean': 'humidity mean',
            'normal_brightness': 'normal brightness',
            'normal_contrast': 'normal contrast',
            'sem_brightness': 'sem brightness',
            'sem_contrast': 'sem contrast',
            'sem_entropy': 'sem entropy',
            'brightness_ratio': 'brightness ratio'
        }
        
        # Rename columns for display
        display_matrix = correlation_matrix.rename(columns=clean_labels, index=clean_labels)
        
        # Visualization
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(display_matrix, dtype=bool))
        sns.heatmap(display_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Environmental-Image Characteristics Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('environmental_correlation_matrix.svg', format='svg', bbox_inches='tight')
        plt.savefig('environmental_correlation_matrix.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return merged_data, correlation_matrix
    
    def analyze_water_sample_effects(self):
        """Analyze the effects of different water sample compositions"""
        if self.water_composition_data.empty:
            print("Error: Water composition data not loaded")
            return

        # Create composition analysis
        composition_analysis = {}

        for sample_idx, sample in enumerate(self.water_samples):
            # Each condition has 25 images: 5 samples × 5 replicates
            # Sample indices within each condition: sample_idx*5+1 to sample_idx*5+5
            sample_image_ids = list(range(sample_idx * self.replicates + 1,
                                          (sample_idx + 1) * self.replicates + 1))
            sample_images = self.image_statistics[
                self.image_statistics['image_id'].isin(sample_image_ids)
            ]

            if not sample_images.empty:
                composition_analysis[sample] = {
                    'mean_normal_brightness': sample_images['normal_brightness'].mean(),
                    'std_normal_brightness': sample_images['normal_brightness'].std(),
                    'mean_sem_contrast': sample_images['sem_contrast'].mean(),
                    'std_sem_contrast': sample_images['sem_contrast'].std(),
                    'mean_entropy': sample_images['sem_entropy'].mean()
                }
        
        # Statistical significance testing
        brightness_by_sample = []
        contrast_by_sample = []
        
        for condition in self.conditions:
            condition_data = self.image_statistics[self.image_statistics['condition'] == condition]
            if not condition_data.empty:
                brightness_by_sample.append(condition_data['normal_brightness'].values)
                contrast_by_sample.append(condition_data['sem_contrast'].values)
        
        if len(brightness_by_sample) > 1:
            f_stat_brightness, p_val_brightness = stats.f_oneway(*brightness_by_sample)
            f_stat_contrast, p_val_contrast = stats.f_oneway(*contrast_by_sample)
            
            print(f"ANOVA Results:")
            print(f"Normal Image Brightness: F={f_stat_brightness:.3f}, p={p_val_brightness:.6f}")
            print(f"SEM Image Contrast: F={f_stat_contrast:.3f}, p={p_val_contrast:.6f}")
        
        return composition_analysis
    
    def perform_clustering_analysis(self):
        """Perform clustering analysis to identify natural groupings in the data"""
        if self.image_statistics.empty:
            print("Error: Image statistics not available")
            return
        
        # Prepare features for clustering
        features = ['normal_brightness', 'normal_contrast', 'sem_brightness', 
                   'sem_contrast', 'sem_entropy']
        
        # Get the indices for valid data before dropna
        valid_indices = self.image_statistics[features].dropna().index
        clustering_data = self.image_statistics.loc[valid_indices, features]
        
        if clustering_data.empty:
            print("Error: No valid data for clustering")
            return
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Perform K-means clustering (still do this for analysis purposes)
        optimal_k = 3  # Based on environmental conditions (temperature ranges)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        # Get experiment conditions for coloring
        conditions = self.image_statistics.loc[valid_indices, 'condition'].values
        # Create a mapping of conditions to numeric values
        unique_conditions = sorted(self.image_statistics['condition'].unique())
        condition_to_num = {cond: i for i, cond in enumerate(unique_conditions)}
        condition_nums = [condition_to_num[cond] for cond in conditions]
        
        # Visualization
        plt.figure(figsize=(14, 6))
        
        # PCA results colored by experiment condition
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=condition_nums, 
                            cmap='tab10', alpha=0.7, s=50)
        plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA of Image Characteristics by Experiment Condition', fontweight='bold')
        
        # Create custom colorbar with condition labels
        cbar = plt.colorbar(scatter, label='Experiment Condition', ticks=range(len(unique_conditions)))
        cbar.ax.set_yticklabels([cond.replace('condition_', '') for cond in unique_conditions])
        
        # Feature importance
        plt.subplot(1, 2, 2)
        feature_importance = np.abs(pca.components_[0])
        plt.barh(features, feature_importance)
        plt.xlabel('Absolute Loading on PC1')
        plt.title('Feature Importance in Principal Component 1', fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('clustering_and_pca_analysis.svg', format='svg', bbox_inches='tight')
        plt.savefig('clustering_and_pca_analysis.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return cluster_labels, pca_result
    
    def generate_comprehensive_report(self):
        """Generate comprehensive statistical report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("Multi-modal Environmental Water Sample Dataset")
        report.append("=" * 80)
        
        # Dataset overview
        report.append("\n1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total conditions analyzed: {len(self.conditions)}")
        report.append(f"Water samples: {len(self.water_samples)}")
        report.append(f"Replicates per sample: {self.replicates}")
        report.append(f"Total image pairs: {len(self.image_statistics)}")
        
        # Environmental conditions summary
        report.append("\n2. ENVIRONMENTAL CONDITIONS")
        report.append("-" * 40)
        temp_summary = self.environmental_data.groupby('condition_letter').agg({
            'temp_mean': ['min', 'max', 'mean'],
            'humidity_mean': ['min', 'max', 'mean']
        }).round(2)
        report.append(temp_summary.to_string())
        
        # Image statistics summary
        if not self.image_statistics.empty:
            report.append("\n3. IMAGE CHARACTERISTICS SUMMARY")
            report.append("-" * 40)
            image_summary = self.image_statistics.describe()
            report.append(image_summary.round(3).to_string())
        
        # Water composition analysis
        if not self.water_composition_data.empty:
            report.append("\n4. WATER COMPOSITION ANALYSIS")
            report.append("-" * 40)
            # Calculate ionic strength: I = 0.5 * Σ(c_i * z_i^2)
            # Ion charges for each component (considering dissociation)
            # NaHCO3 -> Na+(1) + HCO3-(1); CaCl2 -> Ca2+(2) + 2Cl-(1)
            # MgCl2 -> Mg2+(2) + 2Cl-(1); Na2SO4 -> 2Na+(1) + SO4 2-(2)
            # NaH2PO4 -> Na+(1) + H2PO4-(1); KF -> K+(1) + F-(1)
            # Fe(NO3)3 -> Fe3+(3) + 3NO3-(1); CuSO4 -> Cu2+(2) + SO4 2-(2)
            ionic_strength_factors = {
                'NaHCO3': 0.5 * (1**2 + 1**2),          # Na+ + HCO3-
                'CaCl2': 0.5 * (2**2 + 2 * 1**2),       # Ca2+ + 2Cl-
                'MgCl2': 0.5 * (2**2 + 2 * 1**2),       # Mg2+ + 2Cl-
                'Na2SO4': 0.5 * (2 * 1**2 + 2**2),      # 2Na+ + SO42-
                'NaH2PO4': 0.5 * (1**2 + 1**2),         # Na+ + H2PO4-
                'KF': 0.5 * (1**2 + 1**2),              # K+ + F-
                'Fe(NO3)3': 0.5 * (3**2 + 3 * 1**2),    # Fe3+ + 3NO3-
                'CuSO4': 0.5 * (2**2 + 2**2),           # Cu2+ + SO42-
            }
            report.append(f"Ionic strength by sample (I = 0.5 * Σ c_i * z_i^2):")
            for sample in self.water_composition_data.index:
                I = 0.0
                for component, factor in ionic_strength_factors.items():
                    if component in self.water_composition_data.columns:
                        I += self.water_composition_data.loc[sample, component] * factor
                report.append(f"  {sample}: {I:.3f} mM")
        
        # Save report
        with open('comprehensive_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Comprehensive report generated: comprehensive_analysis_report.txt")
        return report
    
    def create_publication_figures(self):
        """Create publication-ready figures"""
        # Figure 1: Environmental conditions matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature distribution
        temp_data = self.environmental_data.groupby('condition_letter')['temp_mean'].first()
        axes[0, 0].bar(temp_data.index, temp_data.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_xlabel('Environmental Condition')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].set_title('Temperature Distribution Across Conditions', fontweight='bold')
        
        # Humidity distribution
        humidity_data = self.environmental_data.groupby('condition_letter')['humidity_mean'].first()
        axes[0, 1].bar(humidity_data.index, humidity_data.values, color='lightcoral', alpha=0.8)
        axes[0, 1].set_xlabel('Environmental Condition')
        axes[0, 1].set_ylabel('Relative Humidity (%)')
        axes[0, 1].set_title('Humidity Distribution Across Conditions', fontweight='bold')
        
        # Water composition heatmap
        if not self.water_composition_data.empty:
            im = axes[1, 0].imshow(self.water_composition_data.T.values, cmap='YlOrRd', aspect='auto')
            axes[1, 0].set_xticks(range(len(self.water_samples)))
            axes[1, 0].set_xticklabels(self.water_samples, rotation=45)
            axes[1, 0].set_yticks(range(len(self.water_composition_data.columns)))
            axes[1, 0].set_yticklabels(self.water_composition_data.columns)
            axes[1, 0].set_title('Water Sample Composition Matrix (mM)', fontweight='bold')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Image characteristics distribution
        if not self.image_statistics.empty:
            axes[1, 1].hist(self.image_statistics['normal_brightness'], alpha=0.6, label='Normal Images', bins=20)
            axes[1, 1].hist(self.image_statistics['sem_brightness'], alpha=0.6, label='SEM Images', bins=20)
            axes[1, 1].set_xlabel('Brightness')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Image Brightness Distribution', fontweight='bold')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('publication_figure_1_dataset_overview.svg', format='svg', bbox_inches='tight')
        plt.savefig('publication_figure_1_dataset_overview.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete statistical analysis pipeline"""
        print("Starting comprehensive dataset analysis...")
        
        # Load and prepare data
        self.load_water_composition_data()
        self.create_environmental_matrix()
        self.analyze_image_basic_statistics()
        
        # Perform analyses
        merged_data, correlation_matrix = self.perform_environmental_correlation_analysis()
        composition_analysis = self.analyze_water_sample_effects()
        cluster_labels, pca_result = self.perform_clustering_analysis()
        
        # Generate outputs
        self.create_publication_figures()
        report = self.generate_comprehensive_report()
        
        print("\nAnalysis completed successfully!")
        print("Generated files:")
        print("- environmental_correlation_matrix.svg/png")
        print("- clustering_and_pca_analysis.svg/png")
        print("- publication_figure_1_dataset_overview.svg/png")
        print("- comprehensive_analysis_report.txt")
        
        return {
            'merged_data': merged_data,
            'correlation_matrix': correlation_matrix,
            'composition_analysis': composition_analysis,
            'cluster_labels': cluster_labels,
            'pca_result': pca_result,
            'report': report
        }

if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = WaterSampleDatasetAnalyzer(base_path='corrected_images')
    results = analyzer.run_complete_analysis() 