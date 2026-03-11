#!/usr/bin/env python3
"""
Complete Computer Vision Analysis with Mathematical Formulations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from scipy import stats
from scipy.fftpack import fft2, fftshift
from skimage import feature, measure, filters, morphology, color
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import professional plotting style
from professional_plot_style import setup_professional_style, style_axis, get_professional_colors

# Set up professional style
setup_professional_style()
colors = get_professional_colors()

class CompleteComputerVisionAnalyzer:
    def __init__(self):
        self.conditions = ['condition_A', 'condition_B', 'condition_C', 'condition_D', 
                          'condition_E', 'condition_F', 'condition_G', 'condition_H', 'condition_I']
        self.corrected_base_path = './corrected_images/'
    
    def format_condition_label(self, condition):
        """Format condition label for display (e.g., 'condition_A' -> 'Condition A')"""
        if isinstance(condition, str) and condition.startswith('condition_'):
            return 'Condition ' + condition.split('_')[1]
        return str(condition)
    
    def format_feature_label(self, feature_name):
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
            return formatted
        return str(feature_name)
        
    def extract_texture_features_glcm(self, image):
        """Extract GLCM texture features"""
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image.astype(np.float64)
        
        gray = ((gray - gray.min()) * 255 / (gray.max() - gray.min())).astype(np.uint8)
        
        glcm_features = {}
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for dist in distances:
            for angle in angles:
                try:
                    glcm = feature.graycomatrix(gray, [dist], [angle], 256, symmetric=True, normed=True)
                    
                    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                    energy = feature.graycoprops(glcm, 'energy')[0, 0]
                    
                    key = f'd{dist}_a{int(angle*180/np.pi)}'
                    glcm_features.update({
                        f'contrast_{key}': contrast,
                        f'dissimilarity_{key}': dissimilarity,
                        f'homogeneity_{key}': homogeneity,
                        f'energy_{key}': energy,
                    })
                except Exception as e:
                    print(f"Error calculating GLCM: {e}")
        
        return glcm_features
    
    def extract_frequency_domain_features(self, image):
        """Extract frequency domain features using FFT"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        fft = fft2(gray)
        fft_shifted = fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        max_radius = min(center)
        low_freq = magnitude_spectrum[radius < max_radius/3].mean()
        mid_freq = magnitude_spectrum[(radius >= max_radius/3) & (radius < 2*max_radius/3)].mean()
        high_freq = magnitude_spectrum[radius >= 2*max_radius/3].mean()
        
        total_energy = np.sum(magnitude_spectrum**2)
        low_freq_energy = np.sum(magnitude_spectrum[radius < max_radius/3]**2) / total_energy
        mid_freq_energy = np.sum(magnitude_spectrum[(radius >= max_radius/3) & (radius < 2*max_radius/3)]**2) / total_energy
        high_freq_energy = np.sum(magnitude_spectrum[radius >= 2*max_radius/3]**2) / total_energy
        
        return {
            'low_freq_magnitude': low_freq,
            'mid_freq_magnitude': mid_freq,
            'high_freq_magnitude': high_freq,
            'low_freq_energy_ratio': low_freq_energy,
            'mid_freq_energy_ratio': mid_freq_energy,
            'high_freq_energy_ratio': high_freq_energy,
            'total_spectral_energy': total_energy
        }
    
    def extract_morphological_features(self, image):
        """Extract morphological features"""
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        
        threshold = filters.threshold_otsu(gray)
        binary = gray > threshold
        
        disk_small = morphology.disk(3)
        eroded = morphology.erosion(binary, disk_small)
        dilated = morphology.dilation(binary, disk_small)
        
        features = {
            'object_area_ratio': np.sum(binary) / binary.size,
            'erosion_area_ratio': np.sum(eroded) / binary.size,
            'dilation_area_ratio': np.sum(dilated) / binary.size,
        }
        
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        if regions:
            areas = [region.area for region in regions]
            features.update({
                'num_objects': len(regions),
                'mean_object_area': np.mean(areas),
                'largest_object_area': max(areas),
            })
        else:
            features.update({
                'num_objects': 0,
                'mean_object_area': 0,
                'largest_object_area': 0,
            })
        
        return features
    
    def calculate_structural_similarity(self, img1, img2):
        """Calculate SSIM between two images"""
        if len(img1.shape) == 3:
            img1_gray = color.rgb2gray(img1)
        else:
            img1_gray = img1
        
        if len(img2.shape) == 3:
            img2_gray = color.rgb2gray(img2)
        else:
            img2_gray = img2
        
        min_shape = np.minimum(img1_gray.shape, img2_gray.shape)
        img1_resized = img1_gray[:min_shape[0], :min_shape[1]]
        img2_resized = img2_gray[:min_shape[0], :min_shape[1]]
        
        # Normalize to 0-1 range for SSIM calculation
        img1_resized = (img1_resized - img1_resized.min()) / (img1_resized.max() - img1_resized.min() + 1e-8)
        img2_resized = (img2_resized - img2_resized.min()) / (img2_resized.max() - img2_resized.min() + 1e-8)
        
        return ssim(img1_resized, img2_resized, data_range=1.0)
    
    def comprehensive_analysis(self):
        """Perform comprehensive analysis"""
        print("Starting comprehensive computer vision analysis...")
        
        all_features = []
        
        for condition in self.conditions:
            corrected_condition_path = os.path.join(self.corrected_base_path, condition)
            
            if not os.path.exists(corrected_condition_path):
                print(f"Warning: {corrected_condition_path} does not exist")
                continue
            
            print(f"Analyzing {condition}...")
            
            for i in range(1, 26):
                normal_image_path = os.path.join(corrected_condition_path, f'Image_{i}.jpg')
                sem_image_path = os.path.join(corrected_condition_path, f'SEM_{i}.jpg')
                
                if os.path.exists(normal_image_path) and os.path.exists(sem_image_path):
                    try:
                        normal_img = np.array(Image.open(normal_image_path))
                        sem_img = np.array(Image.open(sem_image_path))
                        
                        # Extract features
                        normal_texture = self.extract_texture_features_glcm(normal_img)
                        normal_frequency = self.extract_frequency_domain_features(normal_img)
                        normal_morphology = self.extract_morphological_features(normal_img)
                        
                        sem_texture = self.extract_texture_features_glcm(sem_img)
                        sem_frequency = self.extract_frequency_domain_features(sem_img)
                        sem_morphology = self.extract_morphological_features(sem_img)
                        
                        cross_modal_ssim = self.calculate_structural_similarity(normal_img, sem_img)
                        
                        feature_dict = {
                            'condition': condition,
                            'image_id': i,
                            'cross_modal_ssim': cross_modal_ssim,
                        }
                        
                        # Add features with prefixes
                        for key, value in normal_texture.items():
                            feature_dict[f'normal_texture_{key}'] = value
                        for key, value in normal_frequency.items():
                            feature_dict[f'normal_freq_{key}'] = value
                        for key, value in normal_morphology.items():
                            feature_dict[f'normal_morph_{key}'] = value
                        
                        for key, value in sem_texture.items():
                            feature_dict[f'sem_texture_{key}'] = value
                        for key, value in sem_frequency.items():
                            feature_dict[f'sem_freq_{key}'] = value
                        for key, value in sem_morphology.items():
                            feature_dict[f'sem_morph_{key}'] = value
                        
                        all_features.append(feature_dict)
                        
                    except Exception as e:
                        print(f"Error analyzing images: {e}")
        
        features_df = pd.DataFrame(all_features)
        features_df.to_csv('comprehensive_computer_vision_features.csv', index=False)
        
        print(f"Analysis completed. Features saved to comprehensive_computer_vision_features.csv")
        print(f"Total features: {len(features_df.columns)}, Total samples: {len(features_df)}")
        
        return features_df
    
    def create_visualizations(self, features_df):
        """Create professional visualizations in multiple figures"""
        print("Creating professional visualizations...")
        
        correction_df = pd.read_csv('image_correction_statistics.csv')
        
        # Figure 1: Image Correction and Quality Assessment
        fig1 = plt.figure(figsize=(16, 12))
        
        # 1. Brightness improvement
        plt.subplot(2, 3, 1)
        correction_summary = correction_df.groupby(['condition', 'type']).agg({
            'brightness_improvement': 'mean'
        }).reset_index()
        
        # Format condition labels for display
        correction_summary['condition_display'] = correction_summary['condition'].apply(self.format_condition_label)
        
        sns.barplot(data=correction_summary, x='condition_display', y='brightness_improvement', hue='type')
        plt.title('Brightness Improvement After Correction', fontweight='bold', fontsize=14)
        plt.xlabel('Environmental Condition', fontsize=12)
        plt.ylabel('Brightness Improvement', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Image Type', fontsize=10)
        
        # 2. Contrast improvement
        plt.subplot(2, 3, 2)
        contrast_summary = correction_df.groupby(['condition', 'type']).agg({
            'contrast_improvement': 'mean'
        }).reset_index()
        
        # Format condition labels for display
        contrast_summary['condition_display'] = contrast_summary['condition'].apply(self.format_condition_label)
        
        sns.barplot(data=contrast_summary, x='condition_display', y='contrast_improvement', hue='type')
        plt.title('Contrast Enhancement After Correction', fontweight='bold', fontsize=14)
        plt.xlabel('Environmental Condition', fontsize=12)
        plt.ylabel('Contrast Improvement', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Image Type', fontsize=10)
        
        # 3. SSIM distribution by condition
        plt.subplot(2, 3, 3)
        ssim_by_condition = features_df.groupby('condition')['cross_modal_ssim'].mean()
        # Format condition labels for display
        formatted_labels = [self.format_condition_label(idx) for idx in ssim_by_condition.index]
        bars = plt.bar(range(len(ssim_by_condition)), ssim_by_condition.values, color='steelblue', alpha=0.8)
        plt.title('Cross-modal SSIM by Condition', fontweight='bold', fontsize=14)
        plt.xlabel('Environmental Condition', fontsize=12)
        plt.ylabel('SSIM Index', fontsize=12)
        plt.xticks(range(len(formatted_labels)), formatted_labels, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. SSIM histogram
        plt.subplot(2, 3, 4)
        plt.hist(features_df['cross_modal_ssim'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('SSIM Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Cross-modal SSIM Distribution', fontweight='bold', fontsize=14)
        plt.axvline(features_df['cross_modal_ssim'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {features_df["cross_modal_ssim"].mean():.3f}')
        plt.legend()
        
        # 5. Correction effectiveness scatter
        plt.subplot(2, 3, 5)
        if 'brightness_improvement' in correction_df.columns and 'contrast_improvement' in correction_df.columns:
            plt.scatter(correction_df['brightness_improvement'], correction_df['contrast_improvement'], 
                       alpha=0.6, c=pd.Categorical(correction_df['condition']).codes, cmap='tab10')
            plt.xlabel('Brightness Improvement', fontsize=12)
            plt.ylabel('Contrast Improvement', fontsize=12)
            plt.title('Correction Effectiveness', fontweight='bold', fontsize=14)
            plt.colorbar(label='Condition')
        
        # 6. Image quality by type
        plt.subplot(2, 3, 6)
        if 'brightness_after' in correction_df.columns:
            correction_df.boxplot(column='brightness_after', by='type', ax=plt.gca())
            plt.title('Image Brightness After Correction by Type', fontweight='bold', fontsize=14)
            plt.suptitle('')  # Remove automatic suptitle
            plt.ylabel('Brightness', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('figure_1_image_quality_assessment.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_1_image_quality_assessment.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Figure 2: Frequency Domain Analysis
        fig2 = plt.figure(figsize=(16, 10))
        
        # 1. Normal images frequency distribution
        plt.subplot(2, 3, 1)
        freq_features = ['normal_freq_low_freq_energy_ratio', 'normal_freq_mid_freq_energy_ratio', 'normal_freq_high_freq_energy_ratio']
        if all(col in features_df.columns for col in freq_features):
            freq_data = features_df[freq_features].mean()
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            plt.pie(freq_data.values, labels=['Low Freq.', 'Mid Freq.', 'High Freq.'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Normal Images: Frequency Energy Distribution', fontweight='bold', fontsize=14)
        
        # 2. SEM images frequency distribution
        plt.subplot(2, 3, 2)
        sem_freq_features = ['sem_freq_low_freq_energy_ratio', 'sem_freq_mid_freq_energy_ratio', 'sem_freq_high_freq_energy_ratio']
        if all(col in features_df.columns for col in sem_freq_features):
            sem_freq_data = features_df[sem_freq_features].mean()
            plt.pie(sem_freq_data.values, labels=['Low Freq.', 'Mid Freq.', 'High Freq.'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('SEM Images: Frequency Energy Distribution', fontweight='bold', fontsize=14)
        
        # 3. Spectral energy comparison
        plt.subplot(2, 3, 3)
        if 'normal_freq_total_spectral_energy' in features_df.columns and 'sem_freq_total_spectral_energy' in features_df.columns:
            plt.scatter(features_df['normal_freq_total_spectral_energy'], 
                       features_df['sem_freq_total_spectral_energy'], alpha=0.6)
            plt.xlabel('Normal Images: Total Spectral Energy', fontsize=12)
            plt.ylabel('SEM Images: Total Spectral Energy', fontsize=12)
            plt.title('Spectral Energy Correlation', fontweight='bold', fontsize=14)
            
            # Add correlation coefficient
            corr = features_df[['normal_freq_total_spectral_energy', 'sem_freq_total_spectral_energy']].corr().iloc[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Frequency patterns by condition
        plt.subplot(2, 3, 4)
        if 'normal_freq_low_freq_energy_ratio' in features_df.columns:
            condition_freq = features_df.groupby('condition')['normal_freq_low_freq_energy_ratio'].mean()
            plt.plot(condition_freq.index, condition_freq.values, 'o-', linewidth=2, markersize=8)
            plt.title('Low Frequency Energy vs Conditions', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('Low Frequency Energy Ratio', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 5. Mid frequency analysis
        plt.subplot(2, 3, 5)
        if 'normal_freq_mid_freq_energy_ratio' in features_df.columns:
            condition_mid_freq = features_df.groupby('condition')['normal_freq_mid_freq_energy_ratio'].mean()
            plt.plot(condition_mid_freq.index, condition_mid_freq.values, 'o-', linewidth=2, markersize=8, color='green')
            plt.title('Mid Frequency Energy vs Conditions', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('Mid Frequency Energy Ratio', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 6. High frequency analysis
        plt.subplot(2, 3, 6)
        if 'normal_freq_high_freq_energy_ratio' in features_df.columns:
            condition_high_freq = features_df.groupby('condition')['normal_freq_high_freq_energy_ratio'].mean()
            plt.plot(condition_high_freq.index, condition_high_freq.values, 'o-', linewidth=2, markersize=8, color='red')
            plt.title('High Frequency Energy vs Conditions', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('High Frequency Energy Ratio', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figure_2_frequency_domain_analysis.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_2_frequency_domain_analysis.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Figure 3: Texture and Morphological Analysis
        fig3 = plt.figure(figsize=(16, 10))
        
        # 1. Texture contrast correlation
        plt.subplot(2, 3, 1)
        if 'normal_texture_contrast_d1_a0' in features_df.columns and 'sem_texture_contrast_d1_a0' in features_df.columns:
            plt.scatter(features_df['normal_texture_contrast_d1_a0'], features_df['sem_texture_contrast_d1_a0'], 
                       alpha=0.6, c=pd.Categorical(features_df['condition']).codes, cmap='tab10')
            plt.xlabel('Normal Images: Texture Contrast', fontsize=12)
            plt.ylabel('SEM Images: Texture Contrast', fontsize=12)
            plt.title('Texture Contrast Correlation', fontweight='bold', fontsize=14)
            plt.colorbar(label='Condition')
        
        # 2. Texture energy comparison
        plt.subplot(2, 3, 2)
        if 'normal_texture_energy_d1_a0' in features_df.columns and 'sem_texture_energy_d1_a0' in features_df.columns:
            plt.scatter(features_df['normal_texture_energy_d1_a0'], features_df['sem_texture_energy_d1_a0'], alpha=0.6)
            plt.xlabel('Normal Images: Texture Energy', fontsize=12)
            plt.ylabel('SEM Images: Texture Energy', fontsize=12)
            plt.title('Texture Energy Correlation', fontweight='bold', fontsize=14)
        
        # 3. Morphological features
        plt.subplot(2, 3, 3)
        morph_features = ['normal_morph_object_area_ratio', 'sem_morph_object_area_ratio']
        if all(col in features_df.columns for col in morph_features):
            box_data = [features_df[morph_features[0]].dropna(), features_df[morph_features[1]].dropna()]
            box_plot = plt.boxplot(box_data, labels=['Normal', 'SEM'], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightcoral')
            plt.title('Object Area Ratio Distribution', fontweight='bold', fontsize=14)
            plt.ylabel('Object Area Ratio', fontsize=12)
        
        # 4. Object count correlation
        plt.subplot(2, 3, 4)
        if 'normal_morph_num_objects' in features_df.columns and 'sem_morph_num_objects' in features_df.columns:
            plt.scatter(features_df['normal_morph_num_objects'], features_df['sem_morph_num_objects'], alpha=0.6)
            plt.xlabel('Normal Images: Number of Objects', fontsize=12)
            plt.ylabel('SEM Images: Number of Objects', fontsize=12)
            plt.title('Object Count Correlation', fontweight='bold', fontsize=14)
            
            # Add correlation coefficient
            valid_data = features_df[['normal_morph_num_objects', 'sem_morph_num_objects']].dropna()
            if len(valid_data) > 1:
                corr = valid_data.corr().iloc[0, 1]
                plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Texture homogeneity by condition
        plt.subplot(2, 3, 5)
        if 'normal_texture_homogeneity_d1_a0' in features_df.columns:
            condition_homogeneity = features_df.groupby('condition')['normal_texture_homogeneity_d1_a0'].mean()
            # Format condition labels for display
            formatted_labels = [self.format_condition_label(idx) for idx in condition_homogeneity.index]
            plt.bar(range(len(condition_homogeneity)), condition_homogeneity.values, alpha=0.8, color='purple')
            plt.title('Texture Homogeneity by Condition', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('Texture Homogeneity', fontsize=12)
            plt.xticks(range(len(formatted_labels)), formatted_labels, rotation=45)
        
        # 6. Morphological complexity
        plt.subplot(2, 3, 6)
        if 'normal_morph_largest_object_area' in features_df.columns and 'sem_morph_largest_object_area' in features_df.columns:
            plt.scatter(features_df['normal_morph_largest_object_area'], 
                       features_df['sem_morph_largest_object_area'], alpha=0.6)
            plt.xlabel('Normal Images: Largest Object Area', fontsize=12)
            plt.ylabel('SEM Images: Largest Object Area', fontsize=12)
            plt.title('Largest Object Area Correlation', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('figure_3_texture_morphological_analysis.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_3_texture_morphological_analysis.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Figure 4: Statistical Analysis and Feature Importance
        fig4 = plt.figure(figsize=(16, 10))
        
        # 1. PCA analysis
        plt.subplot(2, 3, 1)
        numerical_cols = features_df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'image_id']
        
        if len(numerical_cols) > 2:
            pca_data = features_df[numerical_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pca_data)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            conditions_numeric = pd.Categorical(features_df['condition']).codes
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=conditions_numeric, cmap='tab10', alpha=0.7, s=50)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            plt.title('Principal Component Analysis', fontweight='bold', fontsize=14)
            plt.colorbar(scatter, label='Condition')
        
        # 2. Feature importance (coefficient of variation)
        plt.subplot(2, 3, 2)
        numerical_features = features_df.select_dtypes(include=[np.number])
        cv_values = (numerical_features.std() / numerical_features.mean()).abs().sort_values(ascending=False)
        top_features = cv_values.head(10)
        
        plt.barh(range(len(top_features)), top_features.values, color='orange', alpha=0.7)
        # Format feature names for display
        formatted_names = [self.format_feature_label(name) for name in top_features.index]
        plt.yticks(range(len(top_features)), formatted_names)
        plt.xlabel('Coefficient of Variation', fontsize=12)
        plt.title('Top 10 Most Variable Features', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        
        # 3. Correlation matrix of key features
        plt.subplot(2, 3, 3)
        key_features = ['cross_modal_ssim']
        if 'normal_freq_total_spectral_energy' in features_df.columns:
            key_features.append('normal_freq_total_spectral_energy')
        if 'sem_freq_total_spectral_energy' in features_df.columns:
            key_features.append('sem_freq_total_spectral_energy')
        if 'normal_morph_object_area_ratio' in features_df.columns:
            key_features.append('normal_morph_object_area_ratio')
        if 'sem_morph_object_area_ratio' in features_df.columns:
            key_features.append('sem_morph_object_area_ratio')
        
        if len(key_features) > 1:
            available_features = [f for f in key_features if f in features_df.columns]
            if len(available_features) > 1:
                corr_matrix = features_df[available_features].corr()
                # Format feature names for display
                formatted_labels = [self.format_feature_label(name) for name in available_features]
                # Create heatmap with formatted labels
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, 
                           xticklabels=formatted_labels,
                           yticklabels=formatted_labels)
                plt.title('Key Features Correlation Matrix', fontweight='bold', fontsize=14)
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
        
        # 4. Feature distribution by condition
        plt.subplot(2, 3, 4)
        if 'normal_texture_energy_d1_a0' in features_df.columns:
            condition_texture = features_df.groupby('condition')['normal_texture_energy_d1_a0'].mean()
            # Format condition labels for display
            formatted_conditions = [self.format_condition_label(idx) for idx in condition_texture.index]
            plt.plot(range(len(condition_texture)), condition_texture.values, 'o-', linewidth=3, markersize=8, color='red')
            plt.title('Texture Energy vs Environmental Conditions', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('Texture Energy', fontsize=12)
            plt.xticks(range(len(formatted_conditions)), formatted_conditions, rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 5. Environmental gradient analysis
        plt.subplot(2, 3, 5)
        # Create a gradient analysis showing feature changes across conditions
        if 'cross_modal_ssim' in features_df.columns:
            ssim_stats = features_df.groupby('condition')['cross_modal_ssim'].agg(['mean', 'std']).reset_index()
            x_pos = range(len(ssim_stats))
            # Format condition labels for display
            formatted_conditions = [self.format_condition_label(cond) for cond in ssim_stats['condition']]
            plt.errorbar(x_pos, ssim_stats['mean'], yerr=ssim_stats['std'], 
                        fmt='o-', linewidth=2, markersize=8, capsize=5)
            plt.xticks(x_pos, formatted_conditions, rotation=45)
            plt.title('SSIM Mean ± Std by Condition', fontweight='bold', fontsize=14)
            plt.xlabel('Environmental Condition', fontsize=12)
            plt.ylabel('Cross-modal SSIM', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        plt.subplot(2, 3, 6)
        # Create a summary table as text
        plt.axis('off')
        summary_text = f"""
Dataset Summary Statistics:

Total image pairs analyzed: {len(features_df)}
Number of conditions: {len(features_df['condition'].unique())}
Total features extracted: {len(features_df.columns) - 2}

Cross-modal SSIM:
Mean: {features_df['cross_modal_ssim'].mean():.4f}
Std: {features_df['cross_modal_ssim'].std():.4f}
Min: {features_df['cross_modal_ssim'].min():.4f}
Max: {features_df['cross_modal_ssim'].max():.4f}
"""
        plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                verticalalignment='center')
        plt.title('Dataset Summary', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('figure_4_statistical_analysis.svg', format='svg', bbox_inches='tight')
        plt.savefig('figure_4_statistical_analysis.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print("Professional visualizations saved:")
        print("- figure_1_image_quality_assessment.svg/png")
        print("- figure_2_frequency_domain_analysis.svg/png")
        print("- figure_3_texture_morphological_analysis.svg/png")
        print("- figure_4_statistical_analysis.svg/png")

if __name__ == "__main__":
    analyzer = CompleteComputerVisionAnalyzer()
    features_df = analyzer.comprehensive_analysis()
    analyzer.create_visualizations(features_df)
    print("Complete analysis finished!")
