# Water Sample Analysis Workflow Guide

## Overview
This guide explains the purpose of each Python script and the recommended order to run them for complete water sample analysis.

## Required Files

### 1. **professional_plot_style.py** (Utility Module)
- **Purpose**: Provides professional plotting styles and color schemes for all visualizations
- **Dependencies**: matplotlib, seaborn
- **Usage**: Imported by other scripts, not run directly

### 2. **dataset_statistical_analysis.py**
- **Purpose**: Basic statistical analysis of the water sample dataset
- **Functionality**:
  - Creates environmental matrix with temperature and humidity data
  - Analyzes basic image statistics (brightness, contrast, etc.)
  - Performs correlation analysis between environmental conditions and image characteristics
  - Performs clustering analysis (K-means) and PCA
  - Generates comprehensive statistical report
- **Output Files**:
  - `environmental_correlation_matrix.svg/png`
  - `clustering_and_pca_analysis.svg/png`
  - `publication_figure_1_dataset_overview.svg/png`
  - `comprehensive_analysis_report.txt`

### 3. **complete_cv_analysis.py**
- **Purpose**: Comprehensive computer vision feature extraction and analysis
- **Functionality**:
  - Extracts texture features using GLCM (Gray Level Co-occurrence Matrix)
  - Analyzes frequency domain features using FFT
  - Performs morphological analysis
  - Calculates cross-modal SSIM between normal and SEM images
  - Analyzes image correction statistics
- **Output Files**:
  - `comprehensive_computer_vision_features.csv`
  - `figure_1_image_quality_assessment.svg/png`
  - `figure_2_frequency_domain_analysis.svg/png`
  - `figure_3_texture_morphological_analysis.svg/png`
  - `figure_4_statistical_analysis.svg/png`

### 4. **improved_cv_analysis.py**
- **Purpose**: Enhanced version of computer vision analysis with additional features
- **Functionality**:
  - All features from complete_cv_analysis.py
  - Additional advanced analysis methods
  - Improved visualization layouts
  - More detailed statistical analysis
- **Output Files**: Same as complete_cv_analysis.py but with enhanced visualizations

### 5. **advanced_visualizations.py**
- **Purpose**: Creates advanced 3D and network visualizations
- **Functionality**:
  - 3D feature space visualization
  - Network analysis of feature relationships
  - Advanced statistical plots
  - Comprehensive summary dashboard
- **Output Files**:
  - `figure_4_3d_feature_space.svg/png`
  - `figure_5_network_analysis.svg/png`
  - `figure_6_advanced_statistics.svg/png`
  - `figure_7_summary_dashboard.svg/png`

### 6. **generate_heatmaps.py**
- **Purpose**: Generates correlation heatmaps and feature visualizations
- **Functionality**:
  - Creates correlation heatmap of computer vision features
  - Generates feature values heatmap by environmental condition
  - Creates image characteristics summary heatmap
  - Produces comprehensive analysis plots
- **Output Files**:
  - `heatmap_computer_vision_features.svg/png`
  - `heatmap_features_by_condition.svg/png`
  - `heatmap_image_characteristics_summary.svg/png`
  - `image_characteristics_analysis.svg/png`

## Recommended Execution Order

### Step 1: Basic Statistical Analysis
```bash
python dataset_statistical_analysis.py
```
This creates the initial environmental data analysis and basic statistics.

### Step 2: Computer Vision Feature Extraction
Choose one of the following (improved_cv_analysis.py is recommended):
```bash
python improved_cv_analysis.py
# OR
python complete_cv_analysis.py
```
This extracts all computer vision features and creates the CSV file needed for subsequent analyses.

### Step 3: Advanced Visualizations
```bash
python advanced_visualizations.py
```
This creates advanced 3D plots and network analyses using the features from Step 2.

### Step 4: Generate Heatmaps
```bash
python generate_heatmaps.py
```
This creates correlation heatmaps and summary visualizations using data from previous steps.

## Data Dependencies

- **Input Data**: The scripts expect image data in `corrected_images/` directory with the following structure:
  ```
  corrected_images/
  ├── condition_A/
  │   ├── Image_1.jpg to Image_25.jpg
  │   └── SEM_1.jpg to SEM_25.jpg
  ├── condition_B/
  │   └── ... (same structure)
  └── ... (conditions A through I)
  ```

- **Data Flow**:
  1. `dataset_statistical_analysis.py` → creates `comprehensive_analysis_report.txt`
  2. `complete_cv_analysis.py` or `improved_cv_analysis.py` → creates `comprehensive_computer_vision_features.csv`
  3. `advanced_visualizations.py` → uses `comprehensive_computer_vision_features.csv`
  4. `generate_heatmaps.py` → uses both `comprehensive_computer_vision_features.csv` and `comprehensive_analysis_report.txt`

## Notes

- All scripts use the `professional_plot_style.py` module for consistent visualization styling
- The scripts are designed to work with 9 environmental conditions (A-I) and 25 image pairs per condition
- SVG and PNG versions of all plots are generated for both web display and publication use
- Error handling is included for missing data or features

## Troubleshooting

1. **Import Errors**: Ensure all required packages are installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy pillow opencv-python scikit-image networkx
   ```

2. **File Not Found Errors**: Check that:
   - Image data is in the `corrected_images/` directory
   - Previous steps have been run to generate required input files

3. **Memory Issues**: For large datasets, you may need to:
   - Process conditions in batches
   - Reduce image resolution
   - Use a machine with more RAM