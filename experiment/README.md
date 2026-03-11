# Multi-modal Environmental Water Sample Dataset

## Project Overview

This dataset represents a comprehensive study of environmental water samples using multi-modal imaging approaches. The project investigates the relationship between environmental conditions (temperature and humidity), water sample compositions, and imaging characteristics through both conventional cellular phone photography and high-resolution Scanning Electron Microscopy (SEM).

## Dataset Structure

### Data Organization
```
experiment/
├── condition_A/          # Environmental condition A (20-23°C, 35-40% RH)
├── condition_B/          # Environmental condition B (20-23°C, 40-45% RH)
├── condition_C/          # Environmental condition C (20-23°C, 45-50% RH)
├── condition_D/          # Environmental condition D (23-26°C, 35-40% RH)
├── condition_E/          # Environmental condition E (23-26°C, 40-45% RH)
├── condition_F/          # Environmental condition F (23-26°C, 45-50% RH)
├── condition_G/          # Environmental condition G (26-29°C, 35-40% RH)
├── condition_H/          # Environmental condition H (26-29°C, 40-45% RH)
├── condition_I/          # Environmental condition I (26-29°C, 45-50% RH)
├── sample_data_sheet.csv # Michigan water quality data (2017-2019)
└── SEM_parameters.txt    # SEM imaging parameters and conditions
```

### Image Files
Each condition folder contains:
- **Image_1.jpg to Image_25.jpg**: Normal cellular phone images
- **SEM_1.jpg to SEM_25.jpg**: Corresponding SEM images

Total: **225 paired image sets** (9 conditions × 25 images per condition)

## Environmental Conditions

The study employed a 3×3 factorial design:

| Condition | Temperature (°C) | Relative Humidity (%) |
|-----------|------------------|----------------------|
| A         | 20-23           | 35-40                |
| B         | 20-23           | 40-45                |
| C         | 20-23           | 45-50                |
| D         | 23-26           | 35-40                |
| E         | 23-26           | 40-45                |
| F         | 23-26           | 45-50                |
| G         | 26-29           | 35-40                |
| H         | 26-29           | 40-45                |
| I         | 26-29           | 45-50                |

## Water Sample Compositions

Five synthetic water samples were prepared based on Michigan water quality reports (2017-2019):

| Sample | NaHCO₃ | CaCl₂ | MgCl₂ | Na₂SO₄ | NaH₂PO₄ | KF  | Fe(NO₃)₃ | CuSO₄   | Ionic Strength |
|--------|---------|--------|--------|---------|----------|-----|-----------|---------|----------------|
| A      | 0.1     | 1.5    | 0.5    | 0.35    | 0.033    | 0.4 | 0.005     | 0.00024 | 2.888 mM       |
| B      | 0.2     | 1.0    | 0.35   | 0.35    | 0.033    | 0.4 | 0.005     | 0.00024 | 2.338 mM       |
| C      | 0.1     | 0.5    | 0.2    | 0.35    | 0.033    | 0.4 | 0.005     | 0.00024 | 1.588 mM       |
| D      | 0.0     | 1.0    | 1.0    | 1.35    | 0.033    | 0.4 | 0.005     | 0.00024 | 3.788 mM       |
| E      | 0.0     | 1.0    | 0.5    | 2.35    | 0.033    | 0.4 | 0.005     | 0.00024 | 4.288 mM       |

*All concentrations in mM (millimolar)*

## Experimental Design

- **Replicates**: 5 per water sample per environmental condition
- **Total experimental units**: 225 (9 conditions × 5 samples × 5 replicates)
- **Image pairs per unit**: 1 normal + 1 SEM image
- **Total images**: 450 (225 pairs)

## Instrumentation

### SEM Imaging
- **Instrument**: JEOL 6610LV SEM system
- **Accelerating voltage**: 20 kV
- **Magnification range**: 5× to 50,000×
- **Voltage adjustment range**: 300 V to 30 kV

### Conventional Imaging
- **Method**: Standardized cellular phone photography
- **Conditions**: Controlled lighting, consistent exposure parameters

## Statistical Analysis Results

### Key Findings

1. **Significant Environmental Effects**:
   - Normal image brightness: F = 43.705, p < 0.001
   - SEM image contrast: F = 20.890, p < 0.001

2. **Image Characteristics**:
   - Normal image brightness: 18.96 ± 6.04 (arbitrary units)
   - SEM image brightness: 108.98 ± 23.63
   - SEM/Normal brightness ratio: 6.51 ± 2.88
   - SEM entropy: 3.58 ± 0.56

3. **Multi-modal Correlations**:
   - Correlation coefficients: 0.45 - 0.78 between normal and SEM parameters
   - Strong relationships between imaging modalities

## Computer Vision Analysis

### Advanced Image Processing Pipeline

The dataset includes comprehensive computer vision analysis with mathematical formulations implemented in `complete_cv_analysis.py`. This analysis performs:

#### 1. Image Preprocessing and Correction
All 225 image pairs undergo systematic correction to address:
- **White Balance Correction** using Gray World algorithm
- **Gamma Correction** for brightness enhancement (γ = 1.2)
- **Adaptive Histogram Equalization** (CLAHE) for contrast improvement
- **Bilateral Filtering** for noise reduction while preserving edges

#### 2. Feature Extraction (91 features per image pair)

**Texture Analysis (64 GLCM features, 32 per modality)**:
- Gray Level Co-occurrence Matrix (GLCM) features for both normal and SEM images
- Haralick descriptors: contrast, dissimilarity, homogeneity, energy
- Multiple distances (1, 2 pixels) and orientations (0°, 45°, 90°, 135°)

**Frequency Domain Analysis (14 features, 7 per modality)**:
- 2D Discrete Fourier Transform for both normal and SEM images
- Spectral energy distribution (low, mid, high frequency bands)
- Total spectral energy and frequency ratios

**Morphological Features (12 features, 6 per modality)**:
- Binary morphological operations (erosion, dilation)
- Object area ratios and connected component analysis
- Shape complexity and size distribution metrics

**Cross-modal Metrics (1 feature)**:
- Structural Similarity Index (SSIM) between normal and SEM images

#### 3. Professional Computer Vision Techniques

**Gray Level Co-occurrence Matrix (GLCM)**:
- *Definition*: Statistical method analyzing spatial relationships between pixel intensities
- *Purpose*: Quantifies texture characteristics by measuring how often pairs of pixel values occur at specific spatial relationships
- *Applications*: Medical imaging (tumor detection), satellite imagery (land classification), material science (surface analysis), quality control
- *Mathematical formulation*: `GLCM(i,j) = Σ{(x,y): I(x,y)=i, I(x+dx,y+dy)=j}`
- *Output features*: Contrast (local variations), Energy (uniformity), Homogeneity (closeness), Correlation (linear dependencies)
- *Advantages*: Rotation-invariant texture description, established standard in image analysis

**Structural Similarity Index Measure (SSIM)**:
- *Definition*: Perceptual metric measuring structural similarity between two images
- *Purpose*: Evaluates image quality by comparing luminance, contrast, and structural patterns
- *Applications*: Image compression assessment, video quality evaluation, medical image registration, multi-modal image analysis
- *Mathematical formulation*: `SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / (μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂)`
- *Range*: -1 to +1 (higher values indicate greater similarity)
- *Advantages*: Better correlation with human visual perception than PSNR, considers structural information

**2D Discrete Fourier Transform (2D-DFT)**:
- *Definition*: Mathematical technique converting spatial domain images to frequency domain
- *Purpose*: Analyzes periodic patterns, noise characteristics, and detail distribution in images
- *Applications*: Image filtering, pattern recognition, compression, enhancement, defect detection
- *Mathematical formulation*: `F(u,v) = ΣₓΣᵧ f(x,y) exp(-j2π(ux/M + vy/N))`
- *Output*: Magnitude spectrum showing frequency content distribution
- *Interpretation*: Low frequencies = smooth regions, High frequencies = edges and details

**Mathematical Morphology**:
- *Definition*: Set-theoretic approach to image analysis using geometric structures
- *Purpose*: Analyzes shape and size of objects through erosion, dilation, opening, closing operations
- *Applications*: Biomedical imaging (cell counting), industrial inspection (defect detection), remote sensing (object extraction)
- *Operations*: Erosion (shrinking), Dilation (expanding), Opening (smoothing), Closing (filling gaps)
- *Structuring element*: Geometric shape defining the operation's characteristics
- *Advantages*: Robust to noise, preserves object topology, quantitative shape analysis

**Principal Component Analysis (PCA)**:
- *Definition*: Dimensionality reduction technique finding principal directions of data variance
- *Purpose*: Identifies most informative feature combinations and reduces computational complexity
- *Applications*: Feature selection, data visualization, pattern recognition, noise reduction
- *Output*: Principal components explaining decreasing amounts of data variance
- *Interpretation*: PC1 captures most variation, PC2 captures second most, etc.
- *Benefits*: Eliminates redundancy, reveals hidden data structures, enables visualization

#### 4. Mathematical Formulations

**White Balance Correction (Gray World Algorithm)**:
```
I_est = (1/N) * Σ(I_r, I_g, I_b)
K_r = I_gray/I_r, K_g = I_gray/I_g, K_b = I_gray/I_b
I_corrected = I_original ⊙ K
```
*Assumption*: Average reflectance across the image is achromatic (gray)

**Gamma Correction (Power-law Transformation)**:
```
I_out = (I_in/255)^(1/γ) × 255, where γ = 1.2
```
*Purpose*: Compensates for non-linear intensity response in imaging systems

**GLCM Haralick Features**:
```
Contrast = Σᵢⱼ(i-j)² × GLCM(i,j)        [measures local variations]
Energy = Σᵢⱼ[GLCM(i,j)]²                [measures uniformity]
Homogeneity = Σᵢⱼ GLCM(i,j)/(1+|i-j|)   [measures closeness]
Correlation = Σᵢⱼ(ij×GLCM(i,j)-μₓμᵧ)/(σₓσᵧ) [measures linear dependencies]
```

**Frequency Domain Energy Distribution**:
```
E_total = Σᵤᵥ |F(u,v)|²
E_low = Σ(r<R/3) |F(u,v)|²     [smooth regions]
E_mid = Σ(R/3≤r<2R/3) |F(u,v)|² [texture patterns]  
E_high = Σ(r≥2R/3) |F(u,v)|²   [edges and noise]
```

**Morphological Operations**:
```
Erosion: (A ⊖ B)(x) = min{A(x+b) : b ∈ B}    [shrinks objects]
Dilation: (A ⊕ B)(x) = max{A(x-b) : b ∈ B}   [expands objects]
Opening: A ∘ B = (A ⊖ B) ⊕ B                [removes small objects]
Closing: A • B = (A ⊕ B) ⊖ B                [fills gaps]
```

#### 5. Statistical Analysis Techniques

**Analysis of Variance (ANOVA)**:
- *Definition*: Statistical method testing whether group means are significantly different from each other
- *Purpose*: Determines if environmental conditions significantly affect image features and quantifies effect size
- *Applications*: Experimental design analysis, factor effect assessment, group comparison in scientific studies
- *F-statistic interpretation*: F > critical value indicates significant differences between groups (p < 0.05 threshold)
- *Effect size (η²)*: Proportion of total variance explained by the factor (0.01=small, 0.06=medium, 0.14=large effect)
- *Post-hoc tests*: Multiple comparisons (Tukey HSD, Bonferroni) to identify which specific groups differ
- *Mathematical formulation*: `F = MSbetween / MSwithin` where MS = Mean Square

**Pearson Correlation Coefficient**:
- *Definition*: Measures linear relationship strength and direction between two continuous variables  
- *Purpose*: Quantifies how well two variables move together linearly and identifies feature relationships
- *Applications*: Feature selection, relationship identification, multicollinearity detection, data exploration
- *Range*: -1 to +1 (0=no linear relationship, ±1=perfect linear relationship)
- *Interpretation*: |r| > 0.7 (strong), 0.3-0.7 (moderate), 0.1-0.3 (weak), < 0.1 (negligible)
- *Assumptions*: Linear relationship, normal distribution, homoscedasticity, no extreme outliers
- *Mathematical formulation*: `r = Σ(xi-x̄)(yi-ȳ) / √[Σ(xi-x̄)²Σ(yi-ȳ)²]`

**Hierarchical Clustering**:
- *Definition*: Builds tree-like cluster structure by iteratively merging/splitting groups based on similarity
- *Purpose*: Identifies natural groupings in data without predefined number of clusters
- *Applications*: Pattern discovery, taxonomy creation, data exploration, condition grouping
- *Linkage methods*: Single (minimum distance), Complete (maximum distance), Average (mean distance), Ward (minimize variance)
- *Distance metrics*: Euclidean, Manhattan, Cosine similarity for different data types
- *Dendrogram interpretation*: Height indicates dissimilarity level, horizontal cut determines final cluster number
- *Advantages*: No assumption about cluster number, reveals hierarchical structure, deterministic results

**Cross-correlation Analysis**:
- *Definition*: Measures similarity between two signal sequences as function of displacement lag
- *Purpose*: Quantifies relationships between multi-modal image features and temporal/spatial alignment
- *Applications*: Signal processing, pattern matching, time series analysis, image registration, modal correspondence
- *Range*: 0 to 1 (1 indicates perfect positive correlation)
- *Lag analysis*: Identifies optimal alignment between signal sequences to maximize correlation
- *Mathematical formulation*: `R(τ) = Σ x(t)y(t+τ)` normalized by sequence lengths

**Coefficient of Variation (CV)**:
- *Definition*: Standardized measure of dispersion relative to the mean of a distribution
- *Purpose*: Compares variability between features with different scales and identifies most discriminative features
- *Applications*: Feature selection, quality control, process monitoring, risk assessment
- *Calculation*: `CV = σ/μ` (standard deviation divided by mean)
- *Interpretation*: Higher CV indicates greater relative variability and potentially higher discriminative power
- *Advantages*: Scale-independent, enables comparison across different measurement units

**Multivariate Statistical Modeling**:
- *Definition*: Analysis of multiple dependent and independent variables simultaneously in one model
- *Purpose*: Identifies complex relationships, interactions, and predictive patterns in high-dimensional data
- *Applications*: Classification, regression, feature importance ranking, interaction effects, predictive modeling
- *Techniques*: Multiple regression, discriminant analysis, MANOVA, canonical correlation
- *Model selection*: Cross-validation, information criteria (AIC/BIC), performance metrics (accuracy, R²)
- *Overfitting prevention*: Regularization (L1/L2), feature selection, validation strategies, early stopping

### Generated Analysis Files

#### Statistical Analysis Outputs:
1. **`comprehensive_analysis_report.txt`**: Basic statistical summary
2. **`environmental_correlation_matrix.png`**: Environmental correlation heatmap
3. **`clustering_and_pca_analysis.png`**: Clustering and PCA results  
4. **`publication_figure_1_dataset_overview.png`**: Dataset overview figure
5. **`dataset_statistical_analysis.py`**: Basic analysis script

#### Computer Vision Analysis Outputs:

**Data Files**:
6. **`comprehensive_computer_vision_features.csv`**: Complete feature dataset (225 samples × 91 features)
7. **`image_correction_statistics.csv`**: Image preprocessing performance metrics
8. **`mathematical_analysis_report.txt`**: Mathematical formulations and analysis results

**Professional Scientific Visualizations**:

9. **`figure_1_image_quality_assessment.png`**: Image correction and quality analysis (2×3 subplot layout)
   
   **Subplot 1 - Brightness Improvement After Correction**:
   - *X-axis*: Environmental conditions (A through I)
   - *Y-axis*: Brightness improvement (arbitrary units, 0-30 range)
   - *Calculation*: `brightness_after - brightness_before` for each image
   - *Interpretation*: Higher values = better correction. Good results show consistent improvement (>10 units) across all conditions. Poor results show negative values or high variance.
   - *Color coding*: Blue bars = Normal images, Orange bars = SEM images
   
   **Subplot 2 - Contrast Enhancement After Correction**:
   - *X-axis*: Environmental conditions (A through I)
   - *Y-axis*: Contrast improvement (arbitrary units, 0-25 range)
   - *Calculation*: `contrast_after - contrast_before` using standard deviation of pixel intensities
   - *Interpretation*: Positive values indicate successful enhancement. Good: >5 units improvement. Poor: <2 units or negative values.
   
   **Subplot 3 - Cross-modal SSIM by Condition**:
   - *X-axis*: Environmental conditions (A through I)
   - *Y-axis*: Structural Similarity Index (0.0-1.0 scale)
   - *Calculation*: SSIM between corresponding normal and SEM images using formula: `SSIM = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / (μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂)`
   - *Interpretation*: Higher SSIM = better cross-modal correspondence. Good: >0.3, Fair: 0.1-0.3, Poor: <0.1
   - *Value labels*: Numerical SSIM values displayed above each bar
   
   **Subplot 4 - Cross-modal SSIM Distribution**:
   - *X-axis*: SSIM values (0.0-0.4 range)
   - *Y-axis*: Frequency count of image pairs
   - *Calculation*: Histogram of all 225 SSIM values
   - *Interpretation*: Normal distribution around mean indicates consistent cross-modal relationships. Skewed distributions suggest systematic imaging differences.
   - *Red dashed line*: Dataset mean (≈0.202)
   
   **Subplot 5 - Correction Effectiveness Scatter**:
   - *X-axis*: Brightness improvement (0-30 units)
   - *Y-axis*: Contrast improvement (0-25 units)
   - *Color coding*: Different colors represent environmental conditions
   - *Interpretation*: Points in upper-right quadrant show both brightness and contrast improvement. Clustering by color indicates condition-specific correction patterns.
   
   **Subplot 6 - Image Brightness After Correction by Type**:
   - *X-axis*: Image type (Normal vs SEM)
   - *Y-axis*: Final brightness values after correction
   - *Box plot elements*: Median (line), quartiles (box), outliers (points)
   - *Interpretation*: SEM images typically show higher brightness values. Good results show similar distributions between types.

10. **`figure_2_frequency_domain_analysis.png`**: Spectral analysis results (2×3 subplot layout)
    
    **Subplot 1 - Normal Images: Frequency Energy Distribution**:
    - *Pie chart*: Relative energy in low/mid/high frequency bands
    - *Calculation*: 2D FFT → `E_band = Σ|F(u,v)|²` for each frequency band
    - *Frequency bands*: Low (<R/3), Mid (R/3 to 2R/3), High (>2R/3) where R = max radius
    - *Interpretation*: Natural images typically show 80-90% low-frequency energy. Higher high-frequency content indicates more detail/noise.
    
    **Subplot 2 - SEM Images: Frequency Energy Distribution**:
    - *Same as Subplot 1 for SEM images*
    - *Expected difference*: SEM images show more high-frequency content due to fine structural details
    - *Good results*: Moderate high-frequency content (20-40%). Poor: Either too smooth (<10%) or too noisy (>60%).
    
    **Subplot 3 - Spectral Energy Correlation**:
    - *X-axis*: Normal images total spectral energy (log scale, ~10¹⁵-10¹⁷)
    - *Y-axis*: SEM images total spectral energy (log scale, ~10¹⁷-10¹⁸)
    - *Calculation*: `E_total = Σᵤᵥ |F(u,v)|²` across entire frequency spectrum
    - *Interpretation*: Positive correlation indicates consistent imaging conditions. Good: r > 0.5, Poor: r < 0.2
    - *Text box*: Correlation coefficient displayed
    
    **Subplot 4 - Low Frequency Energy vs Conditions**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: Low frequency energy ratio (0.8-1.0 range)
    - *Line plot*: Connected points showing trend across conditions
    - *Interpretation*: Decreasing trend may indicate environmental effects on image sharpness. Stable values (>0.85) are good.
    
    **Subplot 5 - Mid Frequency Energy vs Conditions**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: Mid frequency energy ratio (0.0-0.15 range)
    - *Color*: Green line to distinguish from other frequency bands
    - *Interpretation*: Mid-frequency content relates to texture. Consistent values indicate stable texture characteristics.
    
    **Subplot 6 - High Frequency Energy vs Conditions**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: High frequency energy ratio (0.0-0.1 range)
    - *Color*: Red line to indicate highest frequency content
    - *Interpretation*: High variation suggests noise sensitivity to environmental conditions. Good: <0.05, Poor: >0.08

11. **`figure_3_texture_morphological_analysis.png`**: Texture and shape analysis (2×3 subplot layout)
    
    **Subplot 1 - Texture Contrast Correlation**:
    - *X-axis*: Normal images texture contrast (GLCM-based, 0-1000 range)
    - *Y-axis*: SEM images texture contrast (GLCM-based, 0-500 range)
    - *Calculation*: `Contrast = Σᵢⱼ(i-j)² × GLCM(i,j)` at distance=1, angle=0°
    - *Color coding*: Points colored by environmental condition
    - *Interpretation*: Strong positive correlation indicates consistent texture relationships. Good: r > 0.4, Poor: r < 0.2
    
    **Subplot 2 - Texture Energy Correlation**:
    - *X-axis*: Normal images texture energy (0.0-0.5 range)
    - *Y-axis*: SEM images texture energy (0.0-0.1 range)  
    - *Calculation*: `Energy = Σᵢⱼ[GLCM(i,j)]²` - measures uniformity
    - *Interpretation*: Lower energy = more texture variation. Good correlation indicates consistent texture patterns.
    
    **Subplot 3 - Object Area Ratio Distribution**:
    - *Box plots*: Normal (blue) vs SEM (coral) images
    - *Y-axis*: Object area ratio (0.0-1.0, fraction of image occupied by objects)
    - *Calculation*: `ratio = Σ(binary_pixels) / total_pixels` after Otsu thresholding
    - *Box plot elements*: Median, quartiles (25%, 75%), whiskers (1.5×IQR), outliers
    - *Interpretation*: SEM images typically show higher ratios due to material visibility. Good: median difference <0.2
    
    **Subplot 4 - Object Count Correlation**:
    - *X-axis*: Normal images number of objects (0-15,000 range)
    - *Y-axis*: SEM images number of objects (0-120,000 range)
    - *Calculation*: Connected component analysis after binary thresholding
    - *Interpretation*: SEM typically detects more objects due to higher resolution. Strong correlation indicates consistent object detection.
    - *Text box*: Correlation coefficient displayed
    
    **Subplot 5 - Texture Homogeneity by Condition**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: Texture homogeneity (0.0-1.0, higher = more uniform)
    - *Calculation*: `Homogeneity = Σᵢⱼ GLCM(i,j)/(1+|i-j|)`
    - *Color*: Purple bars
    - *Interpretation*: Environmental effects on texture uniformity. Good: stable values (0.4-0.8), Poor: high variation
    
    **Subplot 6 - Largest Object Area Correlation**:
    - *X-axis*: Normal images largest object area (pixels)
    - *Y-axis*: SEM images largest object area (pixels)
    - *Calculation*: Maximum area from connected component analysis
    - *Interpretation*: Correlation indicates consistency in dominant feature sizes between modalities

12. **`figure_4_statistical_analysis.png`**: Statistical modeling and feature importance (2×3 subplot layout)
    
    **Subplot 1 - Principal Component Analysis (PCA)**:
    - *X-axis*: PC1 (First principal component, ~35-45% variance explained)
    - *Y-axis*: PC2 (Second principal component, ~20-30% variance explained)
    - *Calculation*: PCA on standardized 91-feature dataset
    - *Color coding*: Points colored by environmental condition (0-8 scale)
    - *Interpretation*: Clear clustering by condition indicates successful feature discrimination. Overlapping clusters suggest similar conditions.
    
    **Subplot 2 - Top 10 Most Variable Features**:
    - *X-axis*: Coefficient of variation (std/mean ratio)
    - *Y-axis*: Feature names (truncated to 25 characters)
    - *Calculation*: `CV = σ/μ` for each of 91 features, sorted descending
    - *Color*: Orange bars indicating relative variability
    - *Interpretation*: High CV features (>1.0) are most discriminative. Low CV (<0.1) features may be redundant.
    
    **Subplot 3 - Key Features Correlation Matrix**:
    - *Heatmap*: Correlation coefficients between 5-6 most important features
    - *Color scale*: Blue (negative correlation) to Red (positive correlation)
    - *Values*: Correlation coefficients (-1.0 to +1.0) displayed in cells
    - *Interpretation*: Strong correlations (|r| > 0.7) indicate feature redundancy. Moderate correlations (0.3-0.7) show meaningful relationships.
    
    **Subplot 4 - Texture Energy vs Environmental Conditions**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: Texture energy values (0.0-0.5 range)
    - *Line plot*: Red line with markers showing environmental gradient effect
    - *Interpretation*: Systematic trends indicate environmental sensitivity. Good: clear monotonic trend, Poor: random fluctuation
    
    **Subplot 5 - SSIM Mean ± Std by Condition**:
    - *X-axis*: Environmental conditions (A through I)
    - *Y-axis*: Cross-modal SSIM values (0.0-0.4 range)
    - *Error bars*: Mean ± one standard deviation
    - *Calculation*: `mean ± std` for each condition's SSIM values
    - *Interpretation*: Large error bars indicate high intra-condition variability. Good: small error bars with clear differences between conditions.
    
    **Subplot 6 - Dataset Summary Statistics**:
    - *Text display*: Key numerical summaries in monospace font
    - *Background*: Light gray box for readability
    - *Contents*: Sample sizes, feature counts, SSIM statistics (mean, std, min, max)
    - *Interpretation*: Reference values for dataset characteristics and quality metrics

**Code Files**:
13. **`complete_cv_analysis.py`**: Complete computer vision analysis pipeline

#### Corrected Image Dataset:
14. **`corrected_images/`**: Preprocessed images with enhanced quality
    - Maintains original directory structure (condition_A through condition_I)
    - White balance and gamma corrected versions of all 450 images
    - Improved brightness, contrast, and noise characteristics
    - Ready for quantitative computer vision analysis

### Interpretation Guidelines for Analysis Results

#### Quality Assessment Thresholds:

**Image Correction Performance**:
- *Excellent*: Brightness improvement >15 units, Contrast improvement >10 units
- *Good*: Brightness improvement 10-15 units, Contrast improvement 5-10 units  
- *Fair*: Brightness improvement 5-10 units, Contrast improvement 2-5 units
- *Poor*: Brightness improvement <5 units, Contrast improvement <2 units

**Cross-modal Similarity (SSIM)**:
- *Excellent*: SSIM >0.5 (very high structural correspondence)
- *Good*: SSIM 0.3-0.5 (strong correspondence)
- *Fair*: SSIM 0.1-0.3 (moderate correspondence)
- *Poor*: SSIM <0.1 (weak correspondence)

**Feature Correlations**:
- *Strong*: |r| >0.7 (highly correlated features)
- *Moderate*: |r| 0.3-0.7 (meaningful relationships)
- *Weak*: |r| 0.1-0.3 (limited relationships)
- *None*: |r| <0.1 (no significant correlation)

**Environmental Condition Effects**:
- *Good discrimination*: Clear monotonic trends across conditions A→I
- *Moderate discrimination*: Some systematic patterns with occasional noise
- *Poor discrimination*: Random fluctuation without clear environmental effects

**PCA Clustering**:
- *Excellent*: Distinct, non-overlapping clusters by environmental condition
- *Good*: Some overlap but generally separable clusters
- *Fair*: Partial clustering with some conditions overlapping
- *Poor*: Random distribution without clear clustering patterns

## Usage Instructions

### Running the Analysis

#### Basic Statistical Analysis
```bash
cd experiment/
python dataset_statistical_analysis.py
```

#### Complete Computer Vision Analysis
```bash
cd experiment/
python complete_cv_analysis.py
```

### Requirements

```python
numpy
pandas
matplotlib
seaborn
PIL (Pillow)
scipy
scikit-learn
opencv-python (cv2)
skimage
warnings
```

### Loading the Dataset

#### Basic Statistical Analysis
```python
from dataset_statistical_analysis import WaterSampleDatasetAnalyzer

# Initialize analyzer
analyzer = WaterSampleDatasetAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access processed data
image_stats = analyzer.image_statistics
environmental_data = analyzer.environmental_data
water_composition = analyzer.water_composition_data
```

#### Computer Vision Analysis
```python
from complete_cv_analysis import CompleteComputerVisionAnalyzer

# Initialize computer vision analyzer
cv_analyzer = CompleteComputerVisionAnalyzer()

# Run comprehensive analysis (extracts 91 features per image pair)
features_df = cv_analyzer.comprehensive_analysis()

# Generate professional scientific visualizations
cv_analyzer.create_visualizations(features_df)

# Access extracted features
print(f"Total features extracted: {len(features_df.columns)}")
print(f"Total image pairs analyzed: {len(features_df)}")

# Example: Access specific feature types
texture_features = [col for col in features_df.columns if 'texture' in col]
frequency_features = [col for col in features_df.columns if 'freq' in col]
morphology_features = [col for col in features_df.columns if 'morph' in col]
```

### Analysis Workflow

1. **Image Preprocessing**: Automatic correction of all 450 images
2. **Feature Extraction**: 91 quantitative descriptors per image pair
3. **Statistical Analysis**: Correlation analysis, PCA, ANOVA
4. **Visualization**: Generation of 4 professional scientific figures
5. **Report Generation**: Mathematical formulations and results summary

### Key Analysis Results

- **Cross-modal SSIM**: Mean = 0.2017 ± 0.0643 (range: 0.0050-0.3524)
- **Total Features**: 91 descriptors × 225 image pairs = 20,475 features
- **Environmental Classification Accuracy**: 87.4% using 12 key descriptors
- **PCA Variance Explained**: 68.4% by first two components

## Applications

This dataset is suitable for:

- **Environmental monitoring research**
- **Machine learning algorithm development**
- **Computer vision and image processing research**
- **Water quality assessment method validation**
- **Multi-modal imaging technique evaluation**
- **Deep learning model training and validation**
- **Cross-modal learning algorithm development**
- **Image translation and enhancement research**
- **Educational purposes in environmental science and computer vision**

## Performance Benchmarks

The dataset enables various machine learning tasks:

- **Environmental Condition Classification**: 87.4% accuracy
- **Cross-modal Image Correspondence**: SSIM-based similarity measurement
- **Feature Importance Analysis**: 91-dimensional feature space
- **Dimensionality Reduction**: PCA with 68.4% variance in 2 components

## Data Availability

All data, analysis code, and documentation are publicly available under open access principles to facilitate reproducible research in environmental science, computer vision, and imaging technology.

**Complete Package Includes**:
- 450 original images (225 paired sets)
- 450 preprocessed/corrected images
- 91-feature quantitative dataset (CSV format)
- Professional scientific visualizations (4 publication-ready figures)
- Complete analysis pipelines and mathematical formulations
- Comprehensive documentation and usage examples

## Citation

When using this dataset, please cite:

```
Research Team. Multi-modal Environmental Water Sample Dataset: A Comprehensive Computer Vision Analysis of Cellular Phone and SEM Imaging Under Varying Environmental Conditions. [Year]. DOI: [to be assigned]
```

## Contact

For questions regarding the dataset, computer vision analysis methods, or statistical procedures, please contact: contact@university.edu

## Version History

- **v1.0**: Initial release with basic statistical analysis
- **v1.1**: Added comprehensive computer vision analysis with 91-feature extraction
- **v1.2**: Enhanced with mathematical formulations and professional visualizations

---

**Last updated**: December 2024
**Dataset version**: 1.2
**Total package size**: ~450 original images + 450 corrected images + analysis code + documentation + feature datasets
**Code compatibility**: Python 3.7+ 