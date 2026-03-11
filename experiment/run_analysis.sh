#!/bin/bash
# Water Sample Analysis Pipeline
# This script runs all analysis steps in the correct order

echo "=========================================="
echo "Water Sample Analysis Pipeline"
echo "=========================================="
echo ""

# Step 1: Basic Statistical Analysis
echo "Step 1: Running basic statistical analysis..."
python dataset_statistical_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in dataset_statistical_analysis.py"
    exit 1
fi
echo "✓ Basic statistical analysis completed"
echo ""

# Step 2: Computer Vision Feature Extraction
echo "Step 2: Extracting computer vision features..."
python improved_cv_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in improved_cv_analysis.py"
    exit 1
fi
echo "✓ Computer vision features extracted"
echo ""

# Step 3: Advanced Visualizations
echo "Step 3: Creating advanced visualizations..."
python advanced_visualizations.py
if [ $? -ne 0 ]; then
    echo "Error in advanced_visualizations.py"
    exit 1
fi
echo "✓ Advanced visualizations created"
echo ""

# Step 4: Generate Heatmaps
echo "Step 4: Generating heatmaps..."
python generate_heatmaps.py
if [ $? -ne 0 ]; then
    echo "Error in generate_heatmaps.py"
    exit 1
fi
echo "✓ Heatmaps generated"
echo ""

echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "- Statistical reports: comprehensive_analysis_report.txt"
echo "- Feature data: comprehensive_computer_vision_features.csv"
echo "- Visualizations: *.svg and *.png files"
echo ""
echo "See WORKFLOW_GUIDE.md for detailed information about each step."