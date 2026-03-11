#!/usr/bin/env python3
"""
Professional plotting style configuration for research publications
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_professional_style():
    """
    Set up professional matplotlib style for research publications
    """
    # Use a clean style as base
    plt.style.use('seaborn-v0_8-paper')
    
    # Update parameters for professional publication quality
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        
        # Save settings - SVG format
        'savefig.dpi': 300,
        'savefig.format': 'svg',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        
        # Font settings - Professional serif font
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.prop_cycle': plt.cycler('color', [
            '#1f77b4',  # Muted blue
            '#ff7f0e',  # Safety orange
            '#2ca02c',  # Cooked asparagus green
            '#d62728',  # Brick red
            '#9467bd',  # Muted purple
            '#8c564b',  # Chestnut brown
            '#e377c2',  # Raspberry yogurt pink
            '#7f7f7f',  # Middle gray
            '#bcbd22',  # Curry yellow-green
            '#17becf'   # Blue-teal
        ]),
        
        # Grid settings
        'grid.linewidth': 0.6,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        
        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend settings
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': False,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'gray',
        'legend.borderpad': 0.5,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.8,
        'lines.markeredgecolor': 'auto',
        
        # Patch settings
        'patch.linewidth': 1.0,
        'patch.edgecolor': 'black',
        
        # Histogram settings
        'hist.bins': 'auto',
        
        # Error bar settings
        'errorbar.capsize': 3,
        
        # Scatter plot settings
        'scatter.marker': 'o',
        
        # Image settings
        'image.cmap': 'viridis',
        'image.interpolation': 'antialiased',
        
        # Contour settings
        'contour.negative_linestyle': 'dashed',
        
        # Box plot settings
        'boxplot.meanline': True,
        'boxplot.showmeans': True,
        'boxplot.showcaps': True,
        'boxplot.showbox': True,
        'boxplot.showfliers': True,
        'boxplot.notch': False,
        'boxplot.vertical': True,
        'boxplot.whiskers': 1.5,
        'boxplot.bootstrap': None,
        'boxplot.capprops.linewidth': 1.0,
        'boxplot.boxprops.linewidth': 1.0,
        'boxplot.whiskerprops.linewidth': 1.0,
        'boxplot.flierprops.markersize': 5,
        'boxplot.flierprops.linewidth': 1.0,
        'boxplot.meanprops.markersize': 6,
        'boxplot.meanprops.linewidth': 1.0,
        'boxplot.medianprops.linewidth': 1.5,
    })
    
    # Set color palette for seaborn
    sns.set_palette("deep")
    
def get_professional_colors():
    """
    Get a set of professional colors for plotting
    """
    return {
        'primary': '#1f77b4',      # Professional blue
        'secondary': '#ff7f0e',    # Orange
        'tertiary': '#2ca02c',     # Green
        'quaternary': '#d62728',   # Red
        'quinary': '#9467bd',      # Purple
        'senary': '#8c564b',       # Brown
        'neutral': '#7f7f7f',      # Gray
        'highlight': '#17becf',    # Cyan
        'background': '#f0f0f0',   # Light gray
        'grid': '#cccccc',         # Medium gray
        'text': '#333333',         # Dark gray
    }

def get_colormap_options():
    """
    Get recommended colormaps for different data types
    """
    return {
        'sequential': ['viridis', 'plasma', 'cividis', 'Blues', 'Greens'],
        'diverging': ['RdBu_r', 'coolwarm', 'seismic_r', 'PuOr_r'],
        'categorical': ['Set2', 'Set3', 'tab10', 'Paired'],
        'continuous': ['viridis', 'plasma', 'magma', 'inferno']
    }

def style_axis(ax):
    """
    Apply professional styling to a single axis
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines thinner
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    
    # Adjust tick parameters
    ax.tick_params(width=0.8, length=5)
    
def add_significance_bar(ax, x1, x2, y, p_value, height=0.02):
    """
    Add significance bars to plots
    """
    # Calculate bar position
    y_max = y + height * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y_max, y_max, y], 'k-', linewidth=1)
    
    # Add significance stars
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    ax.text((x1 + x2) / 2, y_max, sig_text, ha='center', va='bottom', fontsize=10)
    
def format_pvalue(p):
    """
    Format p-values for display
    """
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    elif p < 0.05:
        return f'p = {p:.2f}'
    else:
        return f'p = {p:.2f}'