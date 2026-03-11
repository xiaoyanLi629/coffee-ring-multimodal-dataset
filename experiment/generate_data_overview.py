#!/usr/bin/env python3
"""Generate multi-modal data overview figure for manuscript.
Simple version: original images without cropping/rotation adjustments."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'figure.dpi': 300,
})

data_root = Path('/Volumes/XiaoyanSSD/Data/chapter_3_data/data')
cond = 'B'
samples = [2, 3, 4, 5, 6]

fig, axes = plt.subplots(3, 5, figsize=(16, 10),
                         gridspec_kw={'hspace': 0.03, 'wspace': 0.03,
                                      'height_ratios': [1, 1, 0.79]})

labels_col = ['Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6']
labels_row = ['Mobile Phone', 'SEM', 'EDS (Cl)']

for col, sid in enumerate(samples):
    # Mobile phone
    img = mpimg.imread(str(data_root / 'optical' / cond / f'Image_{sid}.jpg'))
    axes[0, col].imshow(img)
    axes[0, col].set_title(labels_col[col], fontsize=14, fontweight='bold', pad=5)

    # SEM
    img = mpimg.imread(str(data_root / 'sem' / cond / f'SEM_{sid}.jpg'))
    axes[1, col].imshow(img, cmap='gray')

    # EDS (Cl)
    img = mpimg.imread(str(data_root / 'eds' / cond / str(sid) / 'Cl Kα1.png'))
    axes[2, col].imshow(img)

# Row labels
for r, label in enumerate(labels_row):
    bbox = axes[r, 0].get_position()
    y_center = (bbox.y0 + bbox.y1) / 2
    fig.text(0.01, y_center, label, fontsize=14, fontweight='bold',
             rotation=90, va='center', ha='left')

# Clean axes
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('')
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color('#aaaaaa')

plt.subplots_adjust(left=0.07, right=0.998, top=0.96, bottom=0.005)

out_dir = Path('/Volumes/XiaoyanSSD/Data/chapter_3_data/manuscript/figures')
fig.savefig(out_dir / 'data_overview.pdf', format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(out_dir / 'data_overview.png', format='png', dpi=300, bbox_inches='tight')
print('Saved: data_overview.pdf / .png')
plt.close()
