#!/usr/bin/env python3
"""
Shared utilities for particle-level analysis of coffee ring SEM images.
Provides deposit segmentation, particle detection, property extraction,
and dataset iteration helpers.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure, morphology, filters, segmentation, feature
from scipy import ndimage
from scipy.ndimage import uniform_filter

# ── Paths ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Volumes/XiaoyanSSD/Data/chapter_3_data/data")
SEM_DIR = DATA_ROOT / "sem"
EDS_DIR = DATA_ROOT / "eds"
CORRECTED_DIR = DATA_ROOT / "corrected_images"
EXPERIMENT_DIR = Path("/Volumes/XiaoyanSSD/Data/chapter_3_data/experiment")

CONDITIONS = list("ABCDEFGHI")
IMAGES_PER_CONDITION = 25

CONDITION_INFO = {
    'A': {'temp': '20-23', 'rh': '35-40'},
    'B': {'temp': '20-23', 'rh': '40-45'},
    'C': {'temp': '20-23', 'rh': '45-50'},
    'D': {'temp': '23-26', 'rh': '35-40'},
    'E': {'temp': '23-26', 'rh': '40-45'},
    'F': {'temp': '23-26', 'rh': '45-50'},
    'G': {'temp': '26-29', 'rh': '35-40'},
    'H': {'temp': '26-29', 'rh': '40-45'},
    'I': {'temp': '26-29', 'rh': '45-50'},
}

EDS_ELEMENTS = ['C Kα1_2', 'Ca Kα1', 'Cl Kα1', 'Mg Kα1_2', 'Na Kα1_2', 'O Kα1', 'S Kα1']
EDS_ELEMENT_NAMES = ['C', 'Ca', 'Cl', 'Mg', 'Na', 'O', 'S']

# ── Pixel scale ────────────────────────────────────────────────────────
# JEOL 6610LV at 50x: 500 μm micron marker.
# SEM image is 2560×1920 with information bar at bottom (~140 px).
# Effective imaging area ≈ 2560×1780. The 500 μm bar spans ~250 px.
# → pixel size ≈ 2.0 μm/px. We use this as default; can be refined.
PIXEL_SCALE_UM = 2.0  # micrometers per pixel


def get_sample_id(image_num):
    """Map image number (1-25) to water sample ID (1-5)."""
    return (image_num - 1) // 5 + 1


def load_sem_gray(condition, image_num, corrected=True):
    """Load SEM image as grayscale."""
    if corrected:
        path = CORRECTED_DIR / condition / f"SEM_{image_num}.jpg"
    else:
        path = SEM_DIR / condition / f"SEM_{image_num}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def load_eds_map(condition, image_num, element_file):
    """Load a single EDS elemental map."""
    path = EDS_DIR / condition / str(image_num) / f"{element_file}.png"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


# ── Deposit segmentation ──────────────────────────────────────────────

def get_deposit_mask(sem_gray, crop_bottom=140):
    """
    Segment the deposit region from the aluminum substrate using
    local texture (standard deviation) to distinguish the textured
    deposit from the smooth polished substrate.

    Parameters
    ----------
    sem_gray : ndarray
        Grayscale SEM image (2560×1920).
    crop_bottom : int
        Pixels to ignore at the bottom (SEM info bar).

    Returns
    -------
    mask : ndarray (bool)
        Binary mask of the deposit region (same size as input).
    contour : ndarray
        Contour points of the deposit boundary.
    center : tuple (cy, cx)
        Centroid of the deposit.
    equiv_radius : float
        Equivalent circular radius in pixels.
    """
    h, w = sem_gray.shape
    work = sem_gray[:h - crop_bottom, :]

    # Downsample 4× for speed
    small = cv2.resize(work, (w // 4, (h - crop_bottom) // 4))

    # Local standard deviation map (texture metric)
    mean_val = uniform_filter(small.astype(float), size=21)
    mean_sq = uniform_filter(small.astype(float) ** 2, size=21)
    local_std = np.sqrt(np.clip(mean_sq - mean_val ** 2, 0, None))

    # Otsu on the texture map
    ret, _ = cv2.threshold(local_std.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tex_binary = (local_std > ret).astype(np.uint8)

    # Morphological close + fill to form a solid deposit region
    closed = morphology.binary_closing(tex_binary > 0, morphology.disk(12))
    filled = ndimage.binary_fill_holes(closed)

    # Keep only the largest connected component
    labeled = measure.label(filled)
    props = measure.regionprops(labeled)
    if not props:
        mask = np.zeros((h, w), dtype=bool)
        return mask, np.array([]), (h // 2, w // 2), 0.0

    largest = max(props, key=lambda r: r.area)
    mask_small = (labeled == largest.label)
    frac = mask_small.sum() / mask_small.size

    # If deposit covers > 65% of image, raise threshold iteratively
    if frac > 0.65:
        for mult in [1.3, 1.6, 2.0, 2.5]:
            t = int(ret * mult)
            tb = (local_std > t).astype(np.uint8)
            cl = morphology.binary_closing(tb > 0, morphology.disk(10))
            fl = ndimage.binary_fill_holes(cl)
            lb = measure.label(fl)
            pp = measure.regionprops(lb)
            if pp:
                lg = max(pp, key=lambda r: r.area)
                ms = (lb == lg.label)
                f2 = ms.sum() / ms.size
                if 0.05 < f2 < 0.65:
                    mask_small = ms
                    frac = f2
                    break

    # Upsample mask to original resolution
    mask_full = cv2.resize(mask_small.astype(np.uint8),
                           (w, h - crop_bottom),
                           interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((h, w), dtype=bool)
    mask[:h - crop_bottom, :] = mask_full > 0

    # Extract contour
    contours_list = measure.find_contours(mask.astype(float), 0.5)
    contour = max(contours_list, key=len) if contours_list else np.array([])

    # Centroid and equivalent radius
    rp = measure.regionprops(mask.astype(np.uint8))
    if rp:
        cy, cx = rp[0].centroid
        equiv_radius = np.sqrt(rp[0].area / np.pi)
    else:
        cy, cx = h // 2, w // 2
        equiv_radius = 0.0

    return mask, contour, (cy, cx), equiv_radius


# ── Particle detection ────────────────────────────────────────────────

def detect_particles(sem_gray, deposit_mask, min_area=30, max_area=50000):
    """
    Detect individual particles/crystals within the deposit region.

    Uses adaptive thresholding + watershed to separate touching particles.

    Parameters
    ----------
    sem_gray : ndarray
        Grayscale SEM image.
    deposit_mask : ndarray (bool)
        Binary mask of deposit region.
    min_area : int
        Minimum particle area in pixels (filter noise).
    max_area : int
        Maximum particle area in pixels (filter deposit blobs).

    Returns
    -------
    labeled : ndarray
        Labeled image (each particle has unique integer label).
    """
    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(sem_gray)

    # Adaptive thresholding — particles are bright against deposit film
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=51, C=-8
    )

    # Restrict to deposit region
    binary = binary & deposit_mask.astype(np.uint8) * 255

    # Morphological opening to clean noise
    kernel_open = morphology.disk(2)
    binary_clean = morphology.binary_opening(binary > 0, kernel_open)

    # Watershed segmentation to separate touching particles
    distance = ndimage.distance_transform_edt(binary_clean)
    # Find local maxima as seeds
    from skimage.feature import peak_local_max
    coords = peak_local_max(distance, min_distance=8, labels=binary_clean.astype(int))
    seeds = np.zeros(distance.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        seeds[r, c] = i
    seeds = ndimage.label(morphology.binary_dilation(seeds > 0, morphology.disk(2)))[0]
    labeled = segmentation.watershed(-distance, seeds, mask=binary_clean)

    # Filter by area
    props = measure.regionprops(labeled)
    keep = np.zeros_like(labeled)
    for p in props:
        if min_area <= p.area <= max_area:
            keep[labeled == p.label] = p.label
    # Relabel contiguously
    labeled = measure.label(keep > 0)

    return labeled


# ── Property extraction ───────────────────────────────────────────────

def extract_particle_properties(labeled, scale_um=PIXEL_SCALE_UM):
    """
    Extract morphological properties of each detected particle.

    Returns DataFrame with one row per particle, measurements in μm.
    """
    props = measure.regionprops(labeled)
    records = []
    for p in props:
        area_px = p.area
        perim_px = p.perimeter
        circularity = (4 * np.pi * area_px / perim_px ** 2) if perim_px > 0 else 0
        major = p.major_axis_length
        minor = p.minor_axis_length
        aspect_ratio = (major / minor) if minor > 0 else 1.0

        records.append({
            'label': p.label,
            'area_um2': area_px * scale_um ** 2,
            'perimeter_um': perim_px * scale_um,
            'equiv_diameter_um': p.equivalent_diameter * scale_um,
            'feret_diameter_um': p.feret_diameter_max * scale_um if hasattr(p, 'feret_diameter_max') else major * scale_um,
            'major_axis_um': major * scale_um,
            'minor_axis_um': minor * scale_um,
            'circularity': np.clip(circularity, 0, 1),
            'solidity': p.solidity,
            'aspect_ratio': aspect_ratio,
            'eccentricity': p.eccentricity,
            'orientation': np.degrees(p.orientation),
            'centroid_y': p.centroid[0],
            'centroid_x': p.centroid[1],
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Dataset iteration ─────────────────────────────────────────────────

def iterate_dataset(conditions=None, corrected=True):
    """
    Yield (condition, image_num, sample_id, sem_gray) for each image.
    """
    if conditions is None:
        conditions = CONDITIONS
    for cond in conditions:
        for img_num in range(1, IMAGES_PER_CONDITION + 1):
            sample_id = get_sample_id(img_num)
            try:
                sem = load_sem_gray(cond, img_num, corrected=corrected)
                yield cond, img_num, sample_id, sem
            except FileNotFoundError:
                print(f"  [SKIP] {cond}/SEM_{img_num}.jpg not found")
                continue


def save_figure(fig, name, output_dir=None):
    """Save figure as both PNG and SVG."""
    if output_dir is None:
        output_dir = EXPERIMENT_DIR
    output_dir = Path(output_dir)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f"{name}.svg", bbox_inches='tight')
    print(f"  Saved: {name}.png / .svg")
