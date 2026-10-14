#!/usr/bin/env python3
"""Per-image quality metrics for the original optical and SEM images.

sharpness   variance of the Laplacian of the grey image (resized to 1280 px wide)
saturated   fraction of pixels >= 250 (over-exposure / charging)
dark        fraction of pixels <= 5 (under-exposure)
mean, std   grey-level mean and standard deviation (0-255)
An image is flagged when log10(sharpness) lies more than 3.5 robust SD
(median absolute deviation x 1.4826) below the median of its modality, or when
more than 5 % of its pixels are saturated.
Output: image_quality_metrics.csv, image_quality_summary.csv
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
rows = []
for mod, pat in [("optical", "Image_{}.jpg"), ("sem", "SEM_{}.jpg")]:
    for cond in "ABCDEFGHI":
        for i in range(1, 26):
            g = cv2.imread(str(DATA / mod / cond / pat.format(i)), cv2.IMREAD_GRAYSCALE)
            g1280 = cv2.resize(g, (1280, round(g.shape[0] * 1280 / g.shape[1])), interpolation=cv2.INTER_AREA)
            rows.append(dict(modality=mod, condition=cond, image_id=i,
                             sharpness=float(cv2.Laplacian(g1280, cv2.CV_64F).var()),
                             saturated=float((g >= 250).mean()), dark=float((g <= 5).mean()),
                             mean=float(g.mean()), std=float(g.std())))
df = pd.DataFrame(rows)
for mod, g in df.groupby("modality"):
    ls = np.log10(g.sharpness)
    rz = (ls - ls.median()) / (1.4826 * (ls - ls.median()).abs().median())
    df.loc[g.index, "sharpness_robust_z"] = rz
df["flag_blur"] = df.sharpness_robust_z < -3.5
df["flag_saturation"] = df.saturated > 0.05
here = Path(__file__).parent
df.to_csv(here / "image_quality_metrics.csv", index=False)
summ = df.groupby("modality").agg(
    n=("image_id", "size"),
    sharpness_median=("sharpness", "median"), sharpness_p5=("sharpness", lambda v: v.quantile(0.05)),
    sharpness_p95=("sharpness", lambda v: v.quantile(0.95)),
    saturated_median_pct=("saturated", lambda v: 100 * v.median()), saturated_max_pct=("saturated", lambda v: 100 * v.max()),
    dark_median_pct=("dark", lambda v: 100 * v.median()), dark_max_pct=("dark", lambda v: 100 * v.max()),
    mean_median=("mean", "median"), std_median=("std", "median"),
    flagged_blur=("flag_blur", "sum"), flagged_saturation=("flag_saturation", "sum"))
summ.to_csv(here / "image_quality_summary.csv")
print(summ.round(3).T.to_string())
print(df[df.flag_blur | df.flag_saturation][["modality", "condition", "image_id", "sharpness", "sharpness_robust_z", "saturated", "dark", "mean"]].round(4).to_string(index=False))
print(df.groupby(["modality", "condition"])["mean"].median().unstack(0).round(1).T.to_string())
