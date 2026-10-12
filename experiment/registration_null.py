#!/usr/bin/env python3
"""Null distribution for registered crystal-map correlation.

For each pair, the optical image of a different droplet from the same
condition is warped with the pair's own transform and correlated with the
SEM crystal map. A matched pair should correlate far more than this.
"""
import numpy as np
import pandas as pd
import cv2
from cross_modal_registration import DATA, OUT, WORK_W, gray, crystals, droplet_mask, ncc

reg = pd.read_csv(OUT / "cross_modal_registration.csv")
rng = np.random.default_rng(0)
cache = {}


def work(kind, cond, i):
    key = (kind, cond, i)
    if key not in cache:
        name = f"Image_{i}.jpg" if kind == "opt" else f"SEM_{i}.jpg"
        g = gray(DATA / "corrected_images" / cond / name)
        f = WORK_W / g.shape[1]
        cache[key] = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cache[key]


rows = []
for _, r in reg.iterrows():
    cond, i = r.condition, int(r.image_id)
    j = int(rng.choice([k for k in range(1, 26) if k != i]))
    M = np.array([[r.M00, r.M01, r.M02], [r.M10, r.M11, r.M12]])
    sem = work("sem", cond, i)
    h, w = sem.shape
    inner = cv2.erode(droplet_mask(sem).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    g_s = crystals(sem)
    null = ncc(cv2.warpAffine(crystals(work("opt", cond, j)), M, (w, h)), g_s, inner)
    rows.append(dict(condition=cond, image_id=i, shuffled_with=j, matched_ncc=r.final_ncc, shuffled_ncc=null))
df = pd.DataFrame(rows)
df.to_csv(OUT / "registration_null.csv", index=False)
print(df[["matched_ncc", "shuffled_ncc"]].describe().round(3).to_string())
from scipy.stats import wilcoxon
print("Wilcoxon matched > shuffled:", wilcoxon(df.matched_ncc, df.shuffled_ncc, alternative="greater"))
print("matched > shuffled in", (df.matched_ncc > df.shuffled_ncc).sum(), "of", len(df), "pairs")
