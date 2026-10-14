#!/usr/bin/env python3
"""EDS signal versus known elemental composition, for all seven mapped elements.

Signal coverage is the fraction of pixels brighter than 5 (same definition as
eds_validation_analysis.py). For each element the expected concentration (mM of
that element) follows from the recipe; coverage is compared with it
(a) on the five water-sample means (Pearson, n = 5) and
(b) on all 225 maps (Spearman, n = 225; maps from one condition are not
    independent, so this p-value is indicative only).
Output: eds_composition_validation.csv
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EDS = ROOT / "data" / "eds"
COMP = pd.read_csv(ROOT / "data" / "metadata" / "sample_compositions.csv").set_index("sample_id")
FILES = {"C": "C Kα1_2.png", "Ca": "Ca Kα1.png", "Cl": "Cl Kα1.png", "Mg": "Mg Kα1_2.png",
         "Na": "Na Kα1_2.png", "O": "O Kα1.png", "S": "S Kα1.png"}


def element_mM(c):
    """Concentration of each element (mM) in a recipe row."""
    return {
        "C": c.NaHCO3_mM,
        "Ca": c.CaCl2_mM,
        "Mg": c.MgCl2_mM,
        "Cl": 2 * c.CaCl2_mM + 2 * c.MgCl2_mM,
        "Na": c.NaHCO3_mM + 2 * c.Na2SO4_mM + c.NaH2PO4_mM,
        "S": c.Na2SO4_mM + c.CuSO4_mM,
        "O": 3 * c.NaHCO3_mM + 4 * c.Na2SO4_mM + 4 * c.NaH2PO4_mM + 9 * c.Fe_NO3_3_mM + 4 * c.CuSO4_mM,
    }


def coverage(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgb = img[:, :, :3] if img.ndim == 3 else img
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
    return float((gray > 5).mean())


rows = []
for cond in "ABCDEFGHI":
    for i in range(1, 26):
        s = (i - 1) // 5 + 1
        conc = element_mM(COMP.loc[s])
        for el, fn in FILES.items():
            rows.append(dict(condition=cond, image_id=i, sample=s, element=el,
                             coverage=coverage(EDS / cond / str(i) / fn), conc_mM=conc[el]))
df = pd.DataFrame(rows)

out = []
for el, g in df.groupby("element"):
    m = g.groupby("sample").agg(cov=("coverage", "mean"), conc=("conc_mM", "first"))
    if m.conc.nunique() > 1:
        r, p = stats.pearsonr(m.conc, m["cov"])
        rho, prho = stats.spearmanr(g.conc_mM, g.coverage)
    else:
        r = p = rho = prho = np.nan
    out.append(dict(element=el, conc_range_mM=f"{m.conc.min():.3g}-{m.conc.max():.3g}",
                    distinct_levels=m.conc.nunique(), pearson_r_sample_means=r, p_sample_means=p,
                    spearman_rho_all_maps=rho, p_all_maps=prho,
                    coverage_by_sample=";".join(f"{v:.4f}" for v in m["cov"])))
res = pd.DataFrame(out)
res.to_csv(Path(__file__).with_name("eds_composition_validation.csv"), index=False)
print(res.round(4).to_string(index=False))
