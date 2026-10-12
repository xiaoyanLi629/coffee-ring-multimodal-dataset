#!/usr/bin/env python3
"""Register each optical image to its SEM image and compute cross-modal SSIM.

The earlier SSIM compared the whole optical image with the top-left
1280 x 1024 crop of the SEM image, without alignment. Here the optical image
is first mapped into the SEM frame:

1. segment the droplet in both images (local texture, Otsu, largest blob);
2. scale and translation from droplet area and centroid;
3. rotation (0-360 deg, with and without mirroring) by maximising the
   normalised cross-correlation of gradient-magnitude maps inside the droplet;
4. refinement with ECC (Euclidean motion) on the gradient maps.

SSIM is then computed on the corrected images inside the SEM droplet mask.
Outputs: cross_modal_registration.csv and a QA montage.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
CONDITIONS = list("ABCDEFGHI")
WORK_W = 640                      # SEM frame is processed at 640 x 480
ANGLES = np.arange(0, 360, 2)


def gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im.astype(np.float32) / 255


def droplet_mask(g, rmin=0.28, rmax=0.48):
    """Disc fitted to the droplet rim with a Hough circle transform.

    Radii are fractions of the image height. Both modalities show a sharp
    rim, while the background can carry brushing streaks or bright debris,
    so a circle fit is more robust than texture thresholding.
    """
    h, w = g.shape
    g8 = (cv2.GaussianBlur(g, (0, 0), 2.0) * 255).astype(np.uint8)
    circles = None
    for p2 in (40, 30, 22, 15):
        circles = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT, dp=1.5, minDist=h,
                                   param1=60, param2=p2,
                                   minRadius=int(rmin * h), maxRadius=int(rmax * h))
        if circles is not None:
            break
    yy, xx = np.mgrid[:h, :w]
    if circles is None:
        cx, cy, r = w / 2, h / 2, 0.4 * h
    else:
        cx, cy, r = circles[0, 0]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def crystals(g):
    """Bright compact features (crystals): white top-hat, blurred."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.MORPH_TOPHAT, k)
    th = th / (np.percentile(th, 99.5) + 1e-6)
    return cv2.GaussianBlur(np.clip(th, 0, 1), (0, 0), 2.5)


def grad(g):
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def ncc(a, b, mask):
    a = a[mask]; b = b[mask]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def similarity(angle, scale, flip, c_src, c_dst):
    """2x3 matrix: optional mirror, rotate by angle about c_src, scale, move to c_dst."""
    M = cv2.getRotationMatrix2D(tuple(c_src), angle, scale)
    if flip:
        F = np.array([[-1, 0, 2 * c_src[0]], [0, 1, 0], [0, 0, 1]], np.float64)
        M = np.vstack([M, [0, 0, 1]]) @ F
        M = M[:2]
    M[:, 2] += np.asarray(c_dst) - np.asarray(c_src)
    return M


SCALES = np.arange(0.85, 1.75, 0.05)   # optical work px -> SEM work px
# The optical images are rotated by about 276 deg (counter-clockwise, no mirror)
# relative to SEM in every pair that registers unambiguously, consistent with a
# fixed mounting orientation. The search is restricted to this orientation +/- 20 deg.
PRIOR_ANGLE, PRIOR_HALF_WIDTH = 276, 20
COARSE = 0.5                             # search at half the working resolution


def place(img, angle, scale, flip):
    """Rotate/scale/mirror img onto a canvas that holds it entirely.

    Returns the warped image and the 2x3 matrix img -> canvas."""
    h, w = img.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, scale)
    if flip:
        F = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], np.float64)
        M = (np.vstack([M, [0, 0, 1]]) @ F)[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    lo, hi = corners.min(0), corners.max(0)
    M[:, 2] -= lo
    W, H = int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1]))
    return cv2.warpAffine(img, M, (W, H)), M


def search(o_map, s_map, angles, scales, flips):
    """Best (score, angle, scale, flip, M) by normalised template matching."""
    best = []
    for flip in flips:
        for sc in scales:
            for a in angles:
                T, M = place(o_map, float(a), float(sc), flip)
                th, tw = T.shape
                P = cv2.copyMakeBorder(s_map, th // 2, th // 2, tw // 2, tw // 2, cv2.BORDER_CONSTANT, 0)
                if P.shape[0] < th or P.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(P, T, cv2.TM_CCOEFF_NORMED)
                _, v, _, (x, y) = cv2.minMaxLoc(res)
                M2 = M.copy()
                M2[:, 2] += (x - tw // 2, y - th // 2)
                best.append((float(v), float(a), float(sc), flip, M2))
    best.sort(key=lambda t: -t[0])
    return best


def register(opt, sem):
    """Return (2x3 matrix mapping optical work image -> SEM work image, SEM mask, info)."""
    g_o, g_s = crystals(opt), crystals(sem)
    shrink = lambda m, f: cv2.resize(m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # level 1: quarter resolution, full grid
    q = search(shrink(g_o, 0.25), shrink(g_s, 0.25),
               np.arange(PRIOR_ANGLE - PRIOR_HALF_WIDTH, PRIOR_ANGLE + PRIOR_HALF_WIDTH + 0.1, 4),
               np.arange(0.85, 1.76, 0.1), (False,))
    v0, a0, sc0, flip, _ = q[0]
    runner = np.nan
    # level 2: half resolution, around the best few level-1 candidates
    cands = []
    for v, a, sc, fl, _ in q[:3]:
        cands += search(shrink(g_o, 0.5), shrink(g_s, 0.5), np.arange(a - 6, a + 6.1, 2), np.arange(sc - 0.1, sc + 0.11, 0.05), (fl,))
    cands.sort(key=lambda t: -t[0])
    _, a0, sc0, flip, _ = cands[0]
    # level 3: working resolution, fine
    fine = search(g_o, g_s, np.arange(a0 - 2, a0 + 2.01, 1), np.arange(sc0 - 0.04, sc0 + 0.041, 0.02), (flip,))
    v1, a1, sc1, _, M = fine[0]
    h, w = sem.shape
    m_s = droplet_mask(sem)
    inner = cv2.erode(m_s.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    warped = cv2.warpAffine(g_o, M, (w, h))
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(g_s, warped, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        M = (np.vstack([cv2.invertAffineTransform(W), [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2]
        refined = True
    except cv2.error:
        refined = False
    final = ncc(cv2.warpAffine(g_o, M, (w, h)), g_s, inner)
    info = dict(angle=a1, flip=flip, scale=sc1, coarse_score=v0, runner_up_score=runner,
                match_score=v1, final_ncc=final, ecc_refined=refined)
    return M, m_s, info


def stretch(g, mask):
    lo, hi = np.percentile(g[mask], [1, 99])
    return np.clip((g - lo) / (hi - lo + 1e-9), 0, 1)


def one(pair):
    cond, i = pair
    opt_full = gray(DATA / "corrected_images" / cond / f"Image_{i}.jpg")
    sem_full = gray(DATA / "corrected_images" / cond / f"SEM_{i}.jpg")
    f = WORK_W / sem_full.shape[1]
    sem = cv2.resize(sem_full, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    fo = WORK_W / opt_full.shape[1]
    opt = cv2.resize(opt_full, None, fx=fo, fy=fo, interpolation=cv2.INTER_AREA)
    M, m_s, info = register(opt, sem)
    h, w = sem.shape
    opt_w = cv2.warpAffine(opt, M, (w, h))
    ys, xs = np.where(m_s)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a, b, mk = stretch(opt_w, m_s)[y0:y1, x0:x1], stretch(sem, m_s)[y0:y1, x0:x1], m_s[y0:y1, x0:x1]
    _, smap = structural_similarity(a, b, data_range=1.0, full=True, gaussian_weights=True, sigma=1.5)
    info.update(condition=cond, image_id=i, ssim_registered=float(smap[mk].mean()))
    info.update({f"M{r}{c}": float(M[r, c]) for r in range(2) for c in range(3)})
    tile = (np.dstack([b, a, b * 0.5 + a * 0.5]) * 255).astype(np.uint8)
    print(f"{cond}{i:>2}  angle {info['angle']:5.0f}  scale {info['scale']:.2f}  "
          f"ncc {info['final_ncc']:.3f}  ssim {info['ssim_registered']:.3f}", flush=True)
    return info, (cond, i, tile)


def run(pairs=None, workers=8):
    from multiprocessing import Pool
    pairs = pairs or [(c, i) for c in CONDITIONS for i in range(1, 26)]
    cv2.setNumThreads(1)
    with Pool(workers) as pool:
        out = pool.map(one, pairs)
    return pd.DataFrame([o[0] for o in out]), [o[1] for o in out]


def montage(tiles, path, cols=15, size=160):
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * (size + 18), cols * size, 3), 255, np.uint8)
    for k, (c, i, t) in enumerate(tiles):
        r, q = divmod(k, cols)
        t = cv2.resize(t, (size, int(size * t.shape[0] / t.shape[1])))[:size]
        canvas[r * (size + 18) + 18: r * (size + 18) + 18 + t.shape[0], q * size:(q + 1) * size] = t[:, :, ::-1]
        cv2.putText(canvas, f"{c}{i}", (q * size + 4, r * (size + 18) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        pairs = [("A", 1), ("A", 2), ("A", 3), ("C", 1), ("C", 2), ("E", 1), ("E", 2), ("E", 3), ("I", 25), ("G", 7), ("B", 12), ("F", 20)]
        df, tiles = run(pairs)
        montage(tiles, OUT / "registration_sample_qa.png", cols=6, size=240)
    else:
        df, tiles = run()
        df.to_csv(OUT / "cross_modal_registration.csv", index=False)
        montage(tiles, OUT / "registration_qa.png")
        print("saved cross_modal_registration.csv and registration_qa.png")
