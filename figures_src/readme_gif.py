"""Animated overview for the README (assets/dataset_tour.gif).

One deposit is shown as the optical image, wiped across to the SEM image of the
same deposit, overlaid after registration, then layered with EDS element maps;
the view finally zooms out to the same water sample under all nine conditions.
Brightness and the EDS maps are enhanced for display only.

    python figures_src/readme_gif.py        (needs data/ and ffmpeg)
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "assets" / "dataset_tour.gif"

CONDITION, DROPLET = "E", 2                 # condition E is the centre of the 3 x 3 grid
W, H, FOOT = 640, 480, 52                   # image area and footer height, px
FPS = 8
UM_PER_PX = 1.0 * 2560 / W                  # SEM: nominal 1.0 um per original pixel
ELEMENTS = [("Ca", "Ca Kα1.png", (64, 220, 140)), ("Cl", "Cl Kα1.png", (80, 200, 245)),
            ("S", "S Kα1.png", (255, 205, 90)), ("Na", "Na Kα1_2.png", (255, 105, 150))]
SCENES = ["Optical", "SEM", "Registered", "EDS", "9 conditions"]
BG, INK, DIM, ACCENT = (12, 14, 18), (236, 238, 242), (120, 126, 138), (255, 205, 90)

FONT_DIR = Path("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF")


def font(weight, size):
    for p in [FONT_DIR / f"Roboto-{weight}.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


F_TAG, F_FOOT, F_SMALL = font("Medium", 15), font("Regular", 14), font("Regular", 12)


# ── data ──────────────────────────────────────────────────────────────────────
def fit(img, w=W, h=H):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def registered_optical(condition, droplet):
    """Optical image warped into the SEM frame with the published registration (defined at 640 px width)."""
    opt = load_rgb(DATA / "corrected_images" / condition / f"Image_{droplet}.jpg")
    opt = cv2.resize(opt, (640, round(opt.shape[0] * 640 / opt.shape[1])), interpolation=cv2.INTER_AREA)
    reg = pd.read_csv(DATA / "features" / "cross_modal_registration.csv")
    r = reg[(reg.condition == condition) & (reg.image_id == droplet)].iloc[0]
    M = np.array([[r.M00, r.M01, r.M02], [r.M10, r.M11, r.M12]])
    warped = cv2.warpAffine(opt, M, (640, 480))
    return fit(warped), float(r.registration_ncc)


def eds_density(condition, droplet, fname):
    """EDS counts map as a smoothed 0..1 density in the display frame."""
    a = np.asarray(Image.open(DATA / "eds" / condition / str(droplet) / fname).convert("RGB")).max(axis=2)
    a = cv2.GaussianBlur(fit(a.astype(np.float32)), (0, 0), 3.0)
    lo, hi = np.percentile(a, 80), np.percentile(a, 99.7)       # drop the uniform background counts
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) ** 0.8


def sem(condition, droplet):
    return fit(load_rgb(DATA / "corrected_images" / condition / f"SEM_{droplet}.jpg"))


# ── drawing helpers ───────────────────────────────────────────────────────────
def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def blend(a, b, t):
    return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def chip(draw, xy, text, fnt=F_TAG, dot=None, fill=(0, 0, 0, 150)):
    x, y = xy
    tw = draw.textlength(text, font=fnt)
    pad, dw = 9, (16 if dot else 0)
    h = fnt.size + 12
    draw.rounded_rectangle([x, y, x + tw + 2 * pad + dw, y + h], radius=h // 2, fill=fill)
    if dot:
        cy = y + h / 2
        draw.ellipse([x + pad, cy - 5, x + pad + 10, cy + 5], fill=dot)
    draw.text((x + pad + dw, y + h / 2), text, font=fnt, fill=INK, anchor="lm")
    return x + tw + 2 * pad + dw


def scale_bar(draw, um=500):
    n = um / UM_PER_PX
    x1, y = W - 24, H - 26
    draw.rounded_rectangle([x1 - n - 10, y - 26, x1 + 10, y + 10], radius=8, fill=(0, 0, 0, 120))
    draw.rectangle([x1 - n, y - 2, x1, y + 2], fill=INK)
    draw.text((x1 - n / 2, y - 7), f"{um} µm", font=F_SMALL, fill=INK, anchor="mb")


def compose(img, scene, progress, tags=(), bar=True, extra=None):
    """img: H x W x 3 uint8 image area -> full frame with overlays and footer."""
    frame = Image.new("RGB", (W, H + FOOT), BG)
    frame.paste(Image.fromarray(img), (0, 0))
    over = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    x = 16
    for t in tags:
        x = chip(d, (x, 16), *t) + 8 if isinstance(t, tuple) else chip(d, (x, 16), t) + 8
    if bar:
        scale_bar(d)
    if extra:
        extra(d)
    # footer: scene labels with the active one highlighted, and a thin progress line
    d.rectangle([0, H, W, H + 2], fill=(40, 44, 52))
    d.rectangle([0, H, int(W * progress), H + 2], fill=ACCENT)
    xs = 18
    for i, s in enumerate(SCENES):
        col = INK if i == scene else DIM
        d.text((xs, H + FOOT / 2 + 1), s, font=F_FOOT, fill=col, anchor="lm")
        if i == scene:
            tw = d.textlength(s, font=F_FOOT)
            d.rectangle([xs, H + FOOT / 2 + 12, xs + tw, H + FOOT / 2 + 14], fill=ACCENT)
        xs += d.textlength(s, font=F_FOOT) + 22
    d.text((W - 18, H + FOOT / 2 + 1), "225 deposits · 9 conditions", font=F_SMALL, fill=DIM, anchor="rm")
    frame.paste(over, (0, 0), over)
    return frame


# ── animation ─────────────────────────────────────────────────────────────────
def build_frames():
    env = pd.read_csv(DATA / "metadata" / "environmental_conditions.csv").set_index("condition")
    opt, ncc = registered_optical(CONDITION, DROPLET)
    s = sem(CONDITION, DROPLET)
    gray = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    og = cv2.cvtColor(opt, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255

    def features(g):                                  # bright crystals above the local background
        f = np.clip(g - cv2.GaussianBlur(g, (0, 0), 12), 0, None)
        return np.clip(f / (np.percentile(f, 99.7) + 1e-6), 0, 1) ** 1.6   # keep crystals, drop faint speckle
    fs, fo = features(gray), features(og)
    overlay = (np.dstack([fs, fo, fs]) * 255).astype(np.uint8)            # SEM magenta, optical green, both white
    eds = [(sym, eds_density(CONDITION, DROPLET, f), np.array(col, np.float32) / 255) for sym, f, col in ELEMENTS]

    def eds_layer(n_full, t_next):
        base = (np.dstack([gray] * 3) * 0.35)
        acc = np.zeros_like(base)
        for k, (_, dens, col) in enumerate(eds):
            w = 1.0 if k < n_full else (ease(t_next) if k == n_full else 0.0)
            acc = 1 - (1 - acc) * (1 - w * dens[..., None] * col)              # screen blend
        return (np.clip(1 - (1 - base) * (1 - acc), 0, 1) * 255).astype(np.uint8)

    tiles = {c: sem(c, DROPLET) for c in "ABCDEFGHI"}
    tw, th = W // 3, H // 3
    grid = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate("ABCDEFGHI"):
        r, q = divmod(k, 3)
        grid[r * th:(r + 1) * th, q * tw:(q + 1) * tw] = fit(tiles[c], tw, th)

    def grid_labels(d, alpha=1.0):
        for k, c in enumerate("ABCDEFGHI"):
            r, q = divmod(k, 3)
            e = env.loc[c]
            txt = f"{c}   {e.temperature_min_C}–{e.temperature_max_C} °C · {e.humidity_min_pct}–{e.humidity_max_pct}% RH"
            x0, y0 = q * tw + 6, r * th + 6
            a = int(150 * alpha)
            tl = d.textlength(txt, font=F_SMALL)
            d.rounded_rectangle([x0, y0, x0 + tl + 14, y0 + 22], radius=11, fill=(0, 0, 0, a))
            d.text((x0 + 7, y0 + 11), txt, font=F_SMALL, fill=INK + (int(255 * alpha),), anchor="lm")
        for i in (1, 2):
            d.line([(i * tw, 0), (i * tw, H)], fill=(12, 14, 18, int(255 * alpha)), width=2)
            d.line([(0, i * th), (W, i * th)], fill=(12, 14, 18, int(255 * alpha)), width=2)

    e = env.loc[CONDITION]
    where = f"Condition {CONDITION} · {e.temperature_min_C}–{e.temperature_max_C} °C · {e.humidity_min_pct}–{e.humidity_max_pct}% RH"
    plan = []                                            # (n_frames, renderer(t) -> PIL frame)
    sec = lambda x: max(1, round(x * FPS))

    plan.append((sec(1.2), lambda t: (opt, 0, ["Optical microscope", where], False, None)))

    def wipe(t):
        u = ease(t)
        xc = int(W * u)
        img = opt.copy()
        img[:, :xc] = s[:, :xc]
        def line(d):
            d.line([(xc, 0), (xc, H)], fill=INK + (230,), width=2)
            if 70 < xc < W - 90:
                y = H - 48
                tl = d.textlength("SEM", font=F_TAG)
                chip(d, (xc - tl - 18 - 8, y), "SEM")
                chip(d, (xc + 8, y), "Optical")
        return img, 1, ["Same deposit, two instruments"], False, line
    plan.append((sec(2.0), wipe))
    plan.append((sec(1.2), lambda t: (s, 1, ["Scanning electron microscope", "JEOL JSM-6610LV · 50×"], True, None)))
    plan.append((sec(0.9), lambda t: (blend(s, overlay, ease(t)), 2, ["Registered overlay"], True, None)))

    def overlay_hold(t):
        def legend(d):
            x = chip(d, (16, 56), "SEM", F_SMALL, dot=(230, 90, 230)) + 8
            chip(d, (x, 56), "Optical", F_SMALL, dot=(90, 230, 120))
        return overlay, 2, ["Registered overlay", f"NCC {ncc:.2f}"], True, legend
    plan.append((sec(1.6), overlay_hold))
    plan.append((sec(0.8), lambda t: (blend(overlay, eds_layer(0, 0), ease(t)), 3, ["EDS element maps"], True, None)))
    for k in range(len(eds)):
        def add(t, k=k):
            chips = [("EDS element maps",)] + [(sym, F_TAG, tuple(int(c * 255) for c in col)) for sym, _, col in eds[:k + 1]]
            return eds_layer(k, t), 3, chips, True, None
        plan.append((sec(0.8), add))
    plan.append((sec(1.4), lambda t: (eds_layer(len(eds), 0), 3,
                                      [("EDS element maps",)] + [(sym, F_TAG, tuple(int(c * 255) for c in col)) for sym, _, col in eds],
                                      True, None)))
    final_eds = eds_layer(len(eds), 0)

    def zoom(t):
        u = ease(t)
        # shrink the full frame into the centre tile while the grid fades in around it
        w_, h_ = int(W + (tw - W) * u), int(H + (th - H) * u)
        x0, y0 = (W - w_) // 2, (H - h_) // 2
        img = (grid.astype(np.float32) * u).astype(np.uint8)
        src = blend(final_eds, fit(tiles[CONDITION], W, H), min(1, u * 1.6))
        img[y0:y0 + h_, x0:x0 + w_] = fit(src, w_, h_)
        return img, 4, [], False, (lambda d: grid_labels(d, u)) if u > 0.6 else None
    plan.append((sec(1.4), zoom))
    plan.append((sec(2.4), lambda t: (grid, 4, [], False, lambda d: grid_labels(d))))
    plan.append((sec(0.7), lambda t: (blend(grid, opt, ease(t)), 0, [], False, None)))

    total = sum(n for n, _ in plan)
    i = 0
    for n, fn in plan:
        for j in range(n):
            img, scene, tags, bar, extra = fn(j / max(1, n - 1))
            tags = [tuple(x) if isinstance(x, (tuple, list)) else x for x in tags]
            yield compose(img, scene, (i + 1) / total, tags, bar, extra)
            i += 1


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        for k, fr in enumerate(build_frames()):
            fr.save(tmp / f"f{k:04d}.png")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pal = tmp / "palette.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                        "-vf", "palettegen=max_colors=192:stats_mode=diff", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                        "-i", str(pal), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", str(OUT)], check=True)
        print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {k + 1} frames)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
