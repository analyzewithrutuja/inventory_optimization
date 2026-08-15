"""Pilot: auto-generate YOLO bounding boxes for single-item grocery photos
using FastSAM with a center-point prompt (handles background clutter better
than a full-frame box).

Run:  python scripts/auto_annotate_pilot.py
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
CLASSIFICATION_JSON = BASE / "classification.json"
OUT_DIR = BASE / "annotation_pilot"
OUT_IMAGES = OUT_DIR / "preview"
OUT_LABELS = OUT_DIR / "labels"
N_PER_CLASS = 10
PAD_FRAC = 0.04  # expand box by 4% each side to avoid clipping the product
FLAG_LOW = 0.20   # below this box-area fraction -> likely missed part of product
FLAG_HIGH = 0.65  # above this -> likely included background

CLASS_NAMES = ["barilla", "corn_flakes", "indomie", "keya_piri_piri", "nut_bars"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def pick_pilot_images():
    with open(CLASSIFICATION_JSON) as f:
        classification = json.load(f)

    by_class = {name: [] for name in CLASS_NAMES}
    for fname, cls in classification.items():
        if (IMAGES_DIR / fname).exists():
            by_class[cls].append(fname)

    random.seed(42)
    picked = []
    for cls, files in by_class.items():
        picked.extend((f, cls) for f in random.sample(files, min(N_PER_CLASS, len(files))))
    return picked


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    pilot = pick_pilot_images()
    print(f"Pilot set: {len(pilot)} images")
    for fname, cls in pilot:
        print(f"  {fname} -> {cls}")

    model = FastSAM("FastSAM-s.pt")

    for fname, cls in pilot:
        img_path = IMAGES_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"SKIP (unreadable): {fname}")
            continue
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # Center = positive (the product). Corners = negative (background).
        # Any mask that bleeds into a corner (e.g. merged with the textured
        # rack/table background) gets excluded by ultralytics' prompt logic,
        # since later points overwrite earlier ones for masks they share.
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        points = [
            [cx, cy],
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [margin_x, h - margin_y],
            [w - margin_x, h - margin_y],
        ]
        labels = [1, 0, 0, 0, 0]

        results = model.predict(str(img_path), points=points, labels=labels, device="cpu", verbose=False)
        r = results[0]

        # Corner-negative points can cancel out every mask when the product
        # is large/elongated enough to reach near the corners itself. Fall
        # back to a plain center-point prompt in that case.
        used_fallback = False
        if r.masks is None or len(r.masks.data) == 0:
            results = model.predict(str(img_path), points=[[cx, cy]], labels=[1], device="cpu", verbose=False)
            r = results[0]
            used_fallback = True

        if r.masks is None or len(r.masks.data) == 0:
            print(f"NO MASK: {fname}")
            continue

        # FastSAM's point-prompt returns every candidate mask that contains
        # the point (e.g. a small logo AND the whole box). Pick the largest
        # mask within a reasonable size band so we get "the whole object",
        # not a sub-part (too small) and not the whole scene (too big).
        areas = r.masks.data.sum(dim=(1, 2))
        mask_h, mask_w = r.masks.data.shape[1:]
        frac = areas / (mask_h * mask_w)
        valid = (frac > 0.03) & (frac < 0.85)
        if valid.any():
            areas = areas.clone()
            areas[~valid] = -1
        best_idx = int(areas.argmax())

        mask = r.masks.data[best_idx].cpu().numpy()
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bbox = mask_to_bbox(mask)
        if bbox is None:
            print(f"EMPTY MASK: {fname}")
            continue

        x1, y1, x2, y2 = bbox
        # Expand box by PAD_FRAC on each side to reduce the "missed part of
        # product" failure mode, clipped to image bounds.
        pad_x = int((x2 - x1) * PAD_FRAC)
        pad_y = int((y2 - y1) * PAD_FRAC)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)

        class_id = CLASS_TO_ID[cls]

        # YOLO normalized label
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        label_path = OUT_LABELS / (Path(fname).stem + ".txt")
        label_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        area_frac = bw * bh
        flagged = area_frac < FLAG_LOW or area_frac > FLAG_HIGH

        # Preview image with box drawn
        preview = img.copy()
        box_color = (0, 0, 255) if flagged else (0, 255, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), box_color, 3)
        cv2.circle(preview, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(preview, cls, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        cv2.imwrite(str(OUT_IMAGES / fname), preview)

        flag_tag = " <-- FLAGGED FOR REVIEW" if flagged else ""
        fb_tag = " [fallback:center-only]" if used_fallback else ""
        print(f"OK: {fname} -> bbox=({x1},{y1},{x2},{y2}) box_area_pct={area_frac*100:.1f}%{flag_tag}{fb_tag}")

    print(f"\nDone. Previews: {OUT_IMAGES}\nLabels: {OUT_LABELS}")


if __name__ == "__main__":
    main()
