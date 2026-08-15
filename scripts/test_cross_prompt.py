"""Test an improved prompt strategy on the currently-hardest classes:
cross-shaped positive points (center + up/down/left/right) so a visually
segmented object (bottle cap vs label vs base) gets fully covered, unioned
together, with 4 corner negative points to keep background out.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
MANIFEST = json.loads((BASE / "yolo_dataset_raw" / "review_manifest.json").read_text())
OUT_DIR = BASE / "test_cross_prompt_preview"
OUT_DIR.mkdir(exist_ok=True)

CORNER_MARGIN_FRAC = 0.05
OFFSET_FRAC = 0.15
N_TEST_PER_CLASS = 8


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def annotate_cross(model, img_path, w, h):
    cx, cy = w // 2, h // 2
    ox, oy = int(w * OFFSET_FRAC), int(h * OFFSET_FRAC)
    mx, my = int(w * CORNER_MARGIN_FRAC), int(h * CORNER_MARGIN_FRAC)
    points = [
        [cx, cy], [cx, cy - oy], [cx, cy + oy], [cx - ox, cy], [cx + ox, cy],
        [mx, my], [w - mx, my], [mx, h - my], [w - mx, h - my],
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0]
    r = model.predict(str(img_path), points=points, labels=labels, device="cpu", verbose=False)[0]

    if r.masks is None or len(r.masks.data) == 0:
        r = model.predict(str(img_path), points=[[cx, cy]], labels=[1], device="cpu", verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            return None
        masks = r.masks.data
    else:
        masks = r.masks.data

    union = (masks.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
    union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_to_bbox(union)


def main():
    targets = [x for x in MANIFEST["flagged"] if x["class"] in ("keya_piri_piri", "nut_bars")]
    targets += [x for x in MANIFEST["no_mask"] if x["class"] in ("keya_piri_piri", "nut_bars")]

    by_class = {}
    for t in targets:
        by_class.setdefault(t["class"], []).append(t["file"])

    sample = []
    for cls, files in by_class.items():
        sample += [(f, cls) for f in files[:N_TEST_PER_CLASS]]

    model = FastSAM("FastSAM-s.pt")

    for fname, cls in sample:
        img_path = IMAGES_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = annotate_cross(model, img_path, w, h)
        if bbox is None:
            print(f"NO MASK: {fname}")
            continue
        x1, y1, x2, y2 = bbox
        area_pct = (x2 - x1) * (y2 - y1) / (w * h) * 100
        preview = img.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(preview, f"{cls} {area_pct:.0f}%", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(OUT_DIR / fname), preview)
        print(f"{fname} ({cls}) -> area={area_pct:.1f}%")


if __name__ == "__main__":
    main()
