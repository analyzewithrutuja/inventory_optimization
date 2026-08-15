"""Re-run the v1/v2 hybrid method (deterministic) for a specific list of
files whose GrabCut refinement made things worse, restoring their
pre-GrabCut boxes."""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
LABELS_DIR = BASE / "yolo_dataset_raw" / "labels"

PAD_FRAC = 0.04
CORNER_MARGIN_FRAC = 0.05
OFFSET_FRAC = 0.15
FLAG_LOW = 0.20
FLAG_HIGH = 0.65
V2_ACCEPT_LOW = 0.15
V2_ACCEPT_HIGH = 0.70

CLASS_TO_ID = {"barilla": 0, "corn_flakes": 1, "indomie": 2, "keya_piri_piri": 3, "nut_bars": 4}

TARGETS = [
    ("IMG_9803.jpg", "corn_flakes"), ("IMG_9816.jpg", "corn_flakes"),
    ("IMG_9820.jpg", "corn_flakes"), ("IMG_9823.jpg", "corn_flakes"),
    ("IMG_9824.jpg", "corn_flakes"), ("IMG_9828.jpg", "corn_flakes"),
    ("IMG_9734.jpg", "barilla"), ("IMG_9748.jpg", "nut_bars"),
    ("IMG_9751.jpg", "nut_bars"), ("IMG_9773.jpg", "indomie"),
    ("IMG_9888.jpg", "nut_bars"), ("IMG_9945.jpg", "corn_flakes"),
    ("IMG_9950.jpg", "corn_flakes"), ("IMG_9963.jpg", "barilla"),
    ("IMG_9972.jpg", "barilla"), ("IMG_9978.jpg", "barilla"),
]


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_area_frac(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1) / (w * h)


def v1_bbox(model, img_path, w, h):
    cx, cy = w // 2, h // 2
    mx, my = int(w * CORNER_MARGIN_FRAC), int(h * CORNER_MARGIN_FRAC)
    points = [[cx, cy], [mx, my], [w - mx, my], [mx, h - my], [w - mx, h - my]]
    labels = [1, 0, 0, 0, 0]
    r = model.predict(str(img_path), points=points, labels=labels, device="cpu", verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        r = model.predict(str(img_path), points=[[cx, cy]], labels=[1], device="cpu", verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None
    areas = r.masks.data.sum(dim=(1, 2))
    mh, mw = r.masks.data.shape[1:]
    frac = areas / (mh * mw)
    valid = (frac > 0.03) & (frac < 0.85)
    if valid.any():
        areas = areas.clone()
        areas[~valid] = -1
    best_idx = int(areas.argmax())
    mask = r.masks.data[best_idx].cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_to_bbox(mask)


def v2_bbox(model, img_path, w, h):
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
        return None
    union = (r.masks.data.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
    union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_to_bbox(union)


def annotate_one(model, img_path, w, h):
    b1 = v1_bbox(model, img_path, w, h)
    a1 = bbox_area_frac(b1, w, h) if b1 else None
    if a1 is not None and FLAG_LOW <= a1 <= FLAG_HIGH:
        return b1
    b2 = v2_bbox(model, img_path, w, h)
    a2 = bbox_area_frac(b2, w, h) if b2 else None
    if a2 is not None and V2_ACCEPT_LOW <= a2 <= V2_ACCEPT_HIGH:
        return b2
    return b1 if b1 is not None else b2


def main():
    model = FastSAM("FastSAM-s.pt")
    for fname, cls in TARGETS:
        img_path = IMAGES_DIR / fname
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        bbox = annotate_one(model, img_path, w, h)
        if bbox is None:
            print(f"{fname}: FAILED to restore")
            continue
        x1, y1, x2, y2 = bbox
        pad_x, pad_y = int((x2 - x1) * PAD_FRAC), int((y2 - y1) * PAD_FRAC)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        label_path = LABELS_DIR / (Path(fname).stem + ".txt")
        label_path.write_text(f"{CLASS_TO_ID[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"{fname} ({cls}) restored area={bw*bh*100:.1f}%")


if __name__ == "__main__":
    main()
