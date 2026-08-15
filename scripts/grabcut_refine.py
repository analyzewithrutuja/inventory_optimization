"""Refine loose/flagged bounding boxes using GrabCut, initialized from the
current box. Tests on a small sample first (pass --test), or refines all
flagged images and writes labels + preview (default)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
LABELS_DIR = BASE / "yolo_dataset_raw" / "labels"
MANIFEST_PATH = BASE / "yolo_dataset_raw" / "review_manifest.json"
OUT_PREVIEW = BASE / "yolo_dataset_raw" / "grabcut_preview"
OUT_PREVIEW.mkdir(exist_ok=True)


def load_box(label_path, w, h):
    if not label_path.exists():
        return None
    parts = label_path.read_text().split()
    _, xc, yc, bw, bh = (float(p) for p in parts)
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    return [x1, y1, x2, y2]


def refine(img, box, pad_frac=0.06):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    # Slightly expand the init rect so GrabCut has room to pull the true
    # boundary in from either side.
    px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
    rx1, ry1 = max(0, x1 - px), max(0, y1 - py)
    rx2, ry2 = min(w - 1, x2 + px), min(h - 1, y2 + py)
    rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    ys, xs = np.where(fg > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    items = manifest["flagged"]

    test_mode = "--test" in sys.argv
    if test_mode:
        items = items[:6]

    for item in items:
        fname, cls = item["file"], item["class"]
        img = cv2.imread(str(IMAGES_DIR / fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = LABELS_DIR / (Path(fname).stem + ".txt")
        box = load_box(label_path, w, h)
        if box is None:
            continue

        new_box = refine(img, box)
        old_area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
        new_area = None
        preview = img.copy()
        cv2.rectangle(preview, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        if new_box:
            new_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1]) / (w * h)
            cv2.rectangle(preview, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 3)

        print(f"{fname} ({cls}) old={old_area*100:.1f}% new={new_area*100:.1f}%" if new_area else f"{fname}: refine failed")
        cv2.imwrite(str(OUT_PREVIEW / fname), preview)

        if not test_mode and new_box:
            x1, y1, x2, y2 = new_box
            cid_map = {"barilla": 0, "corn_flakes": 1, "indomie": 2, "keya_piri_piri": 3, "nut_bars": 4}
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            label_path.write_text(f"{cid_map[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


if __name__ == "__main__":
    main()
