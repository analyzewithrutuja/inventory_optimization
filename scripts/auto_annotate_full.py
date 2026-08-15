"""Auto-generate YOLO bounding boxes for all grocery photos using FastSAM
with a center-positive + corner-negative point prompt (falls back to a
plain center-point prompt if the corner-negative version yields no mask).

Run:  python scripts/auto_annotate_full.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
CLASSIFICATION_JSON = BASE / "classification.json"
OUT_DIR = BASE / "yolo_dataset_raw"
OUT_LABELS = OUT_DIR / "labels"
OUT_FLAGGED_PREVIEW = OUT_DIR / "flagged_preview"
MANIFEST_PATH = OUT_DIR / "review_manifest.json"

PAD_FRAC = 0.04
FLAG_LOW = 0.20
FLAG_HIGH = 0.65
CORNER_MARGIN_FRAC = 0.05
OFFSET_FRAC = 0.15

CLASS_NAMES = ["barilla", "corn_flakes", "indomie", "keya_piri_piri", "nut_bars"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def annotate_one(model, img_path, w, h):
    # Cross-shaped positive points (center + up/down/left/right) so a
    # visually segmented object (e.g. bottle cap vs label vs base) gets
    # fully covered; 4 corner negative points keep background out. All
    # returned masks are unioned together (not just the single largest),
    # since different positive points may each land on a different part
    # of the same physical object.
    cx, cy = w // 2, h // 2
    ox, oy = int(w * OFFSET_FRAC), int(h * OFFSET_FRAC)
    mx, my = int(w * CORNER_MARGIN_FRAC), int(h * CORNER_MARGIN_FRAC)
    points = [
        [cx, cy], [cx, cy - oy], [cx, cy + oy], [cx - ox, cy], [cx + ox, cy],
        [mx, my], [w - mx, my], [mx, h - my], [w - mx, h - my],
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0]

    r = model.predict(str(img_path), points=points, labels=labels, device="cpu", verbose=False)[0]

    used_fallback = False
    if r.masks is None or len(r.masks.data) == 0:
        r = model.predict(str(img_path), points=[[cx, cy]], labels=[1], device="cpu", verbose=False)[0]
        used_fallback = True

    if r.masks is None or len(r.masks.data) == 0:
        return None, used_fallback

    union = (r.masks.data.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
    union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    bbox = mask_to_bbox(union)
    return bbox, used_fallback


def main():
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    OUT_FLAGGED_PREVIEW.mkdir(parents=True, exist_ok=True)

    with open(CLASSIFICATION_JSON) as f:
        classification = json.load(f)

    items = [(fname, cls) for fname, cls in classification.items() if (IMAGES_DIR / fname).exists()]
    print(f"Total images to annotate: {len(items)}")

    model = FastSAM("FastSAM-s.pt")

    flagged = []
    no_mask = []
    per_class_total = {c: 0 for c in CLASS_NAMES}
    per_class_bad = {c: 0 for c in CLASS_NAMES}

    for i, (fname, cls) in enumerate(items, 1):
        img_path = IMAGES_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{i}/{len(items)}] SKIP (unreadable): {fname}")
            continue
        h, w = img.shape[:2]
        per_class_total[cls] += 1

        bbox, used_fallback = annotate_one(model, img_path, w, h)
        if bbox is None:
            print(f"[{i}/{len(items)}] NO MASK: {fname}")
            no_mask.append({"file": fname, "class": cls})
            per_class_bad[cls] += 1
            continue

        x1, y1, x2, y2 = bbox
        pad_x, pad_y = int((x2 - x1) * PAD_FRAC), int((y2 - y1) * PAD_FRAC)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_y)

        class_id = CLASS_TO_ID[cls]
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        area_frac = bw * bh

        label_path = OUT_LABELS / (Path(fname).stem + ".txt")
        label_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        flagged_this = area_frac < FLAG_LOW or area_frac > FLAG_HIGH
        if flagged_this:
            per_class_bad[cls] += 1
            flagged.append({
                "file": fname, "class": cls, "area_pct": round(area_frac * 100, 1),
                "fallback": used_fallback,
            })
            preview = img.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(preview, f"{cls} {area_frac*100:.0f}%", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(str(OUT_FLAGGED_PREVIEW / fname), preview)

        status = "FLAGGED" if flagged_this else "ok"
        print(f"[{i}/{len(items)}] {fname} ({cls}) area={area_frac*100:.1f}% {status}")

    manifest = {
        "total": len(items),
        "flagged_count": len(flagged),
        "no_mask_count": len(no_mask),
        "per_class_total": per_class_total,
        "per_class_bad": per_class_bad,
        "flagged": flagged,
        "no_mask": no_mask,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("\n=== SUMMARY ===")
    for c in CLASS_NAMES:
        t, b = per_class_total[c], per_class_bad[c]
        pct = (b / t * 100) if t else 0
        print(f"{c}: {b}/{t} flagged ({pct:.0f}%)")
    print(f"TOTAL: {len(flagged)} flagged + {len(no_mask)} no-mask out of {len(items)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Labels: {OUT_LABELS}")
    print(f"Flagged previews: {OUT_FLAGGED_PREVIEW}")


if __name__ == "__main__":
    main()
