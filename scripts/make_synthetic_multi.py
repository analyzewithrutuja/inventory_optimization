"""Generate synthetic multi-object training images via copy-paste
augmentation. Uses the existing 298 single-item training images and their
known YOLO bounding boxes -- no new photography or manual annotation.

For each synthetic image:
  1. Pick a random base image; keep its object + box unchanged.
  2. Pick 1-2 more random images; crop each one's bbox region (with a
     small padding) and paste it onto the base canvas at a random
     location that doesn't overlap existing boxes.
  3. The pasted object's new box is derived purely from arithmetic on
     the paste position and paste size (no re-labeling needed).
"""
import random
from pathlib import Path
import cv2

random.seed(7)

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "yolo_dataset_split" / "train" / "images"
LBL_DIR = BASE / "yolo_dataset_split" / "train" / "labels"

NUM_SYNTHETIC = 500
MAX_PASTE_TRIES = 40
PAD_FRAC = 0.03  # small padding around the cropped source box


def load_labels(stem):
    lines = (LBL_DIR / f"{stem}.txt").read_text().strip().split("\n")
    boxes = []
    for line in lines:
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def denorm_box(box, W, H):
    cls, xc, yc, w, h = box
    bw, bh = w * W, h * H
    x1 = xc * W - bw / 2
    y1 = yc * H - bh / 2
    return cls, int(x1), int(y1), int(bw), int(bh)


def overlaps(x1, y1, w, h, existing):
    for (_, ex1, ey1, ew, eh) in existing:
        if x1 < ex1 + ew and x1 + w > ex1 and y1 < ey1 + eh and y1 + h > ey1:
            return True
    return False


def main():
    stems = [p.stem for p in IMG_DIR.glob("*.jpg")]
    out_count = 0

    for i in range(NUM_SYNTHETIC):
        base_stem = random.choice(stems)
        base_img = cv2.imread(str(IMG_DIR / f"{base_stem}.jpg"))
        H, W = base_img.shape[:2]
        base_boxes_px = [denorm_box(b, W, H) for b in load_labels(base_stem)]
        canvas = base_img.copy()
        placed_boxes_px = list(base_boxes_px)  # (cls, x1, y1, w, h) in pixels

        num_extra = random.choice([1, 1, 2])  # mostly pairs, sometimes triples
        for _ in range(num_extra):
            src_stem = random.choice(stems)
            if src_stem == base_stem:
                continue
            src_img = cv2.imread(str(IMG_DIR / f"{src_stem}.jpg"))
            sH, sW = src_img.shape[:2]
            src_boxes = load_labels(src_stem)
            cls, x1, y1, w, h = denorm_box(src_boxes[0], sW, sH)

            pad_w, pad_h = int(w * PAD_FRAC), int(h * PAD_FRAC)
            x1p, y1p = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2p = min(sW, x1 + w + pad_w)
            y2p = min(sH, y1 + h + pad_h)
            crop = src_img[y1p:y2p, x1p:x2p]
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                continue

            # random scale so pasted objects vary in size (biased smaller,
            # so they fit more easily into leftover space on the canvas)
            scale = random.uniform(0.45, 0.85)
            new_w, new_h = int(cw * scale), int(ch * scale)
            if new_w >= W or new_h >= H:
                continue
            resized = cv2.resize(crop, (new_w, new_h))

            placed = False
            for _try in range(MAX_PASTE_TRIES):
                px = random.randint(0, W - new_w)
                py = random.randint(0, H - new_h)
                if not overlaps(px, py, new_w, new_h, placed_boxes_px):
                    canvas[py:py + new_h, px:px + new_w] = resized
                    placed_boxes_px.append((cls, px, py, new_w, new_h))
                    placed = True
                    break
            if not placed:
                continue

        if len(placed_boxes_px) < 2:
            continue  # paste failed for all extras; skip, not useful as "multi-object"

        out_count += 1
        out_stem = f"synth_{out_count:04d}"
        cv2.imwrite(str(IMG_DIR / f"{out_stem}.jpg"), canvas)

        label_lines = []
        for (cls, x1, y1, w, h) in placed_boxes_px:
            xc = (x1 + w / 2) / W
            yc = (y1 + h / 2) / H
            nw = w / W
            nh = h / H
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (LBL_DIR / f"{out_stem}.txt").write_text("\n".join(label_lines) + "\n")

    print(f"Generated {out_count} synthetic multi-object training images.")


if __name__ == "__main__":
    main()
