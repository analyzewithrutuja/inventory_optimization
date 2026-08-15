"""Generate DENSE synthetic multi-object training images: 3-5 items
arranged in a row with small/negative gaps (touching or slightly
overlapping), matching the tightly-packed real-world layout that the
first round of synthetic data (make_synthetic_multi.py, which only used
spaced-apart pairs) did not cover well."""
import random
from pathlib import Path
import cv2

random.seed(11)

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "yolo_dataset_split" / "train" / "images"
LBL_DIR = BASE / "yolo_dataset_split" / "train" / "labels"

NUM_SYNTHETIC = 220
PAD_FRAC = 0.03


def load_labels(stem):
    lines = (LBL_DIR / f"{stem}.txt").read_text().strip().split("\n")
    boxes = []
    for line in lines:
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def crop_object(stem):
    img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
    H, W = img.shape[:2]
    cls, xc, yc, w, h = load_labels(stem)[0]
    bw, bh = w * W, h * H
    x1 = int(xc * W - bw / 2)
    y1 = int(yc * H - bh / 2)
    pad_w, pad_h = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
    x1p, y1p = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2p = min(W, x1 + int(bw) + pad_w)
    y2p = min(H, y1 + int(bh) + pad_h)
    crop = img[y1p:y2p, x1p:x2p]
    return cls, crop


def main():
    # only use "real" originals as crop sources (skip earlier synthetic
    # images, so we're not compounding paste artifacts on paste artifacts)
    stems = [p.stem for p in IMG_DIR.glob("*.jpg") if not p.stem.startswith("synth")]
    base_stems = [p.stem for p in IMG_DIR.glob("*.jpg")]  # any image ok as background
    out_count = 0

    for i in range(NUM_SYNTHETIC):
        canvas = cv2.imread(str(IMG_DIR / f"{random.choice(base_stems)}.jpg"))
        H, W = canvas.shape[:2]

        n_items = random.choice([3, 3, 3, 4, 4, 5])
        chosen = random.sample(stems, min(n_items, len(stems)))

        target_h = int(H * random.uniform(0.22, 0.38))
        crops = []
        for stem in chosen:
            cls, crop = crop_object(stem)
            ch, cw = crop.shape[:2]
            if ch < 5 or cw < 5:
                continue
            scale = target_h / ch
            new_w, new_h = max(5, int(cw * scale)), target_h
            resized = cv2.resize(crop, (new_w, new_h))
            crops.append((cls, resized))

        if len(crops) < 2:
            continue

        total_w = sum(c.shape[1] for _, c in crops)
        # negative gap = slight overlap/touching, positive = small visible gap
        gap = int(random.uniform(-0.15, 0.05) * (total_w / len(crops)))
        row_w = total_w + gap * (len(crops) - 1)
        if row_w >= W:
            continue

        start_x = random.randint(0, W - row_w)
        y0 = random.randint(0, H - target_h)

        placed_boxes_px = []
        x = start_x
        for cls, crop in crops:
            ch, cw = crop.shape[:2]
            x_end = min(W, x + cw)
            w_eff = x_end - x
            if w_eff <= 0:
                x += cw + gap
                continue
            canvas[y0:y0 + ch, x:x_end] = crop[:, :w_eff]
            placed_boxes_px.append((cls, x, y0, cw, ch))  # full extent, even if next item overlaps it
            x += cw + gap

        if len(placed_boxes_px) < 2:
            continue

        out_count += 1
        out_stem = f"dense_{out_count:04d}"
        cv2.imwrite(str(IMG_DIR / f"{out_stem}.jpg"), canvas)

        label_lines = []
        for (cls, x1, y1, w, h) in placed_boxes_px:
            xc = (x1 + w / 2) / W
            yc = (y1 + h / 2) / H
            nw = w / W
            nh = h / H
            # clip in case an item was pushed slightly outside canvas bounds
            xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
            nw, nh = min(nw, 1), min(nh, 1)
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        (LBL_DIR / f"{out_stem}.txt").write_text("\n".join(label_lines) + "\n")

    print(f"Generated {out_count} dense synthetic multi-object training images.")


if __name__ == "__main__":
    main()
