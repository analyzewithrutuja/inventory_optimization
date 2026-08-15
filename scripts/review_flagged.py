"""Interactive manual review for the 107 flagged auto-annotated images.

Run this yourself in a terminal (needs a display, GUI window opens):
    python scripts/review_flagged.py

For each flagged image:
    - The current auto-generated box is shown in green.
    - Click and drag to draw a NEW box (replaces the shown one).
    - Press:
        s = save current box (drawn or auto) and go to next
        x = exclude this image from the dataset entirely (deletes its label)
        b = go back to previous image
        q = quit (progress so far is already saved to disk)
Progress is saved to the label file immediately on 's' / 'x', so you can
quit anytime with 'q' and resume later (already-handled images are skipped
next run).
"""
import json
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "CV_photos_resized"
LABELS_DIR = BASE / "yolo_dataset_raw" / "labels"
MANIFEST_PATH = BASE / "yolo_dataset_raw" / "review_manifest.json"
EXCLUDED_PATH = BASE / "yolo_dataset_raw" / "excluded.json"
REVIEWED_PATH = BASE / "yolo_dataset_raw" / "reviewed.json"

CLASS_NAMES = ["barilla", "corn_flakes", "indomie", "keya_piri_piri", "nut_bars"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

WINDOW = "review (drag=new box, s=save+next, x=exclude, b=back, q=quit)"


def load_box(label_path, w, h):
    if not label_path.exists():
        return None
    parts = label_path.read_text().split()
    if len(parts) != 5:
        return None
    _, xc, yc, bw, bh = (float(p) for p in parts)
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    return [x1, y1, x2, y2]


def save_box(label_path, cls, box, w, h):
    x1, x2 = sorted((box[0], box[2]))
    y1, y2 = sorted((box[1], box[3]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cid = CLASS_TO_ID[cls]
    xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    label_path.write_text(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    items = manifest["flagged"] + manifest["no_mask"]
    print(f"{len(items)} images to review.")

    reviewed = set(json.loads(REVIEWED_PATH.read_text())) if REVIEWED_PATH.exists() else set()
    excluded = json.loads(EXCLUDED_PATH.read_text()) if EXCLUDED_PATH.exists() else []

    state = {"box": None, "drawing": False, "ix": 0, "iy": 0}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["ix"], state["iy"] = x, y
            state["box"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["box"] = [state["ix"], state["iy"], x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["box"] = [state["ix"], state["iy"], x, y]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    idx = 0
    while idx < len(items):
        item = items[idx]
        fname, cls = item["file"], item["class"]
        if fname in reviewed:
            idx += 1
            continue

        img = cv2.imread(str(IMAGES_DIR / fname))
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        label_path = LABELS_DIR / (Path(fname).stem + ".txt")
        state["box"] = load_box(label_path, w, h)

        advance = 0
        while True:
            disp = img.copy()
            if state["box"]:
                b = state["box"]
                cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(disp, f"{idx+1}/{len(items)} {fname} [{cls}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(disp, "drag=new box  s=save+next  x=exclude  b=back  q=quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(WINDOW, disp)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                if state["box"] is None:
                    continue
                save_box(label_path, cls, state["box"], w, h)
                reviewed.add(fname)
                advance = 1
                break
            elif key == ord("x"):
                if label_path.exists():
                    label_path.unlink()
                excluded.append(fname)
                reviewed.add(fname)
                advance = 1
                break
            elif key == ord("b"):
                advance = -1
                break
            elif key == ord("q"):
                advance = None
                break

        REVIEWED_PATH.write_text(json.dumps(sorted(reviewed), indent=2))
        EXCLUDED_PATH.write_text(json.dumps(excluded, indent=2))

        if advance is None:
            break
        idx = max(0, idx + advance)

    cv2.destroyAllWindows()
    print(f"Reviewed: {len(reviewed)}/{len(items)}  Excluded: {len(excluded)}")
    if len(reviewed) < len(items):
        print("Run this script again to continue where you left off.")


if __name__ == "__main__":
    main()
