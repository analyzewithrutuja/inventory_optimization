"""Class-stratified 70/20/10 train/val/test split for the grocery
detection dataset. Reads classification.json for class labels and
yolo_dataset_raw/labels for the final (428-image) clean set, then
copies images + labels into a YOLOv8-ready folder layout."""
import json
import random
import shutil
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
LABELS_DIR = BASE / "yolo_dataset_raw" / "labels"
IMAGES_DIR = BASE / "CV_photos_resized"
CLASSIFICATION = json.loads((BASE / "classification.json").read_text())

OUT = BASE / "yolo_dataset_split"
CLASSES = ["barilla", "corn_flakes", "indomie", "keya_piri_piri", "nut_bars"]

# 1. Group the 428 clean, labeled images by class.
by_class = {c: [] for c in CLASSES}
for label_file in LABELS_DIR.glob("*.txt"):
    img_name = label_file.stem + ".jpg"
    cls = CLASSIFICATION[img_name]
    by_class[cls].append(img_name)

# 2. Stratified split: shuffle within each class, then slice by the
#    floor(0.7n) / floor(0.2n) / remainder formula.
splits = {"train": [], "val": [], "test": []}
for cls, files in by_class.items():
    files = sorted(files)          # deterministic order before shuffling
    random.shuffle(files)          # randomize within the class only
    n = len(files)
    n_train = int(0.7 * n)         # floor, since n is a positive int
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val   # remainder, guarantees the counts sum to n

    splits["train"] += [(f, cls) for f in files[:n_train]]
    splits["val"] += [(f, cls) for f in files[n_train:n_train + n_val]]
    splits["test"] += [(f, cls) for f in files[n_train + n_val:]]

    print(f"{cls:15s} n={n:3d}  train={n_train:3d}  val={n_val:3d}  test={n_test:3d}")

# 3. Copy into YOLOv8 folder structure: <split>/images/*.jpg, <split>/labels/*.txt
for split_name, items in splits.items():
    (OUT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split_name / "labels").mkdir(parents=True, exist_ok=True)
    for img_name, cls in items:
        stem = Path(img_name).stem
        shutil.copy(IMAGES_DIR / img_name, OUT / split_name / "images" / img_name)
        shutil.copy(LABELS_DIR / f"{stem}.txt", OUT / split_name / "labels" / f"{stem}.txt")

print()
for split_name, items in splits.items():
    print(f"{split_name}: {len(items)} images")
