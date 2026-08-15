"""Continue fine-tuning on the further-augmented training set (298
original + 174 spaced-pair synthetic + 83 dense 3-5-item synthetic = 555
images), starting from the grocery_detect_multiobj weights."""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent

model = YOLO(str(BASE / "runs" / "grocery_detect_multiobj" / "weights" / "best.pt"))

results = model.train(
    data=str(BASE / "yolo_dataset_split" / "data.yaml"),
    epochs=25,
    imgsz=416,
    batch=8,
    device="cpu",
    project=str(BASE / "runs"),
    name="grocery_detect_dense",
    patience=10,
    seed=42,
)
