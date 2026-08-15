"""Continue fine-tuning the grocery detector on the augmented training
set (298 original single-item images + 174 synthetic multi-object
composite images), starting from the previously trained weights so the
model keeps its single-item performance while learning multi-object
scenes."""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent

model = YOLO(str(BASE / "runs" / "grocery_detect" / "weights" / "best.pt"))

results = model.train(
    data=str(BASE / "yolo_dataset_split" / "data.yaml"),
    epochs=30,
    imgsz=416,
    batch=8,
    device="cpu",
    project=str(BASE / "runs"),
    name="grocery_detect_multiobj",
    patience=10,
    seed=42,
)
