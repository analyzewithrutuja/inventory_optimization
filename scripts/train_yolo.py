"""Fine-tune a pretrained YOLOv8n model (transfer learning) on the
428-image, class-stratified grocery detection dataset."""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent

model = YOLO("yolov8n.pt")  # COCO-pretrained weights: theta_0 = theta_COCO

results = model.train(
    data=str(BASE / "yolo_dataset_split" / "data.yaml"),
    epochs=50,
    imgsz=416,
    batch=8,
    device="cpu",
    project=str(BASE / "runs"),
    name="grocery_detect",
    patience=15,  # stop early if val loss hasn't improved in 15 epochs
    seed=42,
)
