"""Opens the webcam and keeps it open for a live detection loop:
every few seconds it grabs the current frame and runs the fine-tuned
YOLOv8 grocery model on it, saving each annotated result."""
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = BASE / "runs" / "grocery_detect" / "weights" / "best.pt"
OUT_DIR = BASE / "webcam_detections"
OUT_DIR.mkdir(exist_ok=True)

NUM_SHOTS = 5
DELAY_SECONDS = 60

model = YOLO(str(WEIGHTS))
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print(f"Camera opened. Will capture + detect {NUM_SHOTS} times, {DELAY_SECONDS}s apart.")
print("Position a grocery item in front of the webcam now.")

try:
    for shot in range(1, NUM_SHOTS + 1):
        print(f"[{shot}] Get ready — capturing in {DELAY_SECONDS}s...")
        time.sleep(DELAY_SECONDS)
        ret, frame = cap.read()
        if not ret:
            print(f"[{shot}] Failed to read frame.")
            continue

        results = model.predict(frame, conf=0.5, device="cpu", verbose=False)
        annotated = results[0].plot()
        out_path = OUT_DIR / f"live_{shot}.jpg"
        cv2.imwrite(str(out_path), annotated)

        if len(results[0].boxes) == 0:
            print(f"[{shot}] No objects detected. Saved {out_path.name}")
        else:
            dets = ", ".join(
                f"{model.names[int(b.cls[0])]} ({float(b.conf[0]):.2f})"
                for b in results[0].boxes
            )
            print(f"[{shot}] Detected: {dets}. Saved {out_path.name}")
finally:
    cap.release()
    print("Camera released.")
