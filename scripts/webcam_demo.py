"""Live webcam detection demo using the fine-tuned YOLOv8 grocery model.

Controls:
  SPACE - run detection on the current frame
  q     - quit
"""
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = BASE / "runs" / "grocery_detect_multiobj" / "weights" / "best.pt"
OUT_DIR = BASE / "webcam_detections"
OUT_DIR.mkdir(exist_ok=True)

model = YOLO(str(WEIGHTS))
shot = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Warm-up: first frames are often black while the camera auto-exposes.
for _ in range(15):
    cap.read()
    time.sleep(0.05)

print("Press SPACE to run detection on the current frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Webcam - press SPACE to detect, q to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        shot += 1
        cv2.imwrite(str(OUT_DIR / f"raw_{shot}.jpg"), frame)

        results = model.predict(frame, conf=0.5, device="cpu", verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Detection result", annotated)
        cv2.imwrite(str(OUT_DIR / f"detection_{shot}.jpg"), annotated)
        print(f"  Saved raw_{shot}.jpg and detection_{shot}.jpg")

        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  {cls_name}: {conf:.2f} confidence")
        if len(results[0].boxes) == 0:
            print("  No objects detected.")

cap.release()
cv2.destroyAllWindows()
