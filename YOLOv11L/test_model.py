import sys
from pathlib import Path
from ultralytics import YOLO

def run_detection(model_path: str = './best.pt', image_path: str = None, conf: float = 0.25):
    if image_path is None:
        print("ERROR: image_path is required.")
        sys.exit(1)

    model  = YOLO(model_path)
    chosen = Path(image_path)

    if not chosen.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    results = model(str(chosen), conf=conf, verbose=False)
    result  = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        results = []
        for box in result.boxes:
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 *= 2
            x2 *= 2
            results.append({"bbox":[x1, y1, x2, y2], "confidence":conf_score, "predicted_class":cls_id})
        return results
    else:
        return []

