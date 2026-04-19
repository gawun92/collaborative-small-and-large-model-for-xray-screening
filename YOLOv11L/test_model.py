import random
import sys
from pathlib import Path

# import cv2
from ultralytics import YOLO

MODEL_PATH   = 'checkpoints/yolov11l/best.pt'
TEST_IMG_DIR = 'YOLO_dataset/images/test'
CONF_THRESH  = 0.25

def run_detection(model_path: str, test_dir: str, conf: float):
    # print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_path  = Path(test_dir)

    if not test_path.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    images = [f for f in test_path.iterdir() if f.suffix.lower() in image_exts]

    if not images:
        print(f"ERROR: No images found in {test_dir}")
        sys.exit(1)

    chosen = random.choice(images)
    print(f"Selected image: {chosen.name}")

    # print(f"cv2.imread shape:      {cv2.imread(str(chosen)).shape}")

    results = model(str(chosen), conf=conf, verbose=False)
    result  = results[0]

    # print(f"result.orig_img shape: {result.orig_img.shape}")
    # print(f"Box xyxy coordinates:\n{result.boxes.xyxy}")

    # img = result.orig_img.copy()

    # palette = [
    #     (68, 68, 255), (0, 136, 255), (0, 221, 255), (68, 255, 68),
    #     (255, 187, 0), (255, 68, 136), (187, 68, 255), (204, 255, 0),
    #     (0, 102, 255), (255, 68, 0), (0, 255, 170), (136, 0, 255)
    # ]

    if result.boxes is not None and len(result.boxes) > 0:
        # for box in result.boxes:
        #     x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        #     x1 *= 2
        #     x2 *= 2
        #     cls_id          = int(box.cls[0].item())
        #     conf_score      = float(box.conf[0].item())
        #     label           = model.names[cls_id]
        #     color           = palette[cls_id % len(palette)]
        #
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        #
        #     label_text = f"{label} {conf_score:.2f}"
        #     (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        #     cv2.putText(img, label_text, (x1 + 2, y1 - 4),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        print(f"\nDetections ({len(result.boxes)} total):")
        for box in result.boxes:
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            label      = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 *= 2
            x2 *= 2
            print(f"  {label:<30} conf={conf_score:.3f}  box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("No detections above confidence threshold.")

    # label_path = chosen.parent.parent.parent / 'labels' / chosen.parent.name / (chosen.stem + '.txt')
    # h, w = result.orig_img.shape[:2]
    # if label_path.exists():
    #     print(f"\nGround-truth labels ({label_path.name}):")
    #     with open(label_path) as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             if len(parts) != 5:
    #                 continue
    #             cls_id = int(parts[0])
    #             xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    #             x1 = (xc - bw / 2) * w * 2
    #             y1 = (yc - bh / 2) * h
    #             x2 = (xc + bw / 2) * w * 2
    #             y2 = (yc + bh / 2) * h
    #             lbl = model.names.get(cls_id, str(cls_id))
    #             print(f"  {lbl:<30} box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    # else:
    #     print(f"\nNo label file found at {label_path}")

    # window_title = f"{chosen.name}  |  {len(result.boxes) if result.boxes else 0} detection(s)  |  conf >= {conf}"
    #
    # def on_click(event, x, y, *_):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print(f"Clicked: x={x}, y={y}")
    #
    # cv2.imshow(window_title, img)
    # cv2.setMouseCallback(window_title, on_click)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection(MODEL_PATH, TEST_IMG_DIR, CONF_THRESH)
