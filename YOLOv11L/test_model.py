# ============================================================
# YOLO Detection Visualizer
# Picks a random image from test set, runs inference,
# and displays bounding boxes + labels overlaid on the image
# ============================================================

import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ============================================================
# CONFIGURATION — UPDATE THESE
# ============================================================

MODEL_PATH   = 'checkpoints/yolov11l/best.pt'   # <-- path to your .pt file
TEST_IMG_DIR = 'YOLO_dataset/images/test'        # <-- path to test images folder
CONF_THRESH  = 0.25                              # <-- minimum confidence to show a box

# ============================================================

def run_detection(model_path: str, test_dir: str, conf: float):

    # ── Load model ───────────────────────────────────────────
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ── Pick a random image ──────────────────────────────────
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

    # ── Run inference ────────────────────────────────────────
    results = model(str(chosen), conf=conf, verbose=False)
    result  = results[0]

    # ── Use the image Ultralytics already processed ──────────
    # result.orig_img is the original image with correct dimensions
    # This ensures bounding box coordinates are correctly mapped
    # back to the original image space rather than the 640x640 input space
    img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    img_rgb = cv2.resize(img_rgb, (w//2, h))

    # ── Set up plot ──────────────────────────────────────────
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img_rgb)
    ax.axis('off')
    ax.set_title(f"{chosen.name}  |  {len(result.boxes)} detection(s)  |  conf ≥ {conf}",
                 fontsize=12, pad=10)

    # Color palette — one color per class
    palette = [
        '#FF4444', '#FF8800', '#FFDD00', '#44FF44', '#00BBFF',
        '#8844FF', '#FF44BB', '#00FFCC', '#FF6600', '#0044FF',
        '#AAFF00', '#FF0088'
    ]

    # ── Draw boxes ───────────────────────────────────────────
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # Box coordinates (xyxy format, pixel space)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id          = int(box.cls[0].item())
            conf_score      = float(box.conf[0].item())
            label           = model.names[cls_id]
            color           = palette[cls_id % len(palette)]

            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # Draw label background + text
            label_text = f"{label} {conf_score:.2f}"
            ax.text(
                x1, y1 - 6,
                label_text,
                fontsize=9,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85, edgecolor='none')
            )

        print(f"\nDetections ({len(result.boxes)} total):")
        for box in result.boxes:
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            label      = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {label:<30} conf={conf_score:.3f}  box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("No detections above confidence threshold.")
        ax.text(
            w / 2, h / 2,
            f"No detections (conf ≥ {conf})",
            fontsize=14, color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6)
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_detection(MODEL_PATH, TEST_IMG_DIR, CONF_THRESH)