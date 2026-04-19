# ============================================================
# YOLOv11l Training Script for LDXray Dataset
# Large Model — Main View X-ray Prohibited Item Detection
# ============================================================
# INSTRUCTIONS FOR LOCAL PC SETUP:
# 1. Install dependency:  pip install ultralytics
# 2. Update DATASET_PATH below to your local dataset location
# 3. Make sure labels (.txt files) are in the same folder as images
# 4. Run: python train_yolov11l.py
# ============================================================

from ultralytics import YOLO
import os
import shutil
import threading
import time

# ============================================================
# CONFIGURATION — UPDATE THESE FOR YOUR LOCAL PC
# ============================================================

# Path to the LDXray dataset folder containing train_A and test_A
# Windows example: 'C:/Users/YourName/Desktop/LDXray/archive/dataset'
# Mac/Linux example: '/home/username/LDXray/archive/dataset'
DATASET_PATH = 'YOLO_dataset'   # <-- CHANGE THIS

# Folder where checkpoints will be saved locally
# This folder will be created automatically if it doesn't exist
WEIGHTS_DIR = 'checkpoints/yolov11l'          # <-- CHANGE THIS if needed

# Name for this training run
# Results will be saved to runs/detect/MODEL_NAME/
MODEL_NAME = 'ldxray_large_model'             # <-- CHANGE THIS if needed

# Batch size — adjust based on your GPU VRAM
# 4GB  VRAM → use 4
# 8GB  VRAM → use 8
# 16GB VRAM → use 16
# 24GB VRAM → use 32
BATCH_SIZE = 8                               # <-- CHANGE THIS based on your GPU

# Device to train on
# 0    → first GPU (recommended if you have NVIDIA GPU)
# 'cpu' → CPU only (very slow, not recommended)
DEVICE = 0                                    # <-- CHANGE THIS if needed

# Number of workers for data loading
# Usually set to number of CPU cores available
# Reduce to 4 if you get memory errors
WORKERS = 8                                   # <-- CHANGE THIS if needed

# ============================================================
# DO NOT CHANGE ANYTHING BELOW THIS LINE
# ============================================================

# Create checkpoint folder if it doesn't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ── STEP 1: SPOT CHECK DATASET ──────────────────────────────
# Checks that both images and label files exist before training
# If labels are missing training will fail immediately
print("\n" + "="*50)
print("STEP 1: Checking dataset...")
print("="*50)

train_img_dir = os.path.join(DATASET_PATH, 'images/train')
train_lbl_dir = os.path.join(DATASET_PATH, 'labels/train')

# Spot check first 5 images and their label files
all_good = True
for s in ['000000', '000001', '000002', '000003', '000004']:
    img_exists   = os.path.exists(f'{train_img_dir}/{s}.jpg')
    label_exists = os.path.exists(f'{train_lbl_dir}/{s}.txt')
    status = 'OK' if (img_exists and label_exists) else 'PROBLEM'
    print(f"  [{status}] {s}.jpg: {'YES' if img_exists else 'NO'} | {s}.txt: {'YES' if label_exists else 'NO'}")
    if not label_exists:
        all_good = False

if not all_good:
    print("\nERROR: Some label files are missing!")
    print("Make sure .txt label files are in the same folder as .jpg images")
    exit()

print("\nDataset check passed! Safe to train.")

# ── STEP 2: CREATE YAML CONFIG FILE ─────────────────────────
# YAML file tells YOLO where the data is and what the classes are
# This is created automatically from DATASET_PATH above
print("\n" + "="*50)
print("STEP 2: Creating dataset config...")
print("="*50)

yaml_content = f"""
# LDXray Dataset Configuration
# Auto-generated — do not edit manually

# Root path of the dataset
path: {DATASET_PATH}

# Training and validation image folders (relative to path above)
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 12

# Class names in order (must match label file class indices)
names:
  0: Mobile_Phone
  1: Orange_Liquid
  2: Portable_Charger_1
  3: Laptop
  4: Green_Liquid
  5: Portable_Charger_2
  6: Tablet
  7: Blue_Liquid
  8: Columnar_Orange_Liquid
  9: Nonmetallic_Lighter
  10: Umbrella
  11: Columnar_Green_Liquid
"""

yaml_path = 'ldxray.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print(f"yaml created at: {yaml_path}")

# ── STEP 3: START AUTO BACKUP THREAD ────────────────────────
# Runs in background every 10 minutes
# Copies the latest checkpoint to WEIGHTS_DIR so progress is never lost
# Even if training crashes you can resume from the last backup
print("\n" + "="*50)
print("STEP 3: Starting auto-backup...")
print("="*50)

def auto_backup():
    """
    Background thread that saves checkpoint to WEIGHTS_DIR every 10 minutes.
    This protects against crashes, power cuts, or accidental interruptions.
    """
    while True:
        time.sleep(600)   # wait 10 minutes
        src = f'runs/detect/{MODEL_NAME}/weights/last.pt'
        dst = os.path.join(WEIGHTS_DIR, 'last.pt')
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"\n[AUTO-BACKUP] Checkpoint saved to {dst}")



if __name__ == "__main__":
    # daemon=True means this thread stops automatically when training ends
    backup_thread = threading.Thread(target=auto_backup, daemon=True)
    backup_thread.start()
    print("Auto-backup started! Saves every 10 mins to:", WEIGHTS_DIR)

    # ── STEP 4: TRAIN OR RESUME ──────────────────────────────────
    # Checks if a saved checkpoint exists in WEIGHTS_DIR
    # If yes  → resumes from where it left off (no progress lost)
    # If no   → starts fresh training from pretrained YOLOv11l weights
    print("\n" + "="*50)
    print("STEP 4: Starting training...")
    print("="*50)

    # Paths for checkpoint files
    local_last = f'runs/detect/{MODEL_NAME}/weights/last.pt'   # where YOLO saves it
    saved_last = os.path.join(WEIGHTS_DIR, 'last.pt')          # our backup copy

    if os.path.exists(saved_last):
        # Resume from saved checkpoint
        print(f"Checkpoint found at: {saved_last}")
        print("RESUMING from last checkpoint — no progress will be lost!")

        # Copy backup checkpoint to where YOLO expects it
        os.makedirs(f'runs/detect/{MODEL_NAME}/weights', exist_ok=True)
        shutil.copy(saved_last, local_last)

        # Load checkpoint and resume
        model   = YOLO(local_last)
        results = model.train(resume=True)

    else:
        # Fresh training from pretrained weights
        print("No checkpoint found — Starting fresh training!")
        print("Downloading YOLOv11l pretrained weights from COCO...")

        # yolov11l.pt will be downloaded automatically on first run
        # It is pretrained on COCO (80 categories) and fine-tuned on LDXray
        model = YOLO('yolo11l.pt')

        results = model.train(
            data        = yaml_path,    # dataset config file created above
            epochs      = 100,          # total training epochs
            imgsz       = 640,          # input image size (pixels)
            batch       = BATCH_SIZE,   # number of images per batch
            name        = MODEL_NAME,   # name of this training run
            patience    = 20,           # stop early if no improvement for 20 epochs
            save        = True,         # save checkpoints
            save_period = 5,            # save extra checkpoint every 5 epochs
            plots       = True,         # generate training plots
            device      = DEVICE,       # GPU or CPU
            workers     = WORKERS,      # data loading workers
            amp = True,
        )

    # ── STEP 5: SAVE FINAL WEIGHTS ───────────────────────────────
    # Copies best and last weights to WEIGHTS_DIR after training finishes
    # best.pt  → highest mAP checkpoint (use this for inference)
    # last.pt  → final epoch checkpoint (use this to resume if needed)
    print("\n" + "="*50)
    print("STEP 5: Saving final weights...")
    print("="*50)

    for fname in ['best.pt', 'last.pt']:
        src = f'runs/detect/{MODEL_NAME}/weights/{fname}'
        dst = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Saved: {dst}")
        else:
            print(f"Warning: {fname} not found — training may not have completed")

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best weights saved to: {WEIGHTS_DIR}/best.pt")
    print(f"Use best.pt for inference and evaluation")