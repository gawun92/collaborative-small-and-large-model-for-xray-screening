"""
dummy_models.py
Small model  : NanoDet-Plus  (small_model/small_model_inference_single.py -> run_inference)
Large model  : YOLOv11L      (YOLOv11L/test_model.py -> run_detection)

Return type for both:
    [{"bbox": [x1, y1, x2, y2], "confidence": float, "predicted_class": int}, ...]
"""

import os
import sys
from enum import IntEnum
from typing import List, Dict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SMALL_MODEL_DIR = os.path.join(PROJECT_DIR, "small_model")
LARGE_MODEL_DIR = os.path.join(PROJECT_DIR, "YOLOv11L")

SMALL_CONFIG = os.path.join(SMALL_MODEL_DIR, "ldxray_mainview_nanodet_plus_m_416_test.yml")
SMALL_CKPT = os.path.join(SMALL_MODEL_DIR, "model_best.pth")
LARGE_CKPT = os.path.join(LARGE_MODEL_DIR, "best.pt")

sys.path.insert(0, SMALL_MODEL_DIR)
sys.path.insert(0, LARGE_MODEL_DIR)

from small_model_inference_single import load_model, run_inference
from test_model import run_detection


class Category(IntEnum):
    MOBILE_PHONE = 1
    ORANGE_LIQUID = 2
    POWER_BANK_WITHOUT_BATT = 3
    LAPTOP = 4
    GREEN_LIQUID = 5
    POWER_BANK_WITH_BATT = 6
    TABLET = 7
    BLUE_LIQUID = 8
    CYLINDRICAL_ORANGE_LIQUID = 9
    NON_METAL_LIGHTER = 10
    UMBRELLA = 11
    CYLINDRICAL_GREEN_LIQUID = 12


# Model singletons (loaded once, reused across calls)
_small_model = None
_small_cfg = None


def _load_small_model():
    global _small_model, _small_cfg
    if _small_model is None:
        _small_cfg, _small_model = load_model(SMALL_CONFIG, SMALL_CKPT)


# Small model inference (NanoDet-Plus)
def run_small_model(image_path: str) -> List[Dict]:
    """
    Run NanoDet-Plus on a single image.
    Returns: [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
    """
    _load_small_model()
    return run_inference(_small_model, _small_cfg, image_path)


# Large model inference (YOLOv11L)
def run_large_model(image_path: str) -> List[Dict]:
    """
    Run YOLOv11L on a single image.
    Returns: [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
    """
    return run_detection(model_path=LARGE_CKPT, image_path=image_path, conf=0.25)
