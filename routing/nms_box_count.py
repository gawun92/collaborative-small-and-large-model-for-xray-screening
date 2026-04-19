"""
nms_box_count.py
Apply NMS on predictions, then use the remaining box count as a proxy for confidence.
Fewer boxes after NMS = more confident.

Input : [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
Output: float (0~1) — higher = more confident = less likely to escalate
"""

import math
from typing import List, Dict


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU calculation. box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _nms(predictions: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Sort by confidence descending, then remove overlapping boxes based on IoU threshold.
    Comparison is done within the same predicted_class only (class-aware NMS).
    """
    if not predictions:
        return []

    # Sort by confidence in descending order
    sorted_preds = sorted(predictions, key=lambda p: p["confidence"], reverse=True)

    kept = []
    while sorted_preds:
        current = sorted_preds.pop(0)
        kept.append(current)

        sorted_preds = [
            p for p in sorted_preds
            if p["predicted_class"] != current["predicted_class"]
               or _compute_iou(current["bbox"], p["bbox"]) < iou_threshold
        ]

    return kept


# Estimate confidence based on the number of boxes remaining after NMS.
# Fewer boxes = more confident prediction.
# Note: this is an indirect signal — box count also depends on the number of actual objects in the image.
def nms_box_count_score(predictions: List[Dict], iou_threshold: float = 0.5) -> float:
    if not predictions:
        return 1.0  # No boxes = nothing to detect = confident

    after_nms = _nms(predictions, iou_threshold)
    n = len(after_nms)

    # Exponential decay: confidence drops as box count increases
    # 1 box -> 1.0, 3 boxes -> ~0.74, 10 boxes -> ~0.14
    return round(math.exp(-0.3 * (n - 1)), 4)
