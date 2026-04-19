"""
mean_score.py
Average confidence score across all predicted boxes.

Input : [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
Output: float (0~1) — higher = more confident = less likely to escalate
"""

from typing import List, Dict


def mean_score(predictions: List[Dict]) -> float:
    if not predictions:
        return 0.0
    return sum(p["confidence"] for p in predictions) / len(predictions)
