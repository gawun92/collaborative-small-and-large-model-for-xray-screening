"""
max_score.py
Use the highest confidence score among all predicted boxes as the routing signal.

Input : [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
Output: float (0~1) — higher = more confident = less likely to escalate
"""

from typing import List, Dict


def max_score(predictions: List[Dict]) -> float:
    if not predictions:
        return 0.0
    return max(p["confidence"] for p in predictions)
