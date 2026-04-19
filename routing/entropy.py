"""
entropy.py
Measures how spread out the confidence distribution is.
Uniform = uncertain (high entropy). Peaked = confident (low entropy).

Input : [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
Output: float (0~1) — higher = more confident = less likely to escalate
"""

import math
from typing import List, Dict


def entropy_score(predictions: List[Dict]) -> float:
    if not predictions:
        return 0.0

    scores = [p["confidence"] for p in predictions]

    # Normalize scores to sum to 1 (treat as probability distribution)
    total = sum(scores)
    if total == 0:
        return 0.0
    probs = [s / total for s in scores]

    # Compute entropy: H = -sum(p * log(p))
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)

    # Normalize by max entropy log(N) -> result in [0, 1]
    max_entropy = math.log(len(scores) + 1e-9)
    if max_entropy == 0:
        return 1.0

    normalized_entropy = entropy / max_entropy
    return round(1.0 - normalized_entropy, 4)
