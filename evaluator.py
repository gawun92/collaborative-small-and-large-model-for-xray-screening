"""
evaluator.py
Per-image AP50 calculation + EvalResult definition + confidence score histogram.

AP50 calculation:
    - Match predicted bboxes against GT bboxes per image
    - Compute AP via 11-point interpolation
    - Average AP across all images -> mean AP50
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EvalResult:
    pipeline_name: str
    total_time: float
    mean_ap50: float
    small_model_calls: int = 0
    large_model_calls: int = 0
    confidence_scores: List[float] = field(default_factory=list)


# IoU
def compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU box format: [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# Per-image AP50
def compute_image_ap50(
        pred: List[Dict],
        gt_list: List[dict],
        iou_threshold: float = 0.5,
) -> float:
    """
    pred   : [{"bbox": [x1,y1,x2,y2], "confidence": float, "predicted_class": int}, ...]
    gt_list: [{"category_id": int, "bbox": [x,y,w,h]}, ...]
    """
    if not gt_list or not pred:
        return 0.0

    n_gt = len(gt_list)

    # Sort predictions by confidence descending
    sorted_pred = sorted(pred, key=lambda p: p["confidence"], reverse=True)

    matched_gt = set()
    tp, fp = 0, 0
    precisions, recalls = [], []

    for p in sorted_pred:
        x1, y1, x2, y2 = p["bbox"]
        pred_box = [x1, y1, x2 - x1, y2 - y1]  # convert xyxy -> xywh for compute_iou
        pred_label = p["predicted_class"]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_list):
            if gt_idx in matched_gt:
                continue
            if gt["category_id"] != pred_label:
                continue
            iou = compute_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)

    # 11-point interpolation
    ap = 0.0
    for r_thresh in [t / 10 for t in range(11)]:
        p_vals = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
        ap += max(p_vals) if p_vals else 0.0
    ap /= 11

    return round(ap, 4)


# Confidence histogram
def print_confidence_histogram(confidence_scores: List[float], n_bins: int = 5) -> None:
    if not confidence_scores:
        return

    total = len(confidence_scores)
    bin_size = 1.0 / n_bins
    bins = [0] * n_bins

    for score in confidence_scores:
        idx = min(int(score / bin_size), n_bins - 1)
        bins[idx] += 1

    print("\n  [Confidence Score Histogram]")
    for i, count in enumerate(bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        bar = "█" * int(count / total * 30)
        print(f"  {lo:.1f}-{hi:.1f} | {bar:<30} {count:>6} ({count / total * 100:.1f}%)")
