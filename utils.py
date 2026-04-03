
def compute_iou(pred_bbox, gt_bbox):
    """
    pred_bbox: [x1, y1, x2, y2]
    gt_bbox: [x, y, w, h]
    returns: IoU score
    """
    gt_x1 = gt_bbox[0]
    gt_y1 = gt_bbox[1]
    gt_x2 = gt_bbox[0] + gt_bbox[2]
    gt_y2 = gt_bbox[1] + gt_bbox[3]

    inter_x1 = max(pred_bbox[0], gt_x1)
    inter_y1 = max(pred_bbox[1], gt_y1)
    inter_x2 = min(pred_bbox[2], gt_x2)
    inter_y2 = min(pred_bbox[3], gt_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = gt_bbox[2] * gt_bbox[3]
    union_area = pred_area + gt_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0