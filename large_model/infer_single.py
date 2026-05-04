import argparse
import json
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_iou
from mmdet.apis import init_detector, inference_detector

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main-view-atss.py')


def run_atss_inference(trained_weight, image_path, gt_bboxes):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_detector(CONFIG, trained_weight, device=device)
    result = inference_detector(model, image_path)

    pred = result.pred_instances
    pred_bboxes = pred.bboxes.cpu().numpy()
    pred_labels = pred.labels.cpu().numpy()
    pred_scores = pred.scores.cpu().numpy()

    correct = 0
    total_gt = len(gt_bboxes)

    for gt in gt_bboxes:
        gt_bbox = gt['bbox']
        gt_label = gt['category_id']
        matched = False
        for pred_bbox, pred_label, pred_score in zip(pred_bboxes, pred_labels, pred_scores):
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou >= 0.5 and (pred_label + 1) == gt_label:
                matched = True
                break
        if matched:
            correct += 1

    accuracy_at_50 = correct / total_gt if total_gt > 0 else 0.0
    return accuracy_at_50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trained_weight', help='trained weight file path')
    parser.add_argument('image', help='image file path')
    parser.add_argument('gt_bboxes', help='ground truth bboxes as JSON string')
    args = parser.parse_args()

    gt_bboxes = json.loads(args.gt_bboxes)
    result = run_atss_inference(args.trained_weight, args.image, gt_bboxes)
    print(f"Accuracy@0.5: {result:.4f}")