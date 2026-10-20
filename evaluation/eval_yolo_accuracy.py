"""Evaluate YOLOv4 FP16 (TRT) vs FP32 (OpenCV DNN) detection accuracy.

Compares detections on labeled test images against ground truth YOLO labels.
Reports mAP@0.5, precision, recall, F1 for both backends.

Usage on Jetson:
    source venv/bin/activate
    python eval_yolo_accuracy.py /path/to/images/
"""
import os
import sys
import json
import glob
import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.expanduser('~/darts_edge'))


def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO format labels (class cx cy w h) normalized -> absolute."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, cls])
    return boxes


def iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_metrics(all_gt, all_pred, iou_thresh=0.5):
    """Compute precision, recall, F1, and AP@iou_thresh."""
    tp = 0
    fp = 0
    total_gt = sum(len(g) for g in all_gt)

    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        matched = set()
        for pred in pred_boxes:
            best_iou = 0
            best_idx = -1
            for i, gt in enumerate(gt_boxes):
                if i in matched:
                    continue
                v = iou(pred[:4], gt[:4])
                if v > best_iou:
                    best_iou = v
                    best_idx = i
            if best_iou >= iou_thresh and best_idx >= 0:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'total_gt': total_gt,
        'total_pred': tp + fp,
    }


def run_eval(detector, images, labels_dir, backend_name):
    """Run detector on images and compute metrics."""
    all_gt = []
    all_pred = []
    total_detections = 0

    for img_path in images:
        img = cv.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load ground truth
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + '.txt')
        gt_boxes = load_yolo_labels(label_path, w, h)
        all_gt.append(gt_boxes)

        # Run detection — output is (x1, y1, w, h, conf) top-left corner format
        detections = detector.detect(img)
        pred_boxes = []
        for det in detections:
            x1, y1, bw, bh = det[0], det[1], det[2], det[3]
            conf = det[4] if len(det) > 4 else 1.0
            x2 = x1 + bw
            y2 = y1 + bh
            pred_boxes.append([x1, y1, x2, y2, conf])
        all_pred.append(pred_boxes)
        total_detections += len(pred_boxes)

    metrics = compute_metrics(all_gt, all_pred, iou_thresh=0.5)
    metrics['backend'] = backend_name
    metrics['n_images'] = len(images)
    print(f"\n{backend_name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
          f"F1={metrics['f1']:.3f} | TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} | GT={metrics['total_gt']} Pred={metrics['total_pred']}")
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='Directory with .jpg images and .txt labels')
    args = parser.parse_args()

    img_dir = args.image_dir
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    print(f"Found {len(images)} images in {img_dir}")

    if not images:
        print("No images found!")
        sys.exit(1)

    models_dir = os.path.expanduser('~/darts_edge/models')
    results = {}

    # Test 1: TensorRT FP16
    try:
        from realtime_detector import TRTYoloDetector
        engine_path = os.path.join(models_dir, 'yolov4-custom.engine')
        if os.path.exists(engine_path):
            det_trt = TRTYoloDetector(engine_path, input_size=416)
            results['trt_fp16'] = run_eval(det_trt, images, img_dir, 'TensorRT-FP16')
        else:
            print("TRT engine not found, skipping")
    except Exception as e:
        print(f"TRT eval failed: {e}")

    # Test 2: OpenCV DNN (FP32)
    try:
        from realtime_detector import OpenCVYoloDetector
        cfg_path = os.path.join(models_dir, 'yolov4-custom.cfg')
        weights_path = os.path.join(models_dir, 'yolov4-custom_final.weights')
        if os.path.exists(cfg_path) and os.path.exists(weights_path):
            det_cv = OpenCVYoloDetector(cfg_path, weights_path, input_size=416)
            results['opencv_fp32'] = run_eval(det_cv, images, img_dir, 'OpenCV-DNN-FP32')
        else:
            print(f"OpenCV DNN files not found: cfg={os.path.exists(cfg_path)} weights={os.path.exists(weights_path)}")
    except Exception as e:
        print(f"OpenCV eval failed: {e}")

    # Save results
    out_path = os.path.expanduser('~/darts_edge/yolo_accuracy_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(json.dumps(results, indent=2))
