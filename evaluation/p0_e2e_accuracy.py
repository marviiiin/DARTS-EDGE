#!/usr/bin/env python3
"""P0 Benchmark 3: End-to-end detection accuracy evaluation.

Runs the full pipeline (YOLO -> tracking -> trajectory -> CNN) on all
labeled test videos and computes per-class and overall confusion matrix,
precision, recall, F1.

Usage on Jetson:
    source venv/bin/activate
    python p0_e2e_accuracy.py
"""
import os
import sys
import time
import json
import statistics
import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.expanduser('~/darts_edge'))

MODELS_DIR = os.path.expanduser('~/darts_edge/models')
VIDEO_DIR = os.path.expanduser('~/darts_edge')
OUTPUT_PATH = os.path.expanduser('~/darts_edge/benchmark_results/e2e_accuracy.json')

# Videos with ground truth labels
TEST_VIDEOS = [
    ('left-Uncongested_conditions_videos1.MP4', 'normal', 'left'),
    ('05_31_22-left-200ft_AA_Free_IRX3-I275-Hillsborough.MP4', 'normal', 'left'),
    ('05_31_22-right-200ft_TA_Free_IRX-I275-Hillsborough.MP4', 'normal', 'right'),
    ('Congested_conditions_Videos.MP4', 'congestion', 'left'),
    ('right-Congested_conditions_Videos.MP4', 'congestion', 'right'),
    ('05_31_22-left-200ft_AA_Congested_IRX-I275-Hillsborough.MP4', 'congestion', 'left'),
    ('left-4_20_22_Road_Ranger_Incident_2_3.MP4', 'incident', 'left'),
    ('8_24_22_I_75_incidentDownstream1.MP4', 'incident', 'left'),
    ('right-8_24_22_I_75_incidentUpstream4.MP4', 'incident', 'right'),
    ('left-5_31_22_Road_Ranger_3.MP4', 'incident', 'left'),
]

CLASS_NAMES = ['normal', 'congestion', 'incident']
CLASS_MAP = {0: 'normal', 1: 'congestion', 2: 'incident'}


def process_video(detector, cnn_trt, cnn_model, video_path, direction):
    """Process full video and return list of per-window classifications."""
    from PIL import Image as PILImage, ImageOps
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv.CAP_PROP_FPS) or 30
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 512
    trajectory_len = int(fps * 3)
    slidingWindow_frame = int(fps)
    speed_background = 0.0

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    trajectories = []
    prev_gray = None
    frame_idx = 0
    fig_idx = 0
    classifications = []
    traj_images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # YOLO detection
        detections = detector.detect(frame)

        # Optical flow tracking
        if prev_gray is not None and trajectories:
            p0 = np.float32([t[-1] for t in trajectories]).reshape(-1, 1, 2)
            p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
            p0r, _, _ = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_t = []
            for tr, (x, y), ok in zip(trajectories, p1.reshape(-1, 2), good):
                if not ok:
                    continue
                tr.append((x, y))
                if len(tr) > trajectory_len:
                    del tr[0]
                new_t.append(tr)
            trajectories = new_t

        # Add new detections
        if detections:
            for (bx, by, bw, bh, _) in detections:
                cx, cy = bx + bw / 2, by + bh / 2
                if not any(abs(t[-1][0] - cx) < bw + 30 and abs(t[-1][1] - cy) < bh
                           for t in trajectories):
                    trajectories.append([(cx, cy)])

        # Trajectory rendering + classification
        startup_grace = int(fps * 10)
        if (frame_idx > max(trajectory_len, startup_grace) and
                (frame_idx - 1) % slidingWindow_frame == 0):

            # Render trajectory image
            selected = []
            for tr in trajectories:
                if len(tr) < 5:
                    continue
                xs = [p[0] - speed_background * i for i, p in enumerate(tr)]
                ys = [p[1] for p in tr]
                if len(xs) > 1:
                    selected.append([xs, ys])

            if len(selected) >= 2:
                buf = BytesIO()
                fig, ax = plt.subplots(figsize=(2.24, 2.24))
                ax.set_xlim([-speed_background * trajectory_len, w])
                ax.set_ylim([0, h])
                ax.invert_yaxis()
                ax.axis('off')
                for xy in selected:
                    if len(xy[0]) >= 2:
                        ax.plot(xy[0], xy[1], color='black', linewidth=1)
                fig.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                buf.seek(0)
                plot_img = PILImage.open(buf).convert('L')
                inverted = ImageOps.invert(plot_img)
                resized = inverted.resize((56, 56), PILImage.LANCZOS)
                traj_np = np.array(resized)
                traj_img = np.stack([traj_np, traj_np, traj_np], axis=-1)
                plot_img.close()
                inverted.close()
                resized.close()
                buf.close()

                traj_images.append(traj_img)
                fig_idx += 1

            # Classify every 10 trajectory images (matching production logic)
            if fig_idx > 0 and fig_idx % 10 == 0 and traj_images:
                # Batch classify
                batch = np.zeros((len(traj_images), 3, 56, 56), dtype=np.float32)
                for i, img in enumerate(traj_images):
                    batch[i] = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

                if cnn_trt is not None:
                    preds = cnn_trt.predict_batch(batch)
                elif cnn_model is not None:
                    tensor = torch.from_numpy(batch)
                    with torch.no_grad():
                        out = cnn_model(tensor)
                        preds = torch.max(out, 1)[1].cpu().numpy()
                else:
                    preds = np.zeros(len(traj_images), dtype=int)

                # Majority vote for this window
                counts = [0, 0, 0]
                for p in preds:
                    counts[int(p)] += 1

                # Determine label using same logic as production
                total = sum(counts)
                incident_ratio = counts[2] / total * 100 if total > 0 else 0
                congestion_ratio = counts[1] / total * 100 if total > 0 else 0

                if incident_ratio >= 30:
                    label = 'incident'
                elif congestion_ratio >= 50:
                    label = 'congestion'
                else:
                    label = 'normal'

                classifications.append({
                    'frame': frame_idx,
                    'label': label,
                    'counts': {'normal': counts[0], 'congestion': counts[1], 'incident': counts[2]},
                    'incident_ratio': round(incident_ratio, 1),
                    'congestion_ratio': round(congestion_ratio, 1),
                })
                traj_images = []

        prev_gray = frame_gray

    cap.release()
    return classifications


def compute_confusion_matrix(results):
    """Compute confusion matrix and metrics from per-video results."""
    # 3x3 confusion matrix: rows=ground truth, cols=predicted
    cm = np.zeros((3, 3), dtype=int)
    label_to_idx = {'normal': 0, 'congestion': 1, 'incident': 2}

    for r in results:
        gt = label_to_idx[r['ground_truth']]
        pred = label_to_idx[r['majority_prediction']]
        cm[gt][pred] += 1

    # Per-class metrics
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3)) - tp
        fn = sum(cm[i][j] for j in range(3)) - tp
        tn = cm.sum() - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }

    # Overall accuracy
    correct = sum(cm[i][i] for i in range(3))
    total = cm.sum()
    overall_accuracy = correct / total if total > 0 else 0

    # Macro-average F1
    macro_f1 = statistics.mean([metrics[n]['f1'] for n in CLASS_NAMES])

    return {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': CLASS_NAMES,
        'per_class': metrics,
        'overall_accuracy': round(overall_accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'n_videos': int(total),
    }


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load models
    print("Loading models...")
    from realtime_detector import TRTYoloDetector, TRTCNNClassifier, _cuda_ctx
    import torch

    engine_path = os.path.join(MODELS_DIR, 'yolov4-custom.engine')
    detector = TRTYoloDetector(engine_path, input_size=416)

    cnn_trt = None
    cnn_model = None
    cnn_engine = os.path.join(MODELS_DIR, 'cnn_classifier.engine')
    cnn_pth = os.path.join(MODELS_DIR, 'gray_trajLen_3_model.pth')
    if os.path.exists(cnn_engine):
        try:
            cnn_trt = TRTCNNClassifier(cnn_engine, _cuda_ctx)
            print("CNN: TensorRT FP16")
        except Exception as e:
            print(f"CNN TRT failed: {e}")
    if cnn_trt is None and os.path.exists(cnn_pth):
        cnn_model = torch.jit.load(cnn_pth, map_location='cpu')
        cnn_model.eval()
        print("CNN: PyTorch CPU")

    video_results = []
    for video_name, gt_label, direction in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"Skipping {video_name} (not found)")
            continue

        print(f"\nProcessing: {video_name} (GT: {gt_label})")
        classifications = process_video(detector, cnn_trt, cnn_model, video_path, direction)

        if not classifications:
            print(f"  No classifications produced")
            continue

        # Majority vote across all windows
        all_labels = [c['label'] for c in classifications]
        from collections import Counter
        label_counts = Counter(all_labels)
        majority = label_counts.most_common(1)[0][0]

        result = {
            'video': video_name,
            'ground_truth': gt_label,
            'direction': direction,
            'majority_prediction': majority,
            'correct': majority == gt_label,
            'n_windows': len(classifications),
            'window_labels': dict(label_counts),
            'all_classifications': classifications,
        }
        video_results.append(result)

        status = 'CORRECT' if result['correct'] else 'WRONG'
        print(f"  Prediction: {majority} ({status}) | Windows: {dict(label_counts)}")

    # Compute confusion matrix
    cm_results = compute_confusion_matrix(video_results)

    output = {
        'confusion_matrix': cm_results,
        'per_video': video_results,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved: {OUTPUT_PATH}")
    print(f"\nOverall Accuracy: {cm_results['overall_accuracy']*100:.1f}%")
    print(f"Macro F1: {cm_results['macro_f1']:.4f}")
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    print(f"           Normal  Congestion  Incident")
    for i, name in enumerate(CLASS_NAMES):
        row = cm_results['confusion_matrix'][i]
        print(f"  {name:10s}  {row[0]:5d}  {row[1]:10d}  {row[2]:8d}")
    print(f"\nPer-class:")
    for name in CLASS_NAMES:
        m = cm_results['per_class'][name]
        print(f"  {name:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # Cleanup
    try:
        if hasattr(detector, 'cuda_ctx'):
            detector.cuda_ctx.pop()
    except:
        pass
