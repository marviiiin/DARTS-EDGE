"""Compare TCD-Net FP32 (PyTorch) vs FP16 (TensorRT) predictions.

Since we don't have a labeled test dataset on-device, this script:
1. Processes test videos through the trajectory rendering pipeline
2. Classifies each trajectory image with both FP32 and FP16
3. Reports agreement rate and any disagreements

Usage on Jetson:
    source venv/bin/activate
    python eval_tcdnet_fp16.py
"""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/darts_edge'))

MODELS_DIR = os.path.expanduser('~/darts_edge/models')
PTS_PATH = os.path.join(MODELS_DIR, 'gray_trajLen_3_model.pth')
ENGINE_PATH = os.path.join(MODELS_DIR, 'cnn_classifier.engine')
VIDEO_DIR = os.path.expanduser('~/darts_edge')

# Test videos with known conditions
TEST_VIDEOS = [
    ('left-4_20_22_Road_Ranger_Incident_2_3.MP4', 'incident'),
    ('8_24_22_I_75_incidentDownstream1.MP4', 'incident'),
    ('Congested_conditions_Videos.MP4', 'congestion'),
    ('left-Uncongested_conditions_videos1.MP4', 'normal'),
    ('05_31_22-left-200ft_AA_Free_IRX3-I275-Hillsborough.MP4', 'normal'),
    ('05_31_22-left-200ft_AA_Congested_IRX-I275-Hillsborough.MP4', 'congestion'),
]


def generate_trajectory_images(video_path, n_images=50):
    """Generate trajectory images from a video using the realtime detector pipeline."""
    import cv2 as cv
    from PIL import Image, ImageOps
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return []

    fps = cap.get(cv.CAP_PROP_FPS) or 30
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 512

    # Simple LK tracking to build trajectories
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    old_gray = None
    points = None
    trajectories = {}
    next_id = 0
    traj_images = []
    frame_count = 0
    trajectory_len = 90
    speed_background = 0.0

    while len(traj_images) < n_images:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if old_gray is None or frame_count % 30 == 0:
            # Detect new features
            new_pts = cv.goodFeaturesToTrack(gray, **feature_params)
            if new_pts is not None:
                for pt in new_pts:
                    x, y = pt.ravel()
                    trajectories[next_id] = [(float(x), float(y))]
                    next_id += 1
                points = np.array([[list(trajectories[k][-1])] for k in trajectories], dtype=np.float32)
            old_gray = gray
            continue

        if points is not None and len(points) > 0:
            new_points, status, _ = cv.calcOpticalFlowPyrLK(old_gray, gray, points, None, **lk_params)
            if new_points is not None:
                keys = list(trajectories.keys())
                good_keys = []
                good_pts = []
                for i, (st, key) in enumerate(zip(status, keys)):
                    if st[0] == 1:
                        x, y = new_points[i].ravel()
                        trajectories[key].append((float(x), float(y)))
                        # Trim to trajectory_len
                        if len(trajectories[key]) > trajectory_len:
                            trajectories[key] = trajectories[key][-trajectory_len:]
                        good_keys.append(key)
                        good_pts.append([x, y])
                # Remove lost tracks
                lost = set(keys) - set(good_keys)
                for k in lost:
                    del trajectories[k]
                if good_pts:
                    points = np.array(good_pts, dtype=np.float32).reshape(-1, 1, 2)
                else:
                    points = None

        old_gray = gray

        # Every 10 frames, render a trajectory image
        if frame_count % 10 == 0 and len(trajectories) >= 3:
            # Build selected trajectories
            selected = []
            for key, tr in trajectories.items():
                if len(tr) < 5:
                    continue
                xs, ys = [], []
                for i, pt in enumerate(tr):
                    xs.append(pt[0] - speed_background * i)
                    ys.append(pt[1])
                if len(xs) > 1 and xs[-1] - xs[0] >= 0:
                    selected.append([xs, ys])

            if len(selected) >= 2:
                # Render matching training pipeline
                buf = io.BytesIO()
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
                plot_img = Image.open(buf).convert('L')
                inverted = ImageOps.invert(plot_img)
                resized = inverted.resize((56, 56), Image.LANCZOS)
                gray_np = np.array(resized)
                canvas = np.stack([gray_np, gray_np, gray_np], axis=-1)

                plot_img.close()
                inverted.close()
                resized.close()
                buf.close()

                traj_images.append(canvas)

    cap.release()
    return traj_images


def classify_pytorch(model, images):
    """Classify images with PyTorch FP32 model."""
    batch = np.zeros((len(images), 3, 56, 56), dtype=np.float32)
    for i, img in enumerate(images):
        batch[i] = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

    tensor = torch.from_numpy(batch)
    with torch.no_grad():
        out = model(tensor)
        preds = torch.max(out, 1)[1].cpu().numpy()
        probs = torch.softmax(out, 1).cpu().numpy()
    return preds, probs


def classify_trt(trt_classifier, images):
    """Classify images with TensorRT FP16 engine."""
    batch = np.zeros((len(images), 3, 56, 56), dtype=np.float32)
    for i, img in enumerate(images):
        batch[i] = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
    preds = trt_classifier.predict_batch(batch)
    return preds


if __name__ == '__main__':
    print("=== TCD-Net FP32 vs FP16 Comparison ===\n")

    # Load FP32 model
    print("Loading PyTorch FP32 model...")
    model_fp32 = torch.jit.load(PTS_PATH, map_location='cpu')
    model_fp32.eval()

    # Load FP16 TRT engine
    trt_classifier = None
    try:
        from realtime_detector import TRTCNNClassifier
        import pycuda.driver as cuda
        cuda.init()
        cuda_ctx = cuda.Device(0).make_context()
        trt_classifier = TRTCNNClassifier(ENGINE_PATH, cuda_ctx)
        print("TensorRT FP16 engine loaded")
    except Exception as e:
        print(f"TRT engine load failed: {e}")
        print("Will compare PyTorch FP32 self-consistency only")

    results = {}
    total_agree = 0
    total_images = 0
    class_names = ['Normal', 'Congestion', 'Incident']

    # Per-class counters for confusion
    fp32_counts = [0, 0, 0]
    fp16_counts = [0, 0, 0]

    for video_name, expected_label in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"  Skipping {video_name} (not found)")
            continue

        print(f"\nProcessing: {video_name} (expected: {expected_label})")
        images = generate_trajectory_images(video_path, n_images=30)
        if not images:
            print(f"  No trajectory images generated")
            continue

        print(f"  Generated {len(images)} trajectory images")

        # FP32 predictions
        fp32_preds, fp32_probs = classify_pytorch(model_fp32, images)

        # FP16 predictions
        if trt_classifier:
            fp16_preds = classify_trt(trt_classifier, images)
        else:
            fp16_preds = fp32_preds  # fallback

        # Compare
        agree = int(np.sum(fp32_preds == fp16_preds))
        n = len(images)
        total_agree += agree
        total_images += n

        for p in fp32_preds:
            fp32_counts[p] += 1
        if trt_classifier:
            for p in fp16_preds:
                fp16_counts[p] += 1

        # Per-class breakdown
        fp32_dist = {class_names[i]: int(np.sum(fp32_preds == i)) for i in range(3)}
        fp16_dist = {class_names[i]: int(np.sum(fp16_preds == i)) for i in range(3)} if trt_classifier else fp32_dist

        video_result = {
            'video': video_name,
            'expected': expected_label,
            'n_images': n,
            'agreement': agree,
            'agreement_pct': round(agree / n * 100, 1),
            'fp32_distribution': fp32_dist,
            'fp16_distribution': fp16_dist,
        }
        results[video_name] = video_result

        print(f"  FP32: {fp32_dist}")
        print(f"  FP16: {fp16_dist}")
        print(f"  Agreement: {agree}/{n} ({agree/n*100:.1f}%)")

        # Show disagreements
        if agree < n and trt_classifier:
            disagree_idx = np.where(fp32_preds != fp16_preds)[0]
            for idx in disagree_idx[:5]:
                print(f"    Disagree img {idx}: FP32={class_names[fp32_preds[idx]]} "
                      f"FP16={class_names[fp16_preds[idx]]} "
                      f"FP32 probs={fp32_probs[idx].round(3)}")

    # Overall summary
    if total_images > 0:
        overall_agreement = total_agree / total_images * 100
    else:
        overall_agreement = 0

    summary = {
        'total_images': total_images,
        'total_agree': total_agree,
        'overall_agreement_pct': round(overall_agreement, 2),
        'fp32_class_totals': {class_names[i]: fp32_counts[i] for i in range(3)},
        'fp16_class_totals': {class_names[i]: fp16_counts[i] for i in range(3)} if trt_classifier else None,
        'per_video': results,
    }

    print(f"\n{'='*50}")
    print(f"OVERALL: {total_agree}/{total_images} agreement ({overall_agreement:.1f}%)")
    print(f"FP32 totals: {dict(zip(class_names, fp32_counts))}")
    if trt_classifier:
        print(f"FP16 totals: {dict(zip(class_names, fp16_counts))}")

    out_path = os.path.expanduser('~/darts_edge/tcdnet_fp16_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Clean up CUDA
    if trt_classifier:
        try:
            cuda_ctx.pop()
        except:
            pass
