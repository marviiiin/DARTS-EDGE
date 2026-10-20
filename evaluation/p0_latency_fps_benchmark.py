#!/usr/bin/env python3
"""P0 Benchmark 1: Per-component latency & end-to-end FPS benchmark.

Measures timing for each pipeline stage:
  1. Frame decode (cv.read)
  2. YOLOv4 TRT inference
  3. Optical flow tracking
  4. Trajectory rendering
  5. TCD-Net CNN classification
  6. End-to-end FPS

Runs on all test videos and outputs JSON results.

Usage on Jetson:
    source venv/bin/activate
    python p0_latency_fps_benchmark.py
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
OUTPUT_PATH = os.path.expanduser('~/darts_edge/benchmark_results/latency_fps_benchmark.json')

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

MAX_FRAMES = 500  # frames per video for benchmarking


def benchmark_video(detector, cnn_trt, cnn_model, video_path, direction, max_frames=MAX_FRAMES):
    """Benchmark all pipeline stages on a single video."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None

    fps_video = cap.get(cv.CAP_PROP_FPS) or 30
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 512
    trajectory_len = int(fps_video * 3)

    # Timing accumulators
    t_decode = []
    t_yolo = []
    t_tracking = []
    t_render = []
    t_cnn = []
    t_total = []

    # Tracking state
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    trajectories = []
    prev_gray = None
    frame_idx = 0

    # Warm-up: 30 frames
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            _ = detector.detect(frame)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    trajectories = []

    while frame_idx < max_frames:
        t_start = time.perf_counter()

        # 1. Frame decode
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t_decode.append(time.perf_counter() - t0)

        frame_idx += 1
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 2. YOLOv4 detection
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t_yolo.append(time.perf_counter() - t0)

        # 3. Optical flow tracking
        t0 = time.perf_counter()
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
        t_tracking.append(time.perf_counter() - t0)

        # 4. Trajectory rendering (every 30 frames after warm-up)
        if frame_idx > trajectory_len and frame_idx % 30 == 0 and len(trajectories) >= 2:
            t0 = time.perf_counter()
            traj_img = _render_trajectory(trajectories, w, h, trajectory_len, direction)
            t_render.append(time.perf_counter() - t0)

            # 5. CNN classification
            if traj_img is not None:
                t0 = time.perf_counter()
                _classify_single(cnn_trt, cnn_model, traj_img)
                t_cnn.append(time.perf_counter() - t0)

        prev_gray = frame_gray
        t_total.append(time.perf_counter() - t_start)

    cap.release()

    def stats(arr):
        if not arr:
            return None
        return {
            'mean_ms': round(statistics.mean(arr) * 1000, 3),
            'median_ms': round(statistics.median(arr) * 1000, 3),
            'min_ms': round(min(arr) * 1000, 3),
            'max_ms': round(max(arr) * 1000, 3),
            'std_ms': round(statistics.stdev(arr) * 1000, 3) if len(arr) > 1 else 0,
            'p95_ms': round(sorted(arr)[int(len(arr) * 0.95)] * 1000, 3) if len(arr) > 1 else round(arr[0] * 1000, 3),
            'n_samples': len(arr),
        }

    end_to_end_fps = frame_idx / sum(t_total) if t_total else 0

    return {
        'frames_processed': frame_idx,
        'video_fps': fps_video,
        'resolution': f'{w}x{h}',
        'end_to_end_fps': round(end_to_end_fps, 1),
        'frame_decode': stats(t_decode),
        'yolo_inference': stats(t_yolo),
        'optical_flow_tracking': stats(t_tracking),
        'trajectory_rendering': stats(t_render),
        'cnn_classification': stats(t_cnn),
        'total_per_frame': stats(t_total),
    }


def _render_trajectory(trajectories, video_width, video_height, trajectory_len, direction):
    """Render trajectory image matching training pipeline."""
    from PIL import Image as PILImage, ImageOps
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    speed_background = 0.0
    selected = []
    for tr in trajectories:
        if len(tr) < 5:
            continue
        xs = [p[0] - speed_background * i for i, p in enumerate(tr)]
        ys = [p[1] for p in tr]
        if len(xs) > 1:
            selected.append([xs, ys])

    if len(selected) < 2:
        return None

    xlim = [-trajectory_len * speed_background, video_width]
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    ax.set_xlim(xlim)
    ax.set_ylim([0, video_height])
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
    result = np.array(resized)
    plot_img.close()
    inverted.close()
    resized.close()
    buf.close()
    return np.stack([result, result, result], axis=-1)


def _classify_single(cnn_trt, cnn_model, traj_img):
    """Classify a single trajectory image."""
    import torch
    img = traj_img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    batch = np.expand_dims(img, 0).astype(np.float32)

    if cnn_trt is not None:
        return int(cnn_trt.predict_batch(batch)[0])
    elif cnn_model is not None:
        tensor = torch.from_numpy(batch)
        with torch.no_grad():
            out = cnn_model(tensor)
            return int(torch.max(out, 1)[1].item())
    return -1


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load detector
    print("Loading YOLOv4 TRT engine...")
    from realtime_detector import TRTYoloDetector, TRTCNNClassifier
    import pycuda.driver as cuda

    engine_path = os.path.join(MODELS_DIR, 'yolov4-custom.engine')
    detector = TRTYoloDetector(engine_path, input_size=416)

    # Load CNN classifier
    cnn_trt = None
    cnn_model = None
    cnn_engine = os.path.join(MODELS_DIR, 'cnn_classifier.engine')
    cnn_pth = os.path.join(MODELS_DIR, 'gray_trajLen_3_model.pth')
    if os.path.exists(cnn_engine):
        try:
            from realtime_detector import _cuda_ctx
            cnn_trt = TRTCNNClassifier(cnn_engine, _cuda_ctx)
            print("CNN: TensorRT FP16")
        except Exception as e:
            print(f"CNN TRT failed: {e}")
    if cnn_trt is None and os.path.exists(cnn_pth):
        import torch
        cnn_model = torch.jit.load(cnn_pth, map_location='cpu')
        cnn_model.eval()
        print("CNN: PyTorch CPU fallback")

    results = {}
    for video_name, label, direction in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"Skipping {video_name} (not found)")
            continue
        print(f"\nBenchmarking: {video_name} ({label})")
        r = benchmark_video(detector, cnn_trt, cnn_model, video_path, direction)
        if r:
            r['ground_truth'] = label
            r['direction'] = direction
            results[video_name] = r
            print(f"  FPS: {r['end_to_end_fps']} | YOLO: {r['yolo_inference']['mean_ms']:.1f}ms | "
                  f"Tracking: {r['optical_flow_tracking']['mean_ms']:.1f}ms")
            if r['trajectory_rendering']:
                print(f"  Traj render: {r['trajectory_rendering']['mean_ms']:.1f}ms | "
                      f"CNN: {r['cnn_classification']['mean_ms']:.1f}ms" if r['cnn_classification'] else "")

    # Summary
    all_fps = [r['end_to_end_fps'] for r in results.values()]
    all_yolo = [r['yolo_inference']['mean_ms'] for r in results.values()]
    summary = {
        'avg_fps': round(statistics.mean(all_fps), 1) if all_fps else 0,
        'min_fps': round(min(all_fps), 1) if all_fps else 0,
        'max_fps': round(max(all_fps), 1) if all_fps else 0,
        'avg_yolo_ms': round(statistics.mean(all_yolo), 2) if all_yolo else 0,
        'detector_backend': 'TensorRT-FP16',
        'cnn_backend': 'TensorRT-FP16' if cnn_trt else 'PyTorch-CPU',
        'n_videos': len(results),
    }

    output = {'summary': summary, 'per_video': results}
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved: {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))

    # Cleanup CUDA
    try:
        if hasattr(detector, 'cuda_ctx'):
            detector.cuda_ctx.pop()
    except:
        pass
