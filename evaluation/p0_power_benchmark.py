#!/usr/bin/env python3
"""P0 Benchmark 2: Full pipeline power consumption benchmark.

Measures power at three phases using tegrastats:
  1. Idle (no processing)
  2. YOLOv4-only (detection without classification)
  3. Full pipeline (YOLO + tracking + trajectory + CNN)

Outputs JSON with per-phase power stats for paper tables.

Usage on Jetson:
    source venv/bin/activate
    sudo python p0_power_benchmark.py
"""
import subprocess
import time
import re
import json
import os
import sys
import threading
import statistics
import signal

sys.path.insert(0, os.path.expanduser('/home/moseikuffour/darts_edge'))

TEGRASTATS_INTERVAL_MS = 500
PHASE_DURATION = 45  # seconds per phase
WARMUP = 10  # seconds warm-up before measuring
VIDEO_PATH = os.path.expanduser('/home/moseikuffour/darts_edge/left-4_20_22_Road_Ranger_Incident_2_3.MP4')
OUTPUT_PATH = os.path.expanduser('/home/moseikuffour/darts_edge/benchmark_results/power_benchmark.json')


def parse_tegrastats_line(line):
    """Extract power and utilization metrics from a tegrastats line."""
    result = {}
    m = re.search(r'VDD_IN\s+(\d+)mW', line)
    if m:
        result['vdd_in_mw'] = int(m.group(1))
    m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
    if m:
        result['vdd_cpu_gpu_cv_mw'] = int(m.group(1))
    m = re.search(r'VDD_SOC\s+(\d+)mW', line)
    if m:
        result['vdd_soc_mw'] = int(m.group(1))
    m = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
    if m:
        result['ram_used_mb'] = int(m.group(1))
        result['ram_total_mb'] = int(m.group(2))
    m = re.search(r'CPU\s+\[([^\]]+)\]', line)
    if m:
        cpus = re.findall(r'(\d+)%', m.group(1))
        if cpus:
            result['cpu_percent'] = [int(c) for c in cpus]
            result['cpu_avg_percent'] = sum(result['cpu_percent']) / len(result['cpu_percent'])
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        result['gpu_percent'] = int(m.group(1))
    m = re.search(r'tj@([\d.]+)C', line)
    if m:
        result['tj_temp_c'] = float(m.group(1))
    m = re.search(r'gpu@([\d.]+)C', line)
    if m:
        result['gpu_temp_c'] = float(m.group(1))
    return result


def collect_tegrastats(duration_sec, label=""):
    """Run tegrastats for duration_sec and return parsed samples."""
    print(f"  Collecting {label} for {duration_sec}s...")
    samples = []
    proc = subprocess.Popen(
        ['tegrastats', '--interval', str(TEGRASTATS_INTERVAL_MS)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    start = time.time()
    try:
        while time.time() - start < duration_sec:
            line = proc.stdout.readline()
            if not line:
                break
            parsed = parse_tegrastats_line(line.strip())
            if parsed:
                samples.append(parsed)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except:
            proc.kill()
    return samples


def summarize_samples(samples):
    """Compute summary statistics from tegrastats samples."""
    if not samples:
        return {}

    def stat(key):
        vals = [s[key] for s in samples if key in s]
        if not vals:
            return None
        return {
            'mean': round(statistics.mean(vals), 1),
            'min': round(min(vals), 1),
            'max': round(max(vals), 1),
            'std': round(statistics.stdev(vals), 1) if len(vals) > 1 else 0,
        }

    return {
        'n_samples': len(samples),
        'vdd_in_mw': stat('vdd_in_mw'),
        'vdd_cpu_gpu_cv_mw': stat('vdd_cpu_gpu_cv_mw'),
        'vdd_soc_mw': stat('vdd_soc_mw'),
        'ram_used_mb': stat('ram_used_mb'),
        'cpu_avg_percent': stat('cpu_avg_percent'),
        'gpu_percent': stat('gpu_percent'),
        'tj_temp_c': stat('tj_temp_c'),
        'gpu_temp_c': stat('gpu_temp_c'),
    }


def run_yolo_only(video_path, stop_event):
    """Run YOLOv4 detection only (no tracking/classification)."""
    import cv2 as cv
    import numpy as np
    from realtime_detector import TRTYoloDetector

    engine_path = os.path.expanduser('/home/moseikuffour/darts_edge/models/yolov4-custom.engine')
    detector = TRTYoloDetector(engine_path, input_size=416)

    cap = cv.VideoCapture(video_path)
    count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        detections = detector.detect(frame)
        count += 1
    cap.release()
    print(f"  YOLO-only processed {count} frames")
    try:
        detector.cuda_ctx.pop()
    except:
        pass


def run_full_pipeline(video_path, stop_event):
    """Run the full RealtimeProcessor pipeline."""
    from realtime_detector import RealtimeProcessor

    models_dir = os.path.expanduser('/home/moseikuffour/darts_edge/models')
    output_dir = os.path.expanduser('/home/moseikuffour/darts_edge/benchmark_results/power_run')
    os.makedirs(output_dir, exist_ok=True)

    proc = RealtimeProcessor(
        camera_source=video_path,
        models_dir=models_dir,
        output_dir=output_dir,
        video_direction='left',
    )
    proc.start()
    while not stop_event.is_set():
        time.sleep(0.5)
    proc.stop()
    print(f"  Full pipeline: {proc._frame_count} frames, {proc._fps:.1f} FPS")


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = {}

    # Phase 1: Idle
    print("\n=== Phase 1: Idle Power ===")
    idle_samples = collect_tegrastats(PHASE_DURATION, "idle")
    results['idle'] = summarize_samples(idle_samples)
    print(f"  Idle: {results['idle'].get('vdd_in_mw', {}).get('mean', 'N/A')} mW")

    # Phase 2: YOLO-only
    print("\n=== Phase 2: YOLO-Only Power ===")
    stop_yolo = threading.Event()
    yolo_thread = threading.Thread(target=run_yolo_only, args=(VIDEO_PATH, stop_yolo), daemon=True)
    yolo_thread.start()
    time.sleep(WARMUP)  # warm up
    yolo_samples = collect_tegrastats(PHASE_DURATION, "yolo-only")
    stop_yolo.set()
    yolo_thread.join(timeout=10)
    results['yolo_only'] = summarize_samples(yolo_samples)
    print(f"  YOLO-only: {results['yolo_only'].get('vdd_in_mw', {}).get('mean', 'N/A')} mW")

    # Cool-down
    print("\n  Cooling down 15s...")
    time.sleep(15)

    # Phase 3: Full pipeline
    print("\n=== Phase 3: Full Pipeline Power ===")
    stop_full = threading.Event()
    full_thread = threading.Thread(target=run_full_pipeline, args=(VIDEO_PATH, stop_full), daemon=True)
    full_thread.start()
    time.sleep(WARMUP)  # warm up
    full_samples = collect_tegrastats(PHASE_DURATION, "full pipeline")
    stop_full.set()
    full_thread.join(timeout=15)
    results['full_pipeline'] = summarize_samples(full_samples)
    print(f"  Full pipeline: {results['full_pipeline'].get('vdd_in_mw', {}).get('mean', 'N/A')} mW")

    # Derived metrics
    idle_power = results['idle'].get('vdd_in_mw', {}).get('mean', 0)
    yolo_power = results['yolo_only'].get('vdd_in_mw', {}).get('mean', 0)
    full_power = results['full_pipeline'].get('vdd_in_mw', {}).get('mean', 0)
    results['derived'] = {
        'idle_power_w': round(idle_power / 1000, 2),
        'yolo_only_power_w': round(yolo_power / 1000, 2),
        'full_pipeline_power_w': round(full_power / 1000, 2),
        'yolo_overhead_w': round((yolo_power - idle_power) / 1000, 2),
        'full_overhead_w': round((full_power - idle_power) / 1000, 2),
        'tracking_cnn_overhead_w': round((full_power - yolo_power) / 1000, 2),
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved: {OUTPUT_PATH}")
    print(json.dumps(results['derived'], indent=2))
