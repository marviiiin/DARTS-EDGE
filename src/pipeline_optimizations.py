"""GPU-accelerated pipeline helpers for DARTS edge detection.

Replaces slow CPU paths (matplotlib rendering, PyTorch CNN, CPU video decode)
with GPU-accelerated alternatives (OpenCV drawing, TensorRT CNN, NVDEC decode).
"""
import numpy as np
import cv2 as cv
import os

# ── Fast trajectory rendering (replaces matplotlib ~200ms → <0.5ms) ────────

def cv_render_trajectory(selected_trajectories, video_width, video_height,
                         speed_background, detect_interval, background_direction,
                         trajectory_len=75):
    """Render trajectory image using matplotlib, exactly matching training (step2).

    Generates 224x224 via matplotlib, converts to grayscale, inverts,
    and resizes to 56x56 with LANCZOS — identical to step2_generate_dataset.py.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    from io import BytesIO

    OUTPUT_SIZE = 56

    # x-range matching training pipeline (step2_generate_dataset.py xlim)
    # Both directions use subtraction compensation, so xlim extends left
    xlim = [-trajectory_len * speed_background, video_width]

    # Render with matplotlib (matching step2 exactly)
    buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(2.24, 2.24))
    ax.set_xlim(xlim)
    ax.set_ylim([0, video_height])
    ax.invert_yaxis()
    ax.axis('off')
    for traj_x, traj_y in selected_trajectories:
        if len(traj_x) >= 2:
            ax.plot(traj_x, traj_y, color='black', linewidth=1)
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)

    # Convert to grayscale, invert, resize to 56x56 (matching step2)
    buffer.seek(0)
    plot_img = Image.open(buffer).convert("L")
    inverted_img = ImageOps.invert(plot_img)
    resized_img = inverted_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    # Convert back to 3-channel numpy for compatibility with downstream code
    gray_np = np.array(resized_img)
    canvas = cv.merge([gray_np, gray_np, gray_np])

    plot_img.close()
    inverted_img.close()
    resized_img.close()
    buffer.close()

    return canvas



# ── NVDEC hardware video decoder (Jetson hardware accelerated) ─────────────

def nvdec_video_capture(video_path):
    """Open video with Jetson NVDEC hardware decoder via GStreamer.

    Falls back to regular OpenCV if GStreamer is not available.
    Returns (cap, is_hw_decoded).
    """
    # Try GStreamer NVDEC pipeline first
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ('.mp4', '.mov', '.avi', '.mkv'):
        # Detect codec — try h264 first (most common), then h265
        for parser, decoder in [('h264parse', 'nvv4l2decoder'),
                                ('h265parse', 'nvv4l2decoder')]:
            gst = (f'filesrc location="{video_path}" ! qtdemux ! {parser} ! '
                   f'{decoder} ! nvvidconv ! '
                   f'video/x-raw, format=BGRx ! videoconvert ! '
                   f'video/x-raw, format=BGR ! appsink drop=1')
            cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, True
            cap.release()

    # Fallback to CPU decode
    cap = cv.VideoCapture(video_path)
    return cap, False


# ── In-memory CNN batch classifier ────────────────────────────────────────

def classify_trajectory_batch_memory(images_np_list, model=None, trt_classifier=None, device=None):
    """Classify a batch of trajectory images (numpy arrays) without disk I/O.

    Args:
        images_np_list: list of numpy arrays (56x56x3 uint8, BGR)
        model: PyTorch model (fallback)
        trt_classifier: TRTCNNClassifier instance (preferred)
        device: torch device for PyTorch fallback

    Returns:
        numpy array of predictions
    """
    if not images_np_list:
        return np.array([])

    import torchvision.transforms as transforms

    # Preprocess: BGR uint8 → RGB float32 tensor [N, 3, 56, 56]
    batch = np.zeros((len(images_np_list), 3, 56, 56), dtype=np.float32)
    for i, img in enumerate(images_np_list):
        # img is 56x56x3 BGR uint8
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch[i] = rgb.transpose(2, 0, 1)  # HWC → CHW

    # TensorRT path (fastest)
    if trt_classifier is not None:
        return trt_classifier.predict_batch(batch)

    # PyTorch fallback
    if model is not None:
        import torch
        tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            out = model(tensor)
            preds = torch.max(out, 1)[1].cpu().numpy()
        return preds

    return np.array([])
