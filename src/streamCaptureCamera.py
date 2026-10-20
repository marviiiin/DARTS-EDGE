import os
import time
import cv2 as cv
import portalocker
from datetime import datetime


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=512,
    framerate=30,
):
    """GStreamer pipeline for CSI camera on Jetson."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def streamCaptureCamera(source, rootFolder, video_direction, shutdown_event,
                        outputFile, segment_duration=120):
    """Capture video from camera/RTSP and save as segments to watchFolder.

    Args:
        source: Camera source — can be:
            - Integer (0, 1) for USB camera
            - "csi" for Jetson CSI camera via GStreamer
            - "rtsp://..." for RTSP stream
            - File path for testing with local video
        rootFolder: Experiment folder root
        video_direction: "left" or "right"
        shutdown_event: Multiprocessing Event to signal shutdown
        outputFile: Log file path
        segment_duration: Duration of each video segment in seconds
    """
    watch_folder = rootFolder + 'watchFolder1/'
    os.makedirs(watch_folder, exist_ok=True)

    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write(f'Camera capture started: source={source}\n')
        portalocker.unlock(file)

    # Open video source
    if source == "csi":
        cap = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)
    elif isinstance(source, int):
        cap = None
        for idx in [source, source + 1, source + 2]:
            c = cv.VideoCapture(idx, cv.CAP_V4L2)
            if c.isOpened():
                c.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
                c.set(cv.CAP_PROP_FRAME_WIDTH, 720)
                c.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                c.set(cv.CAP_PROP_FPS, 30)
                c.set(cv.CAP_PROP_BUFFERSIZE, 1)
                ret, _ = c.read()
                if ret:
                    cap = c
                    break
                c.release()
            else:
                c.release()
        if cap is None:
            cap = cv.VideoCapture(source)
    elif isinstance(source, str) and source.startswith("rtsp"):
        # RTSP with GStreamer for hardware decoding on Jetson
        gst = (
            f'rtspsrc location={source} latency=200 ! '
            f'rtph264depay ! h264parse ! nvv4l2decoder ! '
            f'nvvidconv ! video/x-raw, format=BGRx ! '
            f'videoconvert ! video/x-raw, format=BGR ! appsink'
        )
        cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
        if not cap.isOpened():
            # Fallback to FFmpeg backend
            cap = cv.VideoCapture(source)
    else:
        cap = cv.VideoCapture(source)

    if not cap.isOpened():
        with open(outputFile, 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write(f'Error: Cannot open video source: {source}\n')
            portalocker.unlock(file)
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 512

    segment_idx = 0

    while not shutdown_event.is_set():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_filename = f"{timestamp}-{video_direction}.mp4"
        segment_path = os.path.join(watch_folder, segment_filename)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(segment_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            with open(outputFile, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                file.write(f'Error: Cannot create video writer for {segment_path}\n')
                portalocker.unlock(file)
            break

        frame_count = 0
        target_frames = int(fps * segment_duration)
        start_time = time.time()

        with open(outputFile, 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write(f'Recording segment {segment_idx}: {segment_filename}\n')
            portalocker.unlock(file)

        while frame_count < target_frames and not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # For live camera, brief read failures can happen
                time.sleep(0.01)
                continue
            writer.write(frame)
            frame_count += 1

        writer.release()
        elapsed = time.time() - start_time

        with open(outputFile, 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write(f'Segment {segment_idx} saved: {segment_filename} '
                        f'({frame_count} frames, {elapsed:.1f}s)\n')
            portalocker.unlock(file)

        segment_idx += 1

    cap.release()
    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('Camera capture finished.\n')
        portalocker.unlock(file)
