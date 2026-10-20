"""Real-time YOLOv4 detector for edge device.

Captures frames from HDMI/camera, runs YOLOv4 detection via TensorRT GPU
(~100 FPS) or OpenCV DNN CPU fallback, draws bounding boxes, streams
annotated frames via MJPEG, and runs trajectory-based incident analysis.
"""
import os
import time
import threading
import json
import io
import collections
import numpy as np
import cv2 as cv
from datetime import datetime
# VLM auto-analysis on incidentfrom vlm_analysis import analyze_incident, save_results as save_vlm_results

# PyTorch for CNN classification (CPU only)
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TensorRT + PyCUDA
_cuda_ctx = None
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    cuda.init()
    _cuda_ctx = cuda.Device(0).make_context()
    HAS_TRT = True
    print(f"TensorRT {trt.__version__} + PyCUDA loaded OK")
except Exception as e:
    HAS_TRT = False
    print(f"TensorRT/PyCUDA not available: {e}")


# ── TensorRT YOLOv4 Detector ───────────────────────────────────────────

class TRTYoloDetector:
    """YOLOv4 detector using TensorRT engine on GPU."""

    # YOLOv4-custom anchors and masks from .cfg
    ANCHORS = [(12,16), (19,36), (40,28), (36,75), (76,55), (72,146),
               (142,110), (192,243), (459,401)]
    MASKS = {52: [0,1,2], 26: [3,4,5], 13: [6,7,8]}  # grid_size -> anchor indices

    def __init__(self, engine_path, input_size=416, conf_thresh=0.70, nms_thresh=0.4):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.backend = 'TensorRT-FP16'
        self.cuda_ctx = _cuda_ctx

        # Push context for this thread
        self.cuda_ctx.push()

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = list(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            mode = self.engine.get_tensor_mode(name)
            binding = {'name': name, 'shape': shape, 'dtype': dtype,
                       'host': host_mem, 'device': device_mem}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        print(f"TRT engine loaded: {len(self.inputs)} inputs, {len(self.outputs)} outputs")
        for o in self.outputs:
            print(f"  Output '{o['name']}': shape={o['shape']}, dtype={o['dtype']}")

        self.cuda_ctx.pop()

    def detect(self, frame):
        """Run detection. Returns list of (x, y, w, h, confidence)."""
        self.cuda_ctx.push()
        try:
            return self._detect_inner(frame)
        finally:
            self.cuda_ctx.pop()

    def _detect_inner(self, frame):
        h, w = frame.shape[:2]

        # Preprocess
        img = cv.resize(frame, (self.input_size, self.input_size))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(img, 0).astype(self.inputs[0]['dtype'])
        blob = np.ascontiguousarray(blob)

        # Copy to GPU
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        for b in self.inputs + self.outputs:
            self.context.set_tensor_address(b['name'], int(b['device']))

        # Inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy results back
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o['host'], o['device'], self.stream)
        self.stream.synchronize()

        # Parse YOLO outputs
        boxes, confidences = [], []
        for o in self.outputs:
            data = o['host'].reshape(o['shape'])
            self._parse_yolo(data, w, h, boxes, confidences)

        # NMS
        if len(boxes) > 0:
            indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
            if len(indices) > 0:
                indices = indices.flatten()
                return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                         confidences[i]) for i in indices]
        return []

    def _parse_yolo(self, output, img_w, img_h, boxes, confidences):
        """Parse raw YOLO output tensor with proper anchor decoding."""
        # Remove batch dim: [1, 18, gh, gw] -> [18, gh, gw]
        if len(output.shape) == 4:
            output = output[0]
        if len(output.shape) != 3:
            return

        c, gh, gw = output.shape
        num_attrs = 6  # tx, ty, tw, th, objectness, 1 class
        n_anchors = c // num_attrs
        if c % num_attrs != 0 or n_anchors != 3:
            return

        # Get anchors for this grid size
        mask = self.MASKS.get(gw)
        if mask is None:
            return
        anchors = [self.ANCHORS[m] for m in mask]
        stride = self.input_size // gw  # 416/52=8, 416/26=16, 416/13=32

        # Reshape: [3, 6, gh, gw] -> [3, gh, gw, 6]
        output = output.reshape(n_anchors, num_attrs, gh, gw)
        output = output.transpose(0, 2, 3, 1)  # [3, gh, gw, 6]

        # Build grid offsets
        grid_y, grid_x = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')

        for a in range(n_anchors):
            anchor_w, anchor_h = anchors[a]
            data = output[a]  # [gh, gw, 6]

            # Sigmoid objectness, filter early
            obj = 1.0 / (1.0 + np.exp(-np.clip(data[:, :, 4], -50, 50)))
            cls = 1.0 / (1.0 + np.exp(-np.clip(data[:, :, 5], -50, 50)))
            conf_map = obj * cls

            # Find cells above threshold
            ys, xs = np.where(conf_map >= self.conf_thresh)
            if len(ys) == 0:
                continue

            for y, x in zip(ys, xs):
                conf = float(conf_map[y, x])
                tx, ty, tw, th = data[y, x, :4]

                # Decode: sigmoid(tx)+grid_x, sigmoid(ty)+grid_y, scaled by stride
                bx_center = (1.0 / (1.0 + np.exp(-tx)) + x) * stride
                by_center = (1.0 / (1.0 + np.exp(-ty)) + y) * stride
                # exp(tw)*anchor_w, exp(th)*anchor_h
                bw = np.exp(min(tw, 10.0)) * anchor_w
                bh = np.exp(min(th, 10.0)) * anchor_h

                # Scale from input_size coords to original image coords
                bx_center = bx_center * img_w / self.input_size
                by_center = by_center * img_h / self.input_size
                bw = bw * img_w / self.input_size
                bh = bh * img_h / self.input_size

                bx = int(bx_center - bw / 2)
                by_val = int(by_center - bh / 2)
                bw_int = int(bw)
                bh_int = int(bh)
                if bw_int > 0 and bh_int > 0:
                    boxes.append([bx, by_val, bw_int, bh_int])
                    confidences.append(conf)


# ── OpenCV DNN Fallback ─────────────────────────────────────────────────

class OpenCVYoloDetector:
    """YOLOv4 detector using OpenCV DNN (CPU fallback)."""

    def __init__(self, cfg_path, weights_path, input_size=416,
                 conf_thresh=0.70, nms_thresh=0.4):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.net = cv.dnn.readNetFromDarknet(cfg_path, weights_path)
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
            self.backend = 'OpenCV-CUDA'
        else:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            self.backend = 'OpenCV-CPU'
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1/255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        boxes, confidences = [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                confidence = scores[0]
                if confidence >= self.conf_thresh:
                    cx, cy = int(det[0] * w), int(det[1] * h)
                    bw, bh = int(det[2] * w), int(det[3] * h)
                    boxes.append([cx - bw//2, cy - bh//2, bw, bh])
                    confidences.append(float(confidence))
        if boxes:
            indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
            if len(indices) > 0:
                indices = indices.flatten()
                return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                         confidences[i]) for i in indices]
        return []


# ── TensorRT CNN Classifier ──────────────────────────────────────────────

class TRTCNNClassifier:
    """CNN trajectory classifier using TensorRT on GPU."""

    def __init__(self, engine_path, cuda_ctx):
        self.cuda_ctx = cuda_ctx
        self.cuda_ctx.push()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.input_binding = None
            self.output_binding = None
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = list(self.engine.get_tensor_shape(name))
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_binding = {'name': name, 'shape': shape, 'dtype': dtype}
                else:
                    self.output_binding = {'name': name, 'shape': shape, 'dtype': dtype}

            # Fix dynamic batch dim (-1 -> 1) for memory allocation
            inp_shape = [max(1, s) for s in self.input_binding['shape']]
            out_shape = [max(1, s) for s in self.output_binding['shape']]
            self.input_binding['shape'] = inp_shape
            self.output_binding['shape'] = out_shape
            inp_size = int(np.prod(inp_shape))
            out_size = int(np.prod(out_shape))
            self.inp_host = np.empty(inp_size, dtype=self.input_binding['dtype'])
            self.out_host = np.empty(out_size, dtype=self.output_binding['dtype'])
            self.inp_dev = cuda.mem_alloc(self.inp_host.nbytes)
            self.out_dev = cuda.mem_alloc(self.out_host.nbytes)

            print(f"TRT CNN loaded: input={self.input_binding['shape']}, "
                  f"output={self.output_binding['shape']}", flush=True)
        finally:
            self.cuda_ctx.pop()

    def predict_batch(self, images_np):
        """Run classification on a batch of preprocessed images.

        Args:
            images_np: numpy array [N, 3, 56, 56] float32, normalized [0,1]

        Returns:
            numpy array of predicted class indices [N]
        """
        self.cuda_ctx.push()
        try:
            batch_size = images_np.shape[0]
            results = []
            for i in range(batch_size):
                inp = np.ascontiguousarray(images_np[i:i+1])
                np.copyto(self.inp_host, inp.ravel())

                cuda.memcpy_htod(self.inp_dev, self.inp_host)
                self.context.set_input_shape(self.input_binding['name'], [1, 3, 56, 56])
                self.context.set_tensor_address(self.input_binding['name'], int(self.inp_dev))
                self.context.set_tensor_address(self.output_binding['name'], int(self.out_dev))
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                cuda.memcpy_dtoh(self.out_host, self.out_dev)
                self.stream.synchronize()

                results.append(int(np.argmax(self.out_host[:3])))

            return np.array(results)
        finally:
            self.cuda_ctx.pop()



# ── Real-Time Processing Pipeline ──────────────────────────────────────

class RealtimeProcessor:
    """Continuous real-time video processing with GPU-accelerated YOLO."""

    def __init__(self, camera_source, models_dir, output_dir,
                 video_direction='left', dashboard_url=None,
                 input_size=416, skip_frames=3,
                 detection_results_file=None, experiment_folder=None):
        self.camera_source = camera_source
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.video_direction = video_direction
        self.input_size = input_size
        self.skip_frames = skip_frames
        self.detection_results_file = detection_results_file
        self.experiment_folder = experiment_folder
        self._live_counts = [0, 0, 0]  # [normal, congestion, incident]

        self._frame_lock = threading.Lock()
        self._latest_annotated = None
        self._running = False
        self._thread = None
        self._fps = 0.0
        self._frame_count = 0
        self._detection_count = 0
        self._status = 'idle'
        self._incident_log = []

        self.video_fps = 30
        self.video_width = 640
        self.video_height = 512
        self.speed_background = 0.0  # 0 for stationary camera (capture card)
        self.trajectory_len = 90

        self._recording_incident = False

        # Background classification thread to avoid blocking video pipeline
        import queue as _queue
        self._classify_queue = _queue.Queue(maxsize=4)
        self._classify_thread = threading.Thread(target=self._classify_worker, daemon=True)
        self._classify_thread.start()

        # Load detector — prefer TensorRT, fall back to OpenCV DNN
        engine_path = os.path.join(models_dir, 'yolov4-custom.engine')
        cfg_path = os.path.join(models_dir, 'yolov4-custom.cfg')
        weights_path = os.path.join(models_dir, 'yolov4-custom_final.weights')

        if HAS_TRT and os.path.exists(engine_path):
            try:
                self.detector = TRTYoloDetector(engine_path, input_size)
                self._backend = self.detector.backend
                print("Using TensorRT GPU engine for real-time detection",
                      flush=True)
            except Exception as e:
                print(f"TRT engine load failed ({e}), falling back to OpenCV DNN",
                      flush=True)
                self.detector = OpenCVYoloDetector(cfg_path, weights_path, input_size)
                self._backend = self.detector.backend
        else:
            self.detector = OpenCVYoloDetector(cfg_path, weights_path, input_size)
            self._backend = self.detector.backend

        # CNN model for trajectory classification — prefer TRT GPU, fallback PyTorch CPU
        self.cnn_trt = None
        self.cnn_model = None
        cnn_engine_path = os.path.join(models_dir, 'cnn_classifier.engine')
        cnn_pth_path = os.path.join(models_dir, 'gray_trajLen_3_model.pth')
        if HAS_TRT and os.path.exists(cnn_engine_path):
            try:
                self.cnn_trt = TRTCNNClassifier(cnn_engine_path, _cuda_ctx)
                print("CNN classifier: TensorRT GPU", flush=True)
            except Exception as e:
                print(f"CNN TRT failed ({e}), using PyTorch CPU", flush=True)
                if os.path.exists(cnn_pth_path):
                    self.cnn_model = torch.jit.load(cnn_pth_path, map_location='cpu')
                    self.cnn_model.eval()
        elif os.path.exists(cnn_pth_path):
            self.cnn_model = torch.jit.load(cnn_pth_path, map_location='cpu')
            self.cnn_model.eval()
            print("CNN classifier: PyTorch CPU", flush=True)
        # Always load PyTorch CPU model for background thread (avoids CUDA contention)
        if self.cnn_model is None and os.path.exists(cnn_pth_path):
            self.cnn_model = torch.jit.load(cnn_pth_path, map_location='cpu')
            self.cnn_model.eval()
            print("CNN CPU model also loaded for background classification", flush=True)

    @property
    def annotated_frame(self):
        with self._frame_lock:
            return self._latest_annotated

    @property
    def status(self):
        return {
            'running': self._running,
            'backend': self._backend,
            'fps': round(self._fps, 1),
            'frame_count': self._frame_count,
            'detection_count': self._detection_count,
            'status': self._status,
            'incidents': self._incident_log[-5:],
        }

    def start(self):
        if self._running:
            return
        self._running = True
        self._status = 'starting'
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._status = 'stopped'

    def _run(self):
        cap = self._open_camera()
        if cap is None:
            return

        fps = cap.get(cv.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        self.video_fps = int(fps)
        self.video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
        self.video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 512
        self.trajectory_len = self.video_fps * 3
        detect_interval = self.skip_frames
        slidingWindow_frame = self.video_fps

        self._status = 'running'
        print(f"Realtime: {self.video_width}x{self.video_height}@{fps:.0f}fps, "
              f"backend={self._backend}, skip={self.skip_frames}")

        lk_params = dict(winSize=(11, 11), maxLevel=2,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        trajectories = []
        traj_images = []  # in-memory trajectory images for CNN
        prev_gray = None
        frame_idx = 0
        fig_idx = 0
        bg_dir = self.video_direction

        fps_timer = time.time()
        fps_count = 0

        # Threaded camera reader to avoid blocking on V4L2 select() timeout
        import queue as _queue
        frame_q = _queue.Queue(maxsize=2)
        reader_alive = [True]

        def _reader_thread(cap_ref, q, alive):
            while alive[0]:
                ret, frm = cap_ref[0].read()
                if not ret:
                    q.put((False, None))
                    time.sleep(0.01)
                    continue
                if q.full():
                    try:
                        q.get_nowait()
                    except _queue.Empty:
                        pass
                q.put((True, frm))

        cap_ref = [cap]
        reader_t = threading.Thread(target=_reader_thread,
                                    args=(cap_ref, frame_q, reader_alive), daemon=True)
        reader_t.start()

        stall_timeout = 5.0  # seconds without a frame = reconnect
        last_frame_time = time.time()

        while self._running:
            try:
                ret, frame = frame_q.get(timeout=0.1)
            except _queue.Empty:
                ret, frame = False, None

            if not ret or frame is None:
                if isinstance(self.camera_source, str) and os.path.isfile(self.camera_source):
                    cap_ref[0].set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue
                if time.time() - last_frame_time > stall_timeout:
                    print(f"Camera stalled for {stall_timeout}s, reconnecting...")
                    reader_alive[0] = False
                    reader_t.join(timeout=2)
                    cap_ref[0].release()
                    time.sleep(1)
                    cap = self._open_camera()
                    if cap is None:
                        break
                    cap_ref[0] = cap
                    prev_gray = None
                    trajectories = []
                    reader_alive[0] = True
                    frame_q = _queue.Queue(maxsize=2)
                    reader_t = threading.Thread(target=_reader_thread,
                                                args=(cap_ref, frame_q, reader_alive), daemon=True)
                    reader_t.start()
                    last_frame_time = time.time()
                continue
            last_frame_time = time.time()

            frame_idx += 1
            self._frame_count = frame_idx
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            annotated = frame.copy()

            # YOLO detection
            detections = []
            if frame_idx % detect_interval == 0:
                detections = self.detector.detect(frame)
                self._detection_count += len(detections)

            # Optical flow tracking (forward only, no reverse check)
            if prev_gray is not None and trajectories:
                p0 = np.float32([t[-1] for t in trajectories]).reshape(-1, 1, 2)
                p1, st, _ = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                new_t = []
                for tr, (x, y), ok in zip(trajectories, p1.reshape(-1, 2), st.ravel()):
                    if not ok:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.trajectory_len:
                        del tr[0]
                    new_t.append(tr)
                trajectories = new_t

            # Add new detections to trajectories
            if frame_idx % detect_interval == 0 and detections:
                for (bx, by, bw, bh, _) in detections:
                    cx, cy = bx + bw / 2, by + bh / 2
                    if not any(abs(t[-1][0]-cx) < bw+30 and abs(t[-1][1]-cy) < bh
                               for t in trajectories):
                        trajectories.append([(cx, cy)])

            # Generate trajectory image in memory + CNN classify
            # Skip first 10s (startup grace) to avoid false positives from stale state
            startup_grace = self.video_fps * 10
            if (frame_idx > max(self.trajectory_len, startup_grace) and
                    (frame_idx - detect_interval) % slidingWindow_frame == 0):
                traj_img = self._render_traj_in_memory(trajectories, bg_dir)
                if traj_img is not None:
                    traj_images.append(traj_img)
                    fig_idx += 1
                if fig_idx > 0 and fig_idx % 10 == 0:
                    if not self._classify_queue.full():
                        self._classify_queue.put((list(trajectories), list(traj_images), frame_idx))
                    traj_images = []

            # Overlay
            cv.putText(annotated, f'FPS: {self._fps:.1f} | {self._backend}',
                       (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(annotated, f'Frame: {frame_idx} | Vehicles: {len(detections)}',
                       (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if self._incident_log:
                last = self._incident_log[-1]
                color = (0, 0, 255) if 'Incident' in last['label'] else (0, 255, 0)
                cv.putText(annotated, last['label'],
                           (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            with self._frame_lock:
                self._latest_annotated = annotated
            prev_gray = frame_gray

            fps_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self._fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

        reader_alive[0] = False
        reader_t.join(timeout=2)
        cap_ref[0].release()
        if traj_images:
            self._classify_trajectories(trajectories, traj_images, frame_idx)
        self._status = 'stopped'

    def _write_dashboard_results(self, result):
        import portalocker
        import time as _time
        label = result["label"]
        if "Incident" in label:
            self._live_counts[2] += 1
            class_id = 2
        elif "Congestion" in label:
            self._live_counts[1] += 1
            class_id = 1
        else:
            self._live_counts[0] += 1
            class_id = 0
        if self.detection_results_file:
            try:
                duration = result.get("frame", 0) / max(self.video_fps, 1)
                line = f"live_camera,{duration:.0f},{label}\n"
                with open(self.detection_results_file, "a") as f:
                    portalocker.lock(f, portalocker.LOCK_EX)
                    f.write(line)
                    portalocker.unlock(f)
            except Exception as e:
                print(f"Error writing detection results: {e}")
        if self.experiment_folder:
            try:
                status_file = os.path.join(self.experiment_folder, "live_status.json")
                status = {
                    "label": label,
                    "class_id": int(class_id),
                    "frame_second": float(result.get("frame", 0) / max(self.video_fps, 1)),
                    "video_name": "live_camera",
                    "counts": {"normal": self._live_counts[0], "congestion": self._live_counts[1], "incident": self._live_counts[2]},
                    "total": sum(self._live_counts),
                    "timestamp": _time.time(),
                }
                with open(status_file, "w") as sf:
                    portalocker.lock(sf, portalocker.LOCK_EX)
                    json.dump(status, sf)
                    portalocker.unlock(sf)
            except Exception as e:
                print(f"Error writing live status: {e}")

    def _open_camera(self):
        src = self.camera_source
        if src == 'csi':
            gst = ('nvarguscamerasrc sensor-id=0 ! '
                   'video/x-raw(memory:NVMM), width=640, height=512, framerate=30/1 ! '
                   'nvvidconv ! video/x-raw, format=BGRx ! '
                   'videoconvert ! video/x-raw, format=BGR ! appsink')
            cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
        else:
            dev = src if not isinstance(src, str) or not src.isdigit() else int(src)
            cap = None
            if isinstance(dev, int):
                # V4L2 MJPG at 720x480 (lowest supported res = least USB bandwidth)
                for idx in [dev, dev + 1, dev + 2]:
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
                            print(f"Camera opened on /dev/video{idx} (V4L2 MJPG 720x480)")
                            break
                        c.release()
                    else:
                        c.release()
            if cap is None:
                cap = cv.VideoCapture(dev)
        if not cap.isOpened():
            self._status = f'error: cannot open {src}'
            self._running = False
            return None
        return cap

    def _classify_worker(self):
        """Background thread that runs CNN classification without blocking video.
        Uses PyTorch CPU to avoid CUDA context contention with YOLO on main thread.
        """
        while True:
            try:
                trajectories, traj_images, frame_idx = self._classify_queue.get()
                if traj_images is None:
                    break  # shutdown signal
                self._classify_trajectories_bg(trajectories, traj_images, frame_idx)
            except Exception as e:
                print(f"Classify worker error: {e}", flush=True)

    def _classify_trajectories_bg(self, trajectories, traj_images, frame_idx):
        """Background-safe classification using PyTorch CPU (no CUDA contention)."""
        if not traj_images:
            if trajectories:
                self._classify_heuristic(trajectories, frame_idx)
            return

        total_tracked = len(trajectories)
        try:
            batch = np.zeros((len(traj_images), 3, 56, 56), dtype=np.float32)
            for i, img in enumerate(traj_images):
                if isinstance(img, Image.Image):
                    img_np = np.array(img.convert('RGB'))
                else:
                    img_np = img
                rgb = img_np.astype(np.float32) / 255.0
                batch[i] = rgb.transpose(2, 0, 1)

            # Always use PyTorch CPU here to avoid blocking the YOLO GPU thread
            if self.cnn_model is not None:
                tensor = torch.from_numpy(batch)
                with torch.no_grad():
                    out = self.cnn_model(tensor)
                    predictions = torch.max(out, 1)[1].cpu().numpy()
            elif self.cnn_trt is not None:
                predictions = self.cnn_trt.predict_batch(batch)
            else:
                self._classify_heuristic(trajectories, frame_idx)
                return

            n_images = len(predictions)
            n_normal = int(np.sum(predictions == 0))
            n_congestion = int(np.sum(predictions == 1))
            n_incident = int(np.sum(predictions == 2))

            tau_I = n_incident / n_images
            tau_R = n_congestion / n_images
            tau_N = n_normal / n_images

            if tau_I >= 0.3:
                label = 'Incident Detected'
            elif tau_N >= 0.7:
                label = 'Normal Traffic'
            else:
                label = 'Congestion Propagating'

            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'frame': frame_idx,
                'label': label,
                'incident_ratio': round(float(tau_I * 100), 1),
                'congestion_ratio': round(float(tau_R * 100), 1),
                'normal_ratio': round(float(tau_N * 100), 1),
                'trajectory_count': total_tracked,
                'images_classified': n_images,
                'classifier': 'TCD-Net-CPU-BG',
            }
            self._incident_log.append(result)
            self._write_dashboard_results(result)
            print(f"TCD-Net-BG: {label} | I={tau_I:.2f} C={tau_R:.2f} N={tau_N:.2f} "
                  f"({n_incident}/{n_congestion}/{n_normal} of {n_images}) | "
                  f"tracked={total_tracked}", flush=True)

            results_dir = os.path.join(self.output_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f'result_{frame_idx}.json'), 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"BG classify error: {e}, falling back to heuristic", flush=True)
            self._classify_heuristic(trajectories, frame_idx)

    def _render_traj_in_memory(self, trajectories, bg_dir):
        """Render trajectory plot to a 56x56x3 numpy array matching TCD-Net training pipeline.

        Uses OpenCV line drawing instead of matplotlib for ~200x speedup.
        Produces white lines on black background at 56x56, matching training output.
        """
        try:
            selected = []
            for tr in trajectories:
                xs, ys = [], []
                for i, pt in enumerate(tr):
                    xs.append(pt[0] - self.speed_background * i)
                    ys.append(pt[1])
                if len(xs) == 1 or (len(xs) > 1 and xs[-1] - xs[0] >= 0):
                    selected.append([xs, ys])
            if not selected:
                return None

            RENDER_SIZE = 224  # match matplotlib 2.24in * 100dpi
            # Coordinate mapping matching matplotlib xlim/ylim
            xlim_min = -self.speed_background * self.trajectory_len
            xlim_max = self.video_width
            ylim_min = 0
            ylim_max = self.video_height
            x_range = xlim_max - xlim_min if xlim_max != xlim_min else 1
            y_range = ylim_max - ylim_min if ylim_max != ylim_min else 1

            # White canvas (matching matplotlib white background)
            canvas = np.ones((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8) * 255

            for xs, ys in selected:
                if len(xs) < 2:
                    continue
                pts = []
                for x, y in zip(xs, ys):
                    px = int((x - xlim_min) / x_range * (RENDER_SIZE - 1))
                    py = int((y - ylim_min) / y_range * (RENDER_SIZE - 1))
                    pts.append((px, py))
                # Draw black lines on white (matching matplotlib black lines)
                for j in range(len(pts) - 1):
                    cv.line(canvas, pts[j], pts[j + 1], 0, 1)

            # Invert: black lines on white -> white lines on black (matching PIL ImageOps.invert)
            canvas = 255 - canvas
            # Resize to 56x56 (matching PIL LANCZOS)
            canvas = cv.resize(canvas, (56, 56), interpolation=cv.INTER_LANCZOS4)
            # 3-channel (TCD-Net expects [N, 3, 56, 56])
            result = np.stack([canvas, canvas, canvas], axis=-1)
            return result  # numpy array 56x56x3 uint8
        except Exception as e:
            print(f"Trajectory render error: {e}")
            return None

    def _classify_trajectories(self, trajectories, traj_images, frame_idx):
        """Classify traffic using TCD-Net CNN on trajectory images.

        Uses the original DARTS image-to-video aggregation:
        - Classify each trajectory image with TCD-Net (TRT or PyTorch)
        - Compute tau_I (incident ratio), tau_R (congestion ratio), tau_N (normal ratio)
        - If tau_I > threshold -> Incident Detected
        - If tau_N > threshold -> Normal Traffic
        - Otherwise -> Congestion (recurrent)

        Falls back to heuristic if no CNN model is available.
        """
        if not traj_images:
            if not trajectories:
                return
            # No images available — use heuristic fallback
            self._classify_heuristic(trajectories, frame_idx)
            return

        total_tracked = len(trajectories)

        try:
            # Preprocess trajectory images for CNN: [N, 3, 56, 56] float32
            batch = np.zeros((len(traj_images), 3, 56, 56), dtype=np.float32)
            for i, img in enumerate(traj_images):
                if isinstance(img, Image.Image):
                    # PIL Image — convert to numpy
                    img_np = np.array(img.convert('RGB'))
                else:
                    # Already numpy (56x56x3 uint8)
                    img_np = img
                # Normalize to [0, 1] and transpose HWC -> CHW
                rgb = img_np.astype(np.float32) / 255.0
                batch[i] = rgb.transpose(2, 0, 1)

            # Classify with TCD-Net
            if self.cnn_trt is not None:
                predictions = self.cnn_trt.predict_batch(batch)
            elif self.cnn_model is not None:
                tensor = torch.from_numpy(batch)
                with torch.no_grad():
                    out = self.cnn_model(tensor)
                    predictions = torch.max(out, 1)[1].cpu().numpy()
            else:
                # No CNN available — fall back to heuristic
                self._classify_heuristic(trajectories, frame_idx)
                traj_images.clear()
                return

            # Image-to-video aggregation (matching original DARTS)
            # Class mapping: 0=Normal, 1=Congestion, 2=Incident
            n_images = len(predictions)
            n_normal = int(np.sum(predictions == 0))
            n_congestion = int(np.sum(predictions == 1))
            n_incident = int(np.sum(predictions == 2))

            tau_I = n_incident / n_images      # incident ratio
            tau_R = n_congestion / n_images     # congestion ratio
            tau_N = n_normal / n_images         # normal ratio

            # Decision thresholds (from original DARTS)
            if tau_I >= 0.3:
                label = 'Incident Detected'
            elif tau_N >= 0.7:
                label = 'Normal Traffic'
            else:
                label = 'Congestion Propagating'

            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'frame': frame_idx,
                'label': label,
                'incident_ratio': round(float(tau_I * 100), 1),
                'congestion_ratio': round(float(tau_R * 100), 1),
                'normal_ratio': round(float(tau_N * 100), 1),
                'trajectory_count': total_tracked,
                'images_classified': n_images,
                'classifier': 'TCD-Net-TRT' if self.cnn_trt else 'TCD-Net-PyTorch',
            }
            self._incident_log.append(result)
            self._write_dashboard_results(result)
            print(f"TCD-Net: {label} | I={tau_I:.2f} C={tau_R:.2f} N={tau_N:.2f} "
                  f"({n_incident}/{n_congestion}/{n_normal} of {n_images}) | "
                  f"tracked={total_tracked}", flush=True)

            results_dir = os.path.join(self.output_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f'result_{frame_idx}.json'), 'w') as f:
                json.dump(result, f, indent=2)

            traj_images.clear()
        except Exception as e:
            print(f"TCD-Net classify error: {e}, falling back to heuristic", flush=True)
            self._classify_heuristic(trajectories, frame_idx)
            traj_images.clear()

    def _classify_heuristic(self, trajectories, frame_idx):
        """Fallback heuristic classifier using trajectory kinematics."""
        total_tracked = len(trajectories)
        speeds = []
        stopped_count = 0

        for tr in trajectories:
            if len(tr) < 5:
                continue
            recent = tr[-min(10, len(tr)):]
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]
            dist = (dx**2 + dy**2) ** 0.5
            spd = dist / max(len(recent) - 1, 1)
            speeds.append(spd)
            if spd < 1.5:
                stopped_count += 1

        n_moving = len(speeds)
        if n_moving == 0:
            return

        avg_speed = float(np.mean(speeds))
        stopped_ratio = stopped_count / n_moving
        speed_variance = float(np.std(speeds)) if len(speeds) > 1 else 0

        if n_moving < 5:
            label = 'Normal Traffic'
        elif stopped_ratio >= 0.5 and speed_variance > 3.5 and n_moving >= 8:
            label = 'Incident Detected'
        elif stopped_ratio >= 0.3 or avg_speed < 2.0:
            label = 'Congestion Propagating'
        else:
            label = 'Normal Traffic'

        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'frame': frame_idx,
            'label': label,
            'incident_ratio': 0.0,
            'congestion_ratio': round(float(stopped_ratio * 100), 1),
            'trajectory_count': total_tracked,
            'classifier': 'heuristic-fallback',
        }
        self._incident_log.append(result)
        self._write_dashboard_results(result)
        print(f"HEURISTIC: {label} | avg_spd={avg_speed:.1f} stopped={stopped_count}/{n_moving} "
              f"var={speed_variance:.1f} | tracked={total_tracked}", flush=True)

