import shutil
import os, sys
from datetime import datetime
import re
import cv2 as cv
import struct
import time
import numpy as np
import math
import statistics
import portalocker
import pandas as pd
import matplotlib.pyplot as plt
import pylab as p
import matplotlib.image as mpimg
import matplotlib.cm as cm
import csv
import json

import time as _time

def _write_live_status(status_file, label, class_id, frame_sec, counts, video_name):
    """Write real-time traffic status to a JSON file for dashboard polling."""
    status = {
        "label": label,
        "class_id": int(class_id),
        "frame_second": float(frame_sec),
        "video_name": video_name,
        "counts": {"normal": int(counts[0]), "congestion": int(counts[1]), "incident": int(counts[2])},
        "total": int(sum(counts)),
        "timestamp": _time.time(),
    }
    try:
        import portalocker as _pl
        with open(status_file, 'w') as sf:
            _pl.lock(sf, _pl.LOCK_EX)
            json.dump(status, sf)
            _pl.unlock(sf)
    except Exception:
        with open(status_file, 'w') as sf:
            json.dump(status, sf)

from PIL import Image
from edge_communicator import EdgeCommunicator
import torch
import random
from torch.autograd import Variable
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import MaxNLocator
from pipeline_optimizations import cv_render_trajectory, nvdec_video_capture, classify_trajectory_batch_memory
# Device is set inside autoProcessing() to avoid CUDA init before fork
device = None


def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        imgs = os.listdir(txt)
        if('.DS_Store' in imgs):
            imgs.remove('.DS_Store')
        detailedImgs = [(txt + x, 0) for x in imgs]
        # print(detailedImgs[0])
#         fh = open(txt, 'r', encoding='utf-8-sig')
#         imgs = []
#         for line in fh:
#             line = line.strip('\n')
#             line = line.rstrip()
#             words = line.split(',')
# #             print(txt.split('.')[0]+'/'+words[0],int(words[1]))
#             # imgs.append((txt.split('.')[0]+'/'+words[0],int(words[1])))
#             imgs.append((words[0],int(words[1])))
        self.imgs = detailedImgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Get the video duration time
def get_video_duration(filename):
    cap = cv.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num =cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

# def get_video_duration(video_file):
#     with open(video_file,'rb') as fp:
#         data = fp.read()
    
#     index = data.find(b'mvhd') + 4
#     time_scale = struct.unpack('>I', data[index+13:index+13+4])
#     durations  = struct.unpack('>I', data[index+13+4:index+13+4+4])
#     duration = durations[0] / time_scale[0]
#     return duration


def autoProcessing(rootFolder, watchFolder, shutdown_event, outputFile, resFileName,
                    dashboard_url=None):

    # Initialize CUDA device inside the worker process (not at import time)
    # to avoid CUDA re-init errors after multiprocessing fork
    global device
    # Force CPU to avoid CUDA crash in forked worker process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")

    # Set seed to ensure reproductivity
    setup_seed(199)

    # Initialize edge communicator for sending results to remote dashboard
    edge_comm = None
    if dashboard_url:
        edge_comm = EdgeCommunicator(dashboard_url)
        print(f"Edge communicator initialized: {dashboard_url}")

    folderKeyWord = watchFolder.split('/')[-2] + ": "

    # make sure every folder exists
    filePath = rootFolder + ".cacheFiles/"
    # watchFolder = rootFolder + '/watchFolder/'
    # print("Current file path is: %s" % rootFolder)
    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write(folderKeyWord + "Video Processing %s starts! Current file path is: %s\n" % (folderKeyWord, rootFolder))
        portalocker.unlock(file)
    # Prevent to run the darknet executable file for multiple times and save time.
    # darknetResPath = filePath + "darknetRes/uploadFolder/"
    darknetResPathProcessed = filePath + "darknetRes/"
    # os.makedirs(darknetResPath, exist_ok=True)
    os.makedirs(darknetResPathProcessed, exist_ok=True)
    # middleFilePathPre = filePath + "middleFiles/uploadFolder/"
    middleFilePathPreProcessed = filePath + "middleFiles/"
    # os.makedirs(middleFilePathPre, exist_ok=True)
    os.makedirs(middleFilePathPreProcessed, exist_ok=True)
    # datasetFilePathPre = filePath + "dataset/uploadFolder/"
    datasetFilePathPreProcessed = filePath + "dataset/"
    # os.makedirs(datasetFilePathPre, exist_ok=True)
    os.makedirs(datasetFilePathPreProcessed, exist_ok=True)
    modelResFolder = filePath + "/MLresultsFolder/"
    modelResFolderPerVideo = modelResFolder # + "perVideo/"
    # modelResFolderPerRunning = modelResFolder + "/detailedRes/perRunning/"
    os.makedirs(modelResFolder, exist_ok=True)
    os.makedirs(modelResFolderPerVideo, exist_ok=True)
    # os.makedirs(modelResFolderPerRunning, exist_ok=True)

    darknetResPath = darknetResPathProcessed
    middleFilePathPre = middleFilePathPreProcessed
    datasetFilePathPre = datasetFilePathPreProcessed

    # YOLOv4 model files — paths relative to project root for edge deployment
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # Try TensorRT GPU engine first (~100 FPS), fall back to OpenCV DNN CPU
    trt_detector = None
    yolo_net = None
    yolo_output_layers = None
    engine_path = os.path.join(models_dir, "yolov4-custom.engine")
    if os.path.exists(engine_path):
        try:
            from realtime_detector import TRTYoloDetector
            trt_detector = TRTYoloDetector(engine_path, input_size=416, conf_thresh=0.70, nms_thresh=0.4)  # TRT FP16 calibration differs from DNN; 0.70 gives comparable detection counts
            print(folderKeyWord + "YOLOv4 using TensorRT FP16 GPU engine (~100 FPS)")
            with open(outputFile, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                file.write(folderKeyWord + "YOLOv4 using TensorRT FP16 GPU engine\n")
                portalocker.unlock(file)
        except Exception as e:
            print(folderKeyWord + f"TensorRT load failed ({e}), falling back to OpenCV DNN")
            trt_detector = None

    if trt_detector is None:
        yolo_cfg = os.path.join(models_dir, "yolov4-custom.cfg")
        yolo_weights = os.path.join(models_dir, "yolov4-custom_final.weights")
        yolo_net = cv.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
            print(folderKeyWord + "YOLOv4 using OpenCV CUDA FP16 backend")
        else:
            yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            print(folderKeyWord + "YOLOv4 using OpenCV CPU backend (slow)")
        yolo_output_layers = yolo_net.getUnconnectedOutLayersNames()


    # background_direction = 'left' #set as right or left
    # parameters
    # path = 'C:/Users/keerthana/Desktop/yoloV4/RGB_model_phase3_higher_resolution/darknet/test_videos/SVM/'
    # path = 'C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/baili/'

    # videoFolder = 'C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/baili/uploadFolder/'
    # videoFolderProcessed = 'C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/baili/processedFolder/'
    videoFolder = watchFolder
    videoFolderProcessed = rootFolder + 'ProcessedFolder/'
    os.makedirs(videoFolderProcessed, exist_ok=True)

    # Lucas-Kanade Optical Tracker
    lk_params = dict(winSize  = (30, 30),
                    maxLevel = 3,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    detect_interval = 3  # Must match training (step2: DETECT_INTERVAL=3)

    # # Recording Files
    # video_frame = 25
    # video_width = 640
    # video_height = 512
    # speed_background = 2.40 # pixels/frame at 10mph

    # # YouTube Live
    # video_frame = 30
    # video_width = 600
    # video_height = 480
    # speed_background = 2  # 2.40*<recording_frame>/<live_frame>

    # Parameters matching training pipeline (step2_generate_dataset.py)
    video_frame = 25       # 25 FPS thermal camera
    video_width = 640
    video_height = 512
    speed_background = 2.40  # default; overridden per-video below

    # Per-video background speed (from training step2)
    SPEED_BG_DICT = {
        "10_27_22": 3.03,
        "09_19_22": 1.56,
        "09_10_21": 2.78,
        "05_31_22": 3.66,
    }

    trajectory_len = video_frame * 3
    slidingWindow_frame = video_frame
    # Used to prevent unnecessary part of the beginning of the video
    videoStartSkip = {}

    grayfigFlag = False

    # CNNmodelFile = filePath + "bailiRes/dataset/slidingWin_" + str(slidingWindow_frame) + '_trajectoryLen_' + str(trajectory_len) + '.pt'
    # CNNmodelFile = "C:/Users/baili/Documents/GitHub/Drone_project/bailiRes/dataset/slidingWin_" + str(slidingWindow_frame) + '_trajectoryLen_' + str(trajectory_len) + '.pt'
    CNNmodelFile = os.path.join(models_dir, "gray_trajLen_3_model.pth")
    # Load model
    model = torch.jit.load(CNNmodelFile)
    # model = Net()
    # model = torch.load(CNNmodelFile)
    model = model.to(device)
    model.eval()

    # TensorRT CNN classifier (faster than PyTorch)
    trt_cnn = None
    cnn_engine_path = os.path.join(models_dir, "cnn_classifier.engine")
    if os.path.exists(cnn_engine_path):
        try:
            from realtime_detector import TRTCNNClassifier
            import pycuda.driver as cuda
            cuda.init()
            _cnn_cuda_ctx = cuda.Device(0).make_context()
            trt_cnn = TRTCNNClassifier(cnn_engine_path, _cnn_cuda_ctx)
            print(folderKeyWord + "CNN classifier: TensorRT GPU")
        except Exception as e:
            print(folderKeyWord + f"TRT CNN failed ({e}), using PyTorch")
            trt_cnn = None
    # print(model)

    # Begin to process
    while (not shutdown_event.is_set()):

        videoFiles = os.listdir(videoFolder)
        if('.DS_Store' in videoFiles):
            videoFiles.remove('.DS_Store')
        videoFiles = [x for x in videoFiles if(('.mp4' in x) or ('.MP4' in x))]

        # Judge whether any video been uploaded or not
        if len(videoFiles) > 0:
            print(folderKeyWord + "Detected %d video(s)...Processing..." % len(videoFiles))
            with open(outputFile, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                file.write(folderKeyWord + "Detected %d video(s)...Processing...\n" % len(videoFiles))
                portalocker.unlock(file)
            # Process video in the folder one by one
            for original_filename in videoFiles:
                if (shutdown_event.is_set()):
                    break

                video_path = videoFolder + original_filename
                
                videoDuration = get_video_duration(video_path)
                if(videoDuration<((trajectory_len/video_frame)+1.0)):
                    print(folderKeyWord + f'video with name: {original_filename} is too short (less than {trajectory_len/video_frame} seconds) for analyzing')
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + f'video with name: {original_filename} is too short (less than {trajectory_len/video_frame} seconds) to analyze\n')
                        portalocker.unlock(file)
                    shutil.move(videoFolder + original_filename, videoFolderProcessed + original_filename)
                    continue
                # print('video time is: '+ str(videoDuration))
                # sys.exit(0)
                # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
                # startTime = time.perf_counter()
                # darknet processing
                print('#'*20 + original_filename + '#'*20)
                # Direction is the first part for original videos (right-xxx) or last part for stream segments (timestamp-right)
                name_parts = original_filename.split('.')[0].split('-')
                if name_parts[0] in ('left', 'right', 'Omission'):
                    background_direction = name_parts[0]
                elif len(name_parts) > 1 and name_parts[1] in ('left', 'right', 'Omission'):
                    # Training videos: DATE-left-xxx or DATE-right-xxx
                    background_direction = name_parts[1]
                else:
                    background_direction = name_parts[-1] if name_parts[-1] in ("left", "right", "Omission") else "left"
                # background_direction = "left"
                original_filename_split = original_filename.split('.')[0]
                print(folderKeyWord + "Filename: %s with background direction of: %s\n\n" % (original_filename_split, background_direction))
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(folderKeyWord + "Filename: %s with background direction of: %s\n" % (original_filename_split, background_direction))
                    portalocker.unlock(file)
                # original_filename = '2_14_23_I_75_exit_265_3_C'
                # Per-video background speed (matching training data)
                _matched_speed = False
                for date_key, spd in SPEED_BG_DICT.items():
                    if date_key in original_filename_split:
                        speed_background = spd
                        _matched_speed = True
                        break
                if not _matched_speed:
                    speed_background = 2.40  # default
                print(folderKeyWord + f"  speed_background={speed_background}")
                speed_background_left = speed_background - 0.2
                original_coordinates_filename = original_filename_split + '_original'
                formatted_coordinates_filename = original_filename_split + '_formatted'
                # speed_background = 2.40 # pixels/frame at 10mph
                
                # 1. Vehicle detection — real-time YOLOv4 via OpenCV DNN (GPU on Jetson)
                formatted_out_path = darknetResPath + formatted_coordinates_filename + '.txt'

                if trt_detector is not None:
                    # Fast path: TensorRT GPU inference (~100 FPS)
                    import sys; import time as _time; _t_det_start = _time.perf_counter()
                    print(folderKeyWord + 'Running YOLOv4 detection via TensorRT GPU...')
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + 'Running YOLOv4 detection via TensorRT GPU...\n')
                        portalocker.unlock(file)

                    det_cap = cv.VideoCapture(video_path)
                    outFile = open(formatted_out_path, 'w')
                    outFile.write('frame,x,y,width,height' + "\n")
                    det_frame_num = 0
                    while True:
                        ret, det_frame = det_cap.read()
                        if not ret:
                            break
                        det_frame_num += 1
                        if det_frame_num % detect_interval != 0:
                            continue
                        results = trt_detector.detect(det_frame)
                        for (bx, by, bw, bh, conf) in results:
                            outFile.write(f"{det_frame_num}, {bx}, {by}, {bw}, {bh}\n")
                    outFile.close()
                    det_cap.release()

                    _t_det_end = _time.perf_counter()
                    sys.stdout.flush(); print(folderKeyWord + f'YOLOv4 TensorRT detection finished! ({_t_det_end - _t_det_start:.2f}s)')
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + 'YOLOv4 TensorRT detection finished!\n')
                        portalocker.unlock(file)
                else:
                    # Slow path: OpenCV DNN CPU fallback
                    print(folderKeyWord + 'Running YOLOv4 detection via OpenCV DNN (CPU)...')
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + 'Running YOLOv4 detection via OpenCV DNN (CPU)...\n')
                        portalocker.unlock(file)

                    det_cap = cv.VideoCapture(video_path)
                    det_w = int(det_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    det_h = int(det_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    outFile = open(formatted_out_path, 'w')
                    outFile.write('frame,x,y,width,height' + "\n")
                    det_frame_num = 0
                    while True:
                        ret, det_frame = det_cap.read()
                        if not ret:
                            break
                        det_frame_num += 1
                        if det_frame_num % detect_interval != 0:
                            continue
                        blob = cv.dnn.blobFromImage(det_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                        yolo_net.setInput(blob)
                        outputs = yolo_net.forward(yolo_output_layers)
                        boxes = []
                        confidences = []
                        for output in outputs:
                            for detection in output:
                                scores = detection[5:]
                                confidence = scores[0]
                                if confidence >= 0.70:  # TRT FP16 calibration differs from DNN; 0.70 gives comparable detection counts
                                    cx = int(detection[0] * det_w)
                                    cy = int(detection[1] * det_h)
                                    bw = int(detection[2] * det_w)
                                    bh = int(detection[3] * det_h)
                                    bx = cx - bw // 2
                                    by_val = cy - bh // 2
                                    boxes.append([bx, by_val, bw, bh])
                                    confidences.append(float(confidence))
                        if len(boxes) > 0:
                            indices = cv.dnn.NMSBoxes(boxes, confidences, 0.70, 0.4)
                            if len(indices) > 0:
                                indices = indices.flatten()
                                for idx in indices:
                                    bx, by_val, bw, bh = boxes[idx]
                                    outFile.write(f"{det_frame_num}, {bx}, {by_val}, {bw}, {bh}\n")
                    outFile.close()
                    det_cap.release()

                    print(folderKeyWord + 'YOLOv4 detection finished!')
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + 'YOLOv4 detection finished!\n')
                        portalocker.unlock(file)
                
                ######################################################################## Generate trajectories Figure ##############################################################################
                # print('Making middle files folder...\n')
                _t_traj_start = _time.perf_counter()
                print(folderKeyWord + '*'*10 + '   Trajectories generating   ' + '*'*10)
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(folderKeyWord + '*'*10 + '   Trajectories generating   ' + '*'*10 + '\n')
                    portalocker.unlock(file)
                middleFilePath = middleFilePathPre + original_filename_split + "/"
                os.makedirs(middleFilePath, exist_ok=True)
                datasetFilePath = datasetFilePathPre + original_filename_split + "/"
                os.makedirs(datasetFilePath, exist_ok=True)

                # print('Generate trajectory figures...')
                df = pd.read_csv(darknetResPath + formatted_coordinates_filename + '.txt')
                
                # df.isnull().sum()
                df = df[df['frame'].notnull()]
                df = df.astype(int)
                df = df.set_index('frame', drop=True)
                # df.head()

                # parameters setting 
                # read video
                cap = cv.VideoCapture(video_path)
                trajectory_lists = []
                
                # added by Keerthana
                total_count = []
                avg_median_speed = []
                
                # middleFilePathDetailed = middleFilePath + 'detectInterval_' + str(detect_interval) + '-trajLen_' + str(trajectory_len) + '/'
                # print("Improving middle folder: %s" % middleFilePathDetailed)
                # os.makedirs(middleFilePath, exist_ok=True)
                middleFilePathCSV = middleFilePath + 'csv/'
                middleFilePathFIG = middleFilePath + 'fig/'
                middleFilePathGRAY = middleFilePath + 'gray/'
                os.makedirs(middleFilePathCSV, exist_ok=True)
                os.makedirs(middleFilePathFIG, exist_ok=True)
                os.makedirs(middleFilePathGRAY, exist_ok=True)
                figIdx = 0
                _traj_images_mem = []
                _live_counts = [0, 0, 0]
                _interval_buf = []
                _STATUS_INTERVAL = 30  # classify every ~30 trajectory images (~30 seconds)
                _live_status_file = rootFolder + "live_status.json"
                _write_live_status(_live_status_file, "Analyzing...", -1, 0, _live_counts, original_filename_split)
                # processingCount = 0
                
                # print("going to process the video...")
                if(original_filename in videoStartSkip):
                    delaySeconds = videoStartSkip[original_filename]
                else:
                    delaySeconds = 0
                
                while(1):
                    suc, frame = cap.read()
                    if frame is not None:
                        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        img = frame.copy()
                
                        # index of frame 
                        frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
                
                        # # put the frame index on the frame
                        # cv.putText(img,str(round(frame_idx)),
                        #             (50,50),
                        #             cv.FONT_HERSHEY_SIMPLEX, 1,
                        #             (255,0,255),2)
                
                        total_count.append(len(trajectory_lists))

                        if(frame_idx >= (delaySeconds*video_frame)):
                            # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
                            # -- Optical flow tracking (NaN-based, matching step2) --
                            if len(trajectory_lists) > 0 and prev_gray is not None:
                                img0, img1 = prev_gray, frame_gray
                                selected_idx = []
                                selected_last_points = []
                                new_trajectory_lists = []
                                for idx_t, sublist_t in enumerate(trajectory_lists):
                                    if sublist_t and (sublist_t[-1] is not None) and (not pd.isna(sublist_t[-1])):
                                        selected_idx.append(idx_t)
                                        selected_last_points.append(sublist_t[-1])
                                if selected_last_points:
                                    p0 = np.float32(selected_last_points).reshape(-1, 1, 2)
                                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                                    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                                    good = d < 2
                                    hf, wf = img1.shape[:2]
                                    valid = (p1[:,0,0]>=0)&(p1[:,0,0]<wf)&(p1[:,0,1]>=0)&(p1[:,0,1]<hf)
                                    good = good & valid
                                    fill_map = {}
                                    for idx_t, (px,py), gf in zip(selected_idx, p1.reshape(-1,2), good):
                                        fill_map[idx_t] = (px,py) if gf else np.nan
                                    for idx_t in range(len(trajectory_lists)):
                                        sl = trajectory_lists[idx_t]
                                        nv = fill_map.get(idx_t, np.nan)
                                        sl.append(nv)
                                        if not (pd.isna(nv) and len(sl)>=trajectory_len and all(pd.isna(zz) for zz in sl[-trajectory_len:])):
                                            new_trajectory_lists.append(sl)
                                else:
                                    for idx_t in range(len(trajectory_lists)):
                                        sl = trajectory_lists[idx_t]
                                        sl.append(np.nan)
                                        if not (len(sl)>=trajectory_len and all(pd.isna(zz) for zz in sl[-trajectory_len:])):
                                            new_trajectory_lists.append(sl)
                                trajectory_lists = new_trajectory_lists

                    
                    
                            # Update interval - When to update and detect new features
                            # -- Initialize new trajectories at detect_interval (matching step2) --
                            if ((frame_idx-delaySeconds*video_frame) % detect_interval == 0):
                                if frame_idx in df.index:
                                    bb_potential = df.loc[frame_idx].to_numpy()
                                    bb_potential = np.atleast_2d(bb_potential)
                                    if len(trajectory_lists) == 0:
                                        for x,y,w,h in bb_potential:
                                            trajectory_lists.append(
                                                [np.nan] * int(frame_idx) + [(x+w/2, y+h/2)])
                                    else:
                                        selected_last_points = []
                                        for sublist in trajectory_lists:
                                            if sublist and (sublist[-1] is not None) and (not pd.isna(sublist[-1])):
                                                selected_last_points.append(sublist[-1])
                                        if selected_last_points:
                                            for x,y,w,h in bb_potential:
                                                new_traj_flag = True
                                                for x_center, y_center in selected_last_points:
                                                    if (x_center<=x+w+10 and x_center>=x-10 and
                                                            y_center<=y+h+10 and y_center>=y-10):
                                                        new_traj_flag = False
                                                        break
                                                if new_traj_flag:
                                                    trajectory_lists.append(
                                                        [np.nan] * int(frame_idx) + [(x+w/2, y+h/2)])
                                        else:
                                            for x,y,w,h in bb_potential:
                                                trajectory_lists.append(
                                                    [np.nan] * int(frame_idx) + [(x+w/2, y+h/2)])

                            # This part begins to create the trajectories and corresponding trajectory figures.
                            if ((frame_idx-detect_interval-delaySeconds*video_frame) % slidingWindow_frame == 0) and ((frame_idx-detect_interval-delaySeconds*video_frame) >= trajectory_len):
                                # If no cars in this time period, no need to draw
                                
                                # # Determine whether this time step locates in the incident happending period.
                                # if (original_filename in incidentTimePeriodWITHDirection):
                                #     if (math.floor((frame_idx-detect_interval-trajectory_len) / video_frame) >= (incidentTimePeriodWITHDirection[original_filename][1])) or \
                                #     (math.floor((frame_idx-detect_interval) / video_frame) <= (incidentTimePeriodWITHDirection[original_filename][0])):
                                #         incidentFlag = 0
                                #     else:
                                #         incidentFlag = 1
                                # else:
                                #     incidentFlag = 0
                                # print("incidentFlag of %d s is %d" % (math.floor((frame_idx-detect_interval-trajectory_len) / video_frame), incidentFlag))
                                # print("incidentFlag of %d s is %d" % (math.floor((frame_idx-detect_interval) / video_frame), incidentFlag))
                                # print("The section is: %d to %d" % (incidentTimePeriodWITHDirection[original_filename][0], incidentTimePeriodWITHDirection[original_filename][1]))
                            
                                trajectories_df = pd.DataFrame([x[-trajectory_len:] for x in trajectory_lists])
                                trajectories_df.to_csv(middleFilePathCSV + "trajectories_%d_%d_figNum_%d.csv" % (trajectory_len, frame_idx-detect_interval, figIdx))
                                # Begin to draw the figure
                                # plt.figure(figsize=((video_width+speed_background*detect_interval)/100.0, video_height/100.0))
                                selectedTrajectories = []
                                for sublist in [x[-trajectory_len:] for x in trajectory_lists]:
                                    x = []
                                    y = []
                                    count = 0
                                    if len(sublist) >= 0:
                                        # # Drop the trajectories with the car moving from right to left, that move to west
                                        # if (background_direction == "left") and (sublist[-1][0] - sublist[0][0] + abs(speed_background) * (len(trajectory)-1) < 0):
                                        #     continue
                                        # if (background_direction == "right") and (sublist[-1][0] - sublist[0][0] - abs(speed_background) * (len(trajectory)-1) < 0):
                                        #     continue
                                        # Compensate the drone's movement
                                        for ele in sublist:
                                            if isinstance(ele, tuple):
                                                if background_direction == "left":
                                                    x.append(ele[0]-speed_background_left*count)
                                                else:
                                                    x.append(ele[0]-speed_background*count)
                                                y.append(ele[1])
                                            count+=1
                                    # Select vehicles only move to the right direction of the footage or fixed in one point
                                    if len(x) >= 1 and (len(x) == 1 or (x[-1]-x[0] >= 0)):
                                        selectedTrajectories.append([x, y])
                                if(len(selectedTrajectories)>=0):
                                    saveFigName = middleFilePathFIG + "trajectories_%d_%d_figNum_%d.png" % (trajectory_len, frame_idx-detect_interval, figIdx)
                                    # saveFigNameGray = middleFilePathGRAY + "trajectories_%d_%d_figNum_%d_grayed.png" % (trajectory_len, frame_idx-detect_interval, figIdx)
                                    saveDatasetName = datasetFilePath + str(figIdx) + '.png'
                                    # txtResDf.loc[figIdx, 'path'] = saveDatasetName
                                    # txtResDf.loc[figIdx, 'incidentFlag'] = incidentFlag
                                    # Fast OpenCV rendering (replaces matplotlib, ~500x faster)
                                    canvas = cv_render_trajectory(
                                        selectedTrajectories, video_width, video_height,
                                        speed_background, detect_interval, background_direction,
                                        trajectory_len)
                                    cv.imwrite(saveFigName, canvas)
                                    if not grayfigFlag:
                                        cv.imwrite(saveDatasetName, canvas)
                                        # Collect for in-memory classification
                                        if '_traj_images_mem' not in dir():
                                            _traj_images_mem = []
                                        _traj_images_mem.append(canvas.copy())
                                        # -- Buffered interval classification (~15s batches) --
                                        _interval_buf.append((canvas.copy(), img.copy(), figIdx, frame_idx))
                                        if len(_interval_buf) >= _STATUS_INTERVAL:
                                            try:
                                                _buf_imgs = [item[0] for item in _interval_buf]
                                                _buf_preds = classify_trajectory_batch_memory(_buf_imgs, model=model, device=device)
                                                if len(_buf_preds) > 0:
                                                    for _bp in _buf_preds:
                                                        _live_counts[int(_bp)] += 1
                                                    # Majority vote for this interval
                                                    from collections import Counter as _Counter
                                                    _vote = _Counter(_buf_preds.tolist()).most_common(1)[0][0]
                                                    _cls_label = {0: "Normal Traffic Conditions", 1: "Congestion Propagating", 2: "Incident Detected"}[_vote]
                                                    _frame_sec = frame_idx / video_frame
                                                    _write_live_status(_live_status_file, _cls_label, _vote, _frame_sec, _live_counts, original_filename_split)
                                                    # Save all abnormal trajectory + frame pairs (downsampled to 10 at end)
                                                    _live_evidence_dir = rootFolder + "vlm_evidence/" + original_filename_split + "/"
                                                    for _si, _bp_cls in enumerate(_buf_preds):
                                                        if int(_bp_cls) >= 1:
                                                            os.makedirs(_live_evidence_dir, exist_ok=True)
                                                            _b_canvas, _b_frame, _b_fig, _b_fidx = _interval_buf[_si]
                                                            cv.imwrite(os.path.join(_live_evidence_dir, f"trajectory_{int(_bp_cls)}_{_b_fig}.png"), _b_canvas)
                                                            cv.imwrite(os.path.join(_live_evidence_dir, f"frame_{int(_bp_cls)}_{_b_fig}.png"), _b_frame)
                                            except Exception as _rt_err:
                                                print('EVIDENCE ERROR: ' + str(_rt_err))
                                            _interval_buf = []
                                    # Increase the count of whole figures
                                    figIdx += 1
                            
                        # Stream frame to dashboard (~2 FPS) with detection overlays
                        if edge_comm and int(frame_idx) % 15 == 0:
                            try:
                                _viz_frame = img.copy()
                                # Draw YOLO bounding boxes (green)
                                if frame_idx in df.index:
                                    _bbs = np.atleast_2d(df.loc[frame_idx].to_numpy())
                                    for _bx, _by, _bw, _bh in _bbs:
                                        cv.rectangle(_viz_frame, (int(_bx), int(_by)),
                                                     (int(_bx+_bw), int(_by+_bh)), (0, 255, 0), 2)
                                # Draw tracking trajectories (last 30 points per track)
                                for _traj in trajectory_lists:
                                    _pts = [p for p in _traj[-30:] if p is not None and not (isinstance(p, float) and pd.isna(p)) and not (hasattr(p, '__iter__') and any(pd.isna(c) for c in p))]
                                    if len(_pts) >= 2:
                                        for _ti in range(1, len(_pts)):
                                            _p1 = (int(_pts[_ti-1][0]), int(_pts[_ti-1][1]))
                                            _p2 = (int(_pts[_ti][0]), int(_pts[_ti][1]))
                                            cv.line(_viz_frame, _p1, _p2, (0, 255, 255), 2)
                                    # Draw current position (red dot)
                                    if _pts:
                                        _cp = _pts[-1]
                                        cv.circle(_viz_frame, (int(_cp[0]), int(_cp[1])), 4, (0, 0, 255), -1)
                                edge_comm.send_frame(_viz_frame)
                            except Exception:
                                pass

                        # frame_idx += 1
                        prev_gray = frame_gray
                
                
                
                        # # Show Results
                        # img = cv.resize(img, None, fx=0.5, fy=0.5)
                        # ## This is used to show the processed video
                        # cv.imshow('Optical Flow', img) #comment it if dont want to see the frame o/p
                        # ## cv2.imshow('Mask', mask)
                
                        # if cv.waitKey(1) & 0xFF == ord('q'): #change 0 to 1
                        #     break
                    
                    else:
                        # Flush remaining buffered images
                        if _interval_buf:
                            try:
                                _buf_imgs = [item[0] for item in _interval_buf]
                                _buf_preds = classify_trajectory_batch_memory(_buf_imgs, model=model, device=device)
                                if len(_buf_preds) > 0:
                                    for _bp in _buf_preds:
                                        _live_counts[int(_bp)] += 1
                                    from collections import Counter as _Counter
                                    _vote = _Counter(_buf_preds.tolist()).most_common(1)[0][0]
                                    _cls_label = {0: "Normal Traffic Conditions", 1: "Congestion Propagating", 2: "Incident Detected"}[_vote]
                                    _write_live_status(_live_status_file, _cls_label, _vote, frame_idx / video_frame, _live_counts, original_filename_split)
                                    _live_evidence_dir = rootFolder + "vlm_evidence/" + original_filename_split + "/"
                                    for _si, _bp_cls in enumerate(_buf_preds):
                                        if int(_bp_cls) >= 1:
                                            os.makedirs(_live_evidence_dir, exist_ok=True)
                                            _b_canvas, _b_frame, _b_fig, _b_fidx = _interval_buf[_si]
                                            cv.imwrite(os.path.join(_live_evidence_dir, f"trajectory_{int(_bp_cls)}_{_b_fig}.png"), _b_canvas)
                                            cv.imwrite(os.path.join(_live_evidence_dir, f"frame_{int(_bp_cls)}_{_b_fig}.png"), _b_frame)
                            except Exception:
                                pass
                            _interval_buf = []
                        print(folderKeyWord + "end of video: %s" % (original_filename))
                        with open(outputFile, 'a') as file:
                            portalocker.lock(file, portalocker.LOCK_EX)
                            file.write(folderKeyWord + "end of video: %s\n" % (original_filename))
                            portalocker.unlock(file)
                        break
                
                cap.release()
                cv.destroyAllWindows()
                # end of trajectory detection
                # processingCount += 1
                # print('Finnished! Going to the next one\n')
                # time.sleep(1)

                # print("Done with the video: %s" % original_filename)
                # time.sleep(2)
                # # Move video to the processed folder
                # shutil.move(videoFolder + original_filename, videoFolderProcessed + original_filename)
                # shutil.move(darknetResPath + original_coordinates_filename + '.txt', darknetResPathProcessed + original_coordinates_filename + '.txt')
                # shutil.move(darknetResPath + formatted_coordinates_filename + '.txt', darknetResPathProcessed + formatted_coordinates_filename + '.txt')
                # shutil.move(middleFilePathPre + original_filename_split, middleFilePathPreProcessed + original_filename_split)
                # trajectoryTime = time.perf_counter()

                ##########################################################  CNN Model Processing  ####################################################
                _t_traj_end = _time.perf_counter()
                sys.stdout.flush(); print(folderKeyWord + f'Trajectories done in {_t_traj_end - _t_traj_start:.2f}s')
                _t_ml_start = _time.perf_counter()
                print(folderKeyWord + '*'*10 + '   ML model processing   ' + '*'*10)
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(folderKeyWord + '*'*10 + '   ML model processing   ' + '*'*10 + '\n')
                    portalocker.unlock(file)
                # original_filename_split = "left-4_26_23_I_4_at_MLK10"
                # datasetFilePath = "C:/Users/baili/Documents/GitHub/Drone_project/bailiRes/dataset/processedFolder/left-4_26_23_I_4_at_MLK10/"
                # Load dataset
                # testset = torchvision.datasets.ImageFolder(middleFilePathGRAY, transform=transforms.ToTensor(), loader=default_loader)
                testset = MyDataset(txt=datasetFilePath, transform=transforms.Compose([transforms.Resize((56, 56)), transforms.ToTensor()]))
                # img, label = testset.__getitem__(0)
                # print(img, label)
                # print(testset.__len__())
                testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False)

                # model.eval()                                # set model to evalutation mode
                preds = []
                if trt_cnn is not None and _traj_images_mem:
                    # TensorRT GPU CNN: classify from in-memory images (no disk I/O)
                    preds = classify_trajectory_batch_memory(_traj_images_mem, trt_classifier=trt_cnn)
                elif _traj_images_mem:
                    # PyTorch: classify from in-memory images (no disk I/O)
                    preds = classify_trajectory_batch_memory(_traj_images_mem, model=model, device=device)
                else:
                    # Fallback: load from disk (original path)
                    for x, y in testloader:
                        x = Variable(x).to(device)
                        with torch.no_grad():
                            out = model(x)
                            out = out.detach().cpu()
                            pred = torch.max(out, 1)[1]
                            preds.append(pred)
                    if preds:
                        preds = torch.cat(preds, dim=0).numpy()

                # Convert to numpy if needed (TRT/in-memory return numpy already)
                if isinstance(preds, list) and len(preds) == 0:
                    preds = None
                elif not isinstance(preds, np.ndarray) and preds is not None:
                    preds = np.array(preds)

                _t_ml_end = _time.perf_counter()
                sys.stdout.flush(); print(folderKeyWord + f'ML classification done in {_t_ml_end - _t_ml_start:.2f}s')

                if preds is not None and len(preds) > 0:
                    modelPredFile = modelResFolderPerVideo+original_filename_split+'_pred.csv'
                    print(folderKeyWord + 'Saving prediction results to {}'.format(modelPredFile))
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(folderKeyWord + 'Saving prediction results to {}\n'.format(modelPredFile))
                        portalocker.unlock(file)
                    with open(modelPredFile, 'w', newline='', encoding='utf-8') as fp:
                        writer = csv.writer(fp)
                        writer.writerow(['second of video', 'incidentDetected?'])
                        for i, p in enumerate(preds):
                            writer.writerow([i+trajectory_len/video_frame, p])
                    # Judge whether incidents occurred in the video
                    incidentRatio = np.sum(preds == 2) / preds.size
                    congestionRatio = np.sum(preds == 1) / preds.size
                    if incidentRatio >= 0.3:
                        tempWriteStr = original_filename_split + ',' + str(videoDuration) + ',Incident Detected!'
                    elif congestionRatio >= 0.3:
                        tempWriteStr = original_filename_split + ',' + str(videoDuration) + ',Congestion Propagating.'
                    else:
                        tempWriteStr = original_filename_split + ',' + str(videoDuration) + ',Normal Traffic Conditions.'
                else:
                    tempWriteStr = original_filename_split + ',' + str(videoDuration) + ',Normal Traffic Conditions.'
                with open(resFileName, 'a') as tempWriter:
                    portalocker.lock(tempWriter, portalocker.LOCK_EX)
                    tempWriter.write(tempWriteStr + '\n')
                    portalocker.unlock(tempWriter)
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(tempWriteStr + '\n')
                    portalocker.unlock(file)

                # VLM analysis using sampled evidence saved during interval classification
                if preds is not None and len(preds) > 0 and (np.sum(preds == 2) / preds.size >= 0.3 or np.sum(preds == 1) / preds.size >= 0.3):
                    evidence_dir = rootFolder + "vlm_evidence/" + original_filename_split + "/"
                    if os.path.isdir(evidence_dir):
                        try:
                            # Downsample to 10 trajectories + 10 frames for VLM
                            all_trajs = sorted([f for f in os.listdir(evidence_dir) if f.startswith('trajectory_') and f.endswith('.png')])
                            all_frames = sorted([f for f in os.listdir(evidence_dir) if f.startswith('frame_') and f.endswith('.png')])
                            if len(all_trajs) > 10:
                                _keep_step = max(1, len(all_trajs) // 10)
                                keep_trajs = all_trajs[::_keep_step][:10]
                                for f in all_trajs:
                                    if f not in keep_trajs:
                                        os.remove(os.path.join(evidence_dir, f))
                                all_trajs = keep_trajs
                            if len(all_frames) > 10:
                                _keep_step = max(1, len(all_frames) // 10)
                                keep_frames = all_frames[::_keep_step][:10]
                                for f in all_frames:
                                    if f not in keep_frames:
                                        os.remove(os.path.join(evidence_dir, f))
                                all_frames = keep_frames
                            manifest = {
                                'video_name': original_filename_split,
                                'video_path': video_path,
                                'incident_ratio': float(np.sum(preds == 2) / preds.size),
                                'congestion_ratio': float(np.sum(preds == 1) / preds.size),
                                'total_predictions': int(preds.size),
                                'incident_count': int(np.sum(preds == 2)),
                                'evidence_frames': all_frames,
                                'trajectory_images': all_trajs,
                                'timestamp': datetime.now().isoformat(),
                            }
                            with open(os.path.join(evidence_dir, "manifest.json"), 'w') as mf:
                                json.dump(manifest, mf, indent=2)

                            print(folderKeyWord + f'VLM evidence: {len(all_trajs)} trajectories + {len(all_frames)} frames')
                            with open(outputFile, 'a') as file:
                                portalocker.lock(file, portalocker.LOCK_EX)
                                file.write(folderKeyWord + f'VLM evidence saved to {evidence_dir}\n')
                                portalocker.unlock(file)

                            # Auto-trigger VLM analysis in background thread
                            import threading as _threading
                            from vlm_analysis import analyze_incident as _analyze, save_results as _save_vlm
                            def _auto_vlm(edir, vname, rdir):
                                try:
                                    imgs = [os.path.join(edir, f) for f in os.listdir(edir)
                                            if f.endswith(('.png', '.jpg'))]
                                    if imgs:
                                        print('VLM auto: analyzing ' + str(len(imgs)) + ' images for ' + vname)
                                        results = _analyze(imgs)
                                        _save_vlm(results, rdir, vname)
                                        for r in results:
                                            s = 'OK' if r.get('report') else 'err=' + str(r.get('error',''))
                                            print('VLM auto [' + r['provider'] + ']: ' + s)
                                except Exception as ve:
                                    print('VLM auto error: ' + str(ve))
                            _vlm_out = os.path.join(rootFolder, 'vlm_reports')
                            _threading.Thread(target=_auto_vlm, args=(evidence_dir, original_filename_split, _vlm_out), daemon=True).start()
                        except Exception as vlm_err:
                            print(folderKeyWord + f'VLM evidence save error: {vlm_err}')

                # Send detection result + evidence to remote dashboard
                if edge_comm:
                    try:
                        manifest_data = None
                        manifest_file = os.path.join(evidence_dir, "manifest.json") if 'evidence_dir' in dir() else None
                        if manifest_file and os.path.exists(manifest_file):
                            with open(manifest_file, 'r') as mf:
                                manifest_data = json.load(mf)
                        sent = edge_comm.send_detection_result(
                            video_name=original_filename_split,
                            duration=videoDuration,
                            detection_label=tempWriteStr.split(',')[-1],
                            evidence_dir=evidence_dir if 'evidence_dir' in dir() else None,
                            manifest=manifest_data,
                        )
                        if sent:
                            print(folderKeyWord + 'Detection result sent to dashboard')
                            with open(outputFile, 'a') as file:
                                portalocker.lock(file, portalocker.LOCK_EX)
                                file.write(folderKeyWord + 'Detection result sent to remote dashboard\n')
                                portalocker.unlock(file)
                    except Exception as comm_err:
                        print(folderKeyWord + f'Dashboard communication error: {comm_err}')

                # Move video to the processed folder
                shutil.move(videoFolder + original_filename, videoFolderProcessed + original_filename)
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(f"Video Processing {folderKeyWord} {original_filename} ends, move video to processed folder!\n")
                    portalocker.unlock(file)

                # try:
                #     childPipe.send(tempWriteStr)
                # except Exception as err:
                #     """ 当out_pipe接受不到输出的时候且输入被关闭的时候，会抛出EORFError，可以捕获并且退出子进程 """
                #     print("Video processing process cannot send results back because of bad pipes\n")
                #     with open(outputFile, 'a') as file:
                #         portalocker.lock(file, portalocker.LOCK_EX)
                #         file.write(f"Video processing process cannot send results back because of {err}\n")
                #         portalocker.unlock(file)
                #     processingFlag = False
                #     childPipe.close()
                #     break

                # print('*'*10 + '   Time elasped   ' + '*'*10)
                # darknetDuration = round((darknetTime - startTime), 2)
                # trajectoriesDuration = round((trajectoryTime - darknetTime), 2)
                # machineLearningDuration = round((modelTime - trajectoryTime), 2)
                # totalProcessingDuration = darknetDuration+trajectoriesDuration+machineLearningDuration
                # print('darknet: %.2f seconds' % (darknetDuration))
                # print('trajectoriesGenerating: %.2f seconds' % (trajectoriesDuration))
                # print('machineLearningModel: %.2f seconds' % (machineLearningDuration))
                # print('#'*20 + '   Done   ' + '#'*20)

                ################################################ Save processing time ################################################
                # processTimePath = modelResFolder + 'processTimeRecord.csv'
                # if(not os.path.isfile(processTimePath)):
                #     with open(processTimePath, 'w', newline='', encoding='utf-8') as fp:
                #         writer = csv.writer(fp)
                #         writer.writerow(['ProcessingDatetime', 'videoName', 'darknetTime (s)', 'trajectoriesTime (s)', 'machineLearningTime (s)', 'TotalProcessingTime (s)', 'videoDuration (s)', 'processingDuration/videoDuration', 'darknetPercentage', 'trajectoriesPercentage', 'machingLearningPercentage'])
                # with open(processTimePath, 'a', newline='', encoding='utf-8') as fp:
                #     writer = csv.writer(fp)
                #     writer.writerow([TIMESTAMP, original_filename_split, darknetDuration, trajectoriesDuration, machineLearningDuration, totalProcessingDuration, videoDuration, round((totalProcessingDuration/videoDuration), 2), round(darknetDuration/videoDuration*100, 2), round(trajectoriesDuration/videoDuration*100, 2), round(machineLearningDuration/videoDuration*100, 2)])
                # print('#'*20 + '   Writing Duration to File Done   ' + '#'*20)

                # sys.exit(0)
                # print(folderKeyWord + 'Video processing process Listening...')
                # with open(outputFile, 'a') as file:
                #     portalocker.lock(file, portalocker.LOCK_EX)
                #     file.write(folderKeyWord + 'Video processing process Listening...\n')
                #     portalocker.unlock(file)
                
                if(shutdown_event.is_set()):
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(f"Video Processing {folderKeyWord} ends!\n")
                        portalocker.unlock(file)
                    break

        else:
            # print(folderKeyWord + 'Video processing process Listening...')
            # with open(outputFile, 'a') as file:
            #     portalocker.lock(file, portalocker.LOCK_EX)
            #     file.write(folderKeyWord + 'Video processing process Listening...\n')
            #     portalocker.unlock(file)
            if(shutdown_event.is_set()):
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(f"Video Processing {folderKeyWord} ends!\n")
                    portalocker.unlock(file)
                break
            
        time.sleep(3)
            