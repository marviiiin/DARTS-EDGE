# DARTS-Edge: Real-Time Drone-Assisted Traffic Incident Detection and VLM Reporting on Edge Device

[![Paper](https://img.shields.io/badge/Paper-WACV%202027-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/marviiiin/DARTS-thermal)
[![Original DARTS](https://img.shields.io/badge/Original-DARTS-green)](https://github.com/marviiiin/DARTS-project)

## Overview

DARTS-Edge is an edge-computing implementation of the [DARTS](https://github.com/marviiiin/DARTS-project) drone-based traffic incident detection system. It runs the complete detection pipeline on an NVIDIA Jetson Orin Nano (8 GB, 15 W) and generates natural-language incident reports using Vision-Language Models (VLMs).

**Key Results:**
- 30 FPS end-to-end real-time inference at 5.7 W average power
- YOLOv4 F1 = 0.982 with zero FP16 accuracy degradation
- TCD-Net 100% FP16/FP32 agreement
- ~15 hours battery life on 86.4 Wh power bank
- VLM incident reports via Claude Sonnet 4.6 and GPT-4o

## System Architecture

```
Drone (Autel EVO II) → Live Deck → HDMI Capture → Jetson Orin Nano
                                                      │
                                    ┌─────────────────┴──────────────────┐
                                    │  YOLOv4 (TensorRT FP16, 107 FPS)  │
                                    │  Lucas-Kanade Optical Flow (CPU)   │
                                    │  Trajectory Image Generation       │
                                    │  TCD-Net Classification (TRT FP16) │
                                    │  Decision Logic (τ_I≥0.3, τ_N≥0.7)│
                                    │  VLM Incident Reporting            │
                                    └─────────────────┬──────────────────┘
                                                      │
                                              Web Dashboard (Flask)
```

## Hardware Requirements

- NVIDIA Jetson Orin Nano 9 (8 GB) with JetPack 6.2+
- USB HDMI capture card (640×512 thermal input)
- Autel EVO II Enterprise drone with 640T thermal sensor
- Anker 24,000 mAh USB-C power bank
- Wi-Fi antenna for dashboard access

## Installation

```bash
# Clone the repository
git clone https://github.com/marviiiin/DARTS-Edge.git
cd DARTS-Edge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Convert models to TensorRT (on Jetson only)
python scripts/darknet_to_onnx.py --cfg configs/yolov4-custom.cfg --weights models/yolov4-custom_final.weights --output models/yolov4-custom.onnx
trtexec --onnx=models/yolov4-custom.onnx --saveEngine=models/yolov4-custom.engine --fp16
python scripts/build_cnn_engine_v2.py
```

## Usage

```bash
# Start the detection dashboard
./start.sh

# Or run manually
python src/app.py
```

Access the dashboard at `http://<jetson-ip>:5000`

## Project Structure

```
DARTS-Edge/
├── src/                    # Core application code
│   ├── app.py              # Flask web dashboard
│   ├── realtime_detector.py # Real-time detection pipeline
│   ├── trt_inference.py    # TensorRT inference engine
│   ├── vlm_analysis.py     # VLM incident report generation
│   ├── bailiAPLibWeb.py    # DARTS detection library (edge-adapted)
│   ├── gpsFetchLib.py      # GPS coordinate processing
│   ├── edge_communicator.py # Edge device communication
│   ├── pipeline_optimizations.py # Pipeline optimization utilities
│   └── streamCapture*.py   # Video stream capture modules
├── scripts/                # Model conversion & utility scripts
│   ├── darknet_to_onnx.py  # YOLOv4 Darknet → ONNX conversion
│   ├── build_cnn_engine.py # TCD-Net → TensorRT conversion
│   └── api_helper.sh       # VLM API configuration
├── evaluation/             # Evaluation & benchmarking scripts
│   ├── eval_yolo_accuracy.py
│   ├── eval_tcdnet_fp16.py
│   ├── p0_latency_fps_benchmark.py
│   ├── p0_power_benchmark.py
│   └── *.json              # Benchmark results
├── configs/                # Model configurations
│   ├── yolov4-custom.cfg
│   └── obj.names
├── models/                 # Model weights (download separately)
├── templates/              # Dashboard HTML templates
├── static/                 # Dashboard static assets
├── start.sh                # Startup script
└── requirements.txt
```

## Model Weights

Download the pre-trained model weights and place them in the `models/` directory:

| Model | File | Size | Description |
|-------|------|------|-------------|
| YOLOv4 | `yolov4-custom_final.weights` | 245 MB | Custom thermal-trained Darknet weights |
| YOLOv4 TRT | `yolov4-custom.engine` | 126 MB | TensorRT FP16 engine (Orin Nano) |
| TCD-Net | `gray_trajLen_3_model.pth` | 15 MB | PyTorch classification model |
| TCD-Net TRT | `cnn_classifier.engine` | 7.5 MB | TensorRT FP16 engine (Orin Nano) |

> **Note:** TensorRT engines are architecture-specific. You must rebuild `.engine` files on your target Jetson device using the provided conversion scripts.

## Dataset

The drone-based thermal traffic monitoring dataset is available on HuggingFace:
[https://huggingface.co/datasets/marviiiin/DARTS-thermal](https://huggingface.co/datasets/marviiiin/DARTS-thermal)

## Citation

```bibtex
@inproceedings{oseikuffour2027dartsedge,
  title={DARTS-Edge: Real-Time Drone-Assisted Traffic Incident Detection and VLM Reporting on Edge Device},
  author={Osei-Kuffour, Marvin and Zhang, Yu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```

## Acknowledgments

This research was supported by the National Institute for Congestion Reduction (NICR), a U.S. DOT University Transportation Center.

## License

MIT License
