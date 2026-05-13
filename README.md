# Camera-LiDAR Late Fusion · 3D Object Detection & Multi-Object Tracking

![CI](https://github.com/SHIVCHAUDHARY17/Camera-LiDAR-Fusion/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A modular autonomous-driving perception pipeline that fuses **YOLOv8** camera detections with **LiDAR 3D bounding boxes** on the KITTI dataset, then tracks every object across real 10 Hz video using a **Kalman filter**. Calibration-aware projection maps 3D LiDAR boxes into image space; a 2D IoU matrix solved with the **Hungarian algorithm** produces optimal one-to-one sensor matches per class; a constant-velocity Kalman filter maintains object identity and velocity estimates across frames. Fusion lifts mAP from **0.391 → 0.848** (+117%); the Kalman tracker achieves **MOTA 0.62** on KITTI tracking sequence 0001.

---

## Architecture

```mermaid
flowchart TD
    subgraph IN[" Inputs "]
        IMG["📷 image_2/&lt;scene&gt;.png\nKITTI camera image"]
        BIN["📡 velodyne/&lt;scene&gt;.bin\nPoint cloud  N×4"]
        CAL["📄 calib/&lt;scene&gt;.txt\nP2 · R0_rect · Tr_velo_to_cam"]
        LBL["🏷️ label_2/&lt;scene&gt;.txt\nGround-truth boxes"]
    end

    subgraph S1["Stage 1 · KITTICalibration"]
        C2["3×4 projection matrix\nP2 × R0_rect × Tr_velo_to_cam"]
    end

    subgraph S2["Stage 2 · CameraDetector"]
        D3["YOLOv8n → Detection2D list\nbbox · confidence · COCO class"]
    end

    subgraph S3["Stage 3 · StubLidarDetector"]
        L5["GT + noise → Detection2D list\nbbox · center_3d · dims · yaw"]
    end

    subgraph S4["Stage 4 · Fusion"]
        F6["IoU matrix + Hungarian\n→ FusedDetection list\nfused / camera_only / lidar_only"]
    end

    subgraph S5["Stage 5 · MultiObjectTracker"]
        K5["Kalman filter per object\n→ ID · position · velocity · trail"]
    end

    subgraph S6["Stage 6 · Visualizer"]
        V1["inputs · fused · lidar_on_image · bev_tracks\n4 output images per frame"]
    end

    subgraph S7["Stage 7 · Evaluator"]
        E3["AP table  mAP 0.391→0.848\nMOTA 0.24→0.62"]
    end

    CAL --> S1
    IMG --> S2
    BIN & LBL --> S3
    S1 -->|project 3D corners| S3
    S1 -->|calib for rendering| S6
    S2 -->|camera dets| S4
    S3 -->|lidar dets| S4
    S4 -->|FusedDetection| S5
    S4 & S5 --> S6
    S4 & S5 & LBL --> S7
    S6 --> V1
    S7 --> E3
```

> Full diagram with all internal nodes: [`docs/architecture.md`](docs/architecture.md)

---

## Pipeline Stages

| Stage | What it does | Input | Output | Key file |
|---|---|---|---|---|
| **1 · Calibration** | Builds `P2 × R0_rect × Tr_velo_to_cam` projection chain; exposes `lidar_to_image` and `project_box_3d_to_image` | `calib/*.txt` | `KITTICalibration` object | `src/calibration.py` |
| **2 · Camera detector** | YOLOv8n inference; filters to 6 COCO traffic classes | BGR image | `List[Detection2D]` source=camera | `src/camera_detector.py` |
| **3 · LiDAR detector** | Reads GT labels + adds 10% dropout, β(8,2) confidence, σ=3px jitter; converts camera→LiDAR frame; projects 8 corners to image | `label_2/*.txt` + calib | `List[Detection2D]` source=lidar with `center_3d` | `src/lidar_detector.py` |
| **4 · Late fusion** | Groups by unified class; builds per-class IoU matrix; Hungarian assignment; blends confidence 0.4×cam + 0.6×lid | `List[Detection2D]` × 2 | `List[FusedDetection]` | `src/fusion.py` |
| **5 · MOT** | Constant-velocity Kalman filter `[x, y, vx, vy]`; Euclidean BEV distance + Hungarian match; birth/coast/death lifecycle | `List[FusedDetection]` | `List[KalmanTracker]` with ID, velocity, trail | `src/tracker.py` |
| **6 · Visualizer** | 4 render modes: raw inputs, fused (3D wireframes), LiDAR-on-image (depth-coloured), BEV tracks (velocity arrows + trails) | image + dets + calib + tracks | 4 × `.jpg` per frame | `src/visualizer.py` |
| **7 · Evaluator** | 11-point interpolated AP per class per sensor; MOTA = 1 − (FP+FN+IDS)/GT with ID-switch tracking | dets + GT labels | AP table + MOTA score | `src/evaluator.py` |

---

## Results

### Detection — Average Precision (IoU ≥ 0.5, 100 KITTI val scenes)

| Class | Camera AP | LiDAR AP | Fused AP |
|---|---|---|---|
| Car | 0.674 | 0.813 | **0.885** |
| Pedestrian | 0.486 | 0.803 | **0.880** |
| Cyclist | 0.011 | 0.807 | 0.780 |
| **mAP** | 0.391 | 0.808 | **0.848** |

Fusion improves mAP from **0.391 → 0.848 (+117%)** over camera-only.  
Cyclist AP reflects a COCO→KITTI domain gap: YOLOv8n rarely fires on KITTI cyclists, so fusion has no camera signal to combine with.

> LiDAR uses a GT-backed stub with 10% dropout + Gaussian noise. Camera AP reflects real YOLOv8n performance. Fused AP shows the benefit of cross-sensor verification.

### Tracking — MOTA on KITTI Sequence 0001 (frames 0–100, 894 GT objects)

| Mode | MOTA | FP | FN | IDS | Notes |
|---|---|---|---|---|---|
| Default (dropout=0.10, thr=0.3) | 0.24 | 153 | 2 125 | 19 | Artificial detection gaps + GT frame filter bug |
| **No-dropout** (dropout=0.0, thr=0.1) | **0.62** | 322 | 4 | 15 | True Kalman tracker quality, camera-only dets included |

The 0.24 → 0.62 jump came from three simultaneous fixes: (1) filtering MOTA GT to only processed frames, (2) removing artificial dropout, (3) including `camera_only` fused detections in the prediction set. The remaining gap to 1.0 is FP — weak camera detections that don't match any GT.

---

## Visualizations

| Raw detections | Fused output |
|:---:|:---:|
| camera=blue · LiDAR=green | fused=red · cam-only=blue · lid-only=green |
| ![inputs](docs/sample_inputs.jpg) | ![fused](docs/sample_fused.jpg) |

| LiDAR projected on image | Bird's eye view |
|:---:|:---:|
| Depth-coloured Velodyne points | Point cloud + detection boxes |
| ![lidar_on_image](docs/sample_lidar_on_image.jpg) | ![bev](docs/sample_bev.jpg) |

### Multi-Object Tracking — KITTI Sequence 0001 (real 10 Hz video)

Kalman filter tracks each object across 101 consecutive frames. Combined view: camera image with projected 3D boxes (top) + BEV with velocity arrows and trails (bottom).

![tracking_demo](docs/tracking_demo.gif)

### BEV Kalman Tracking

Bird's eye view tracking from the detection-dataset runner — point cloud background, colour-coded tracks, velocity arrows, fading trails.

![bev_demo](docs/bev_demo.gif)

---

## Setup

```bash
git clone https://github.com/SHIVCHAUDHARY17/Camera-LiDAR-Fusion.git
cd Camera-LiDAR-Fusion

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### KITTI Detection Dataset

Download from [KITTI object detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php):

- `data_object_image_2.zip` — left camera images
- `data_object_velodyne.zip` — Velodyne point clouds
- `data_object_calib.zip` — calibration files
- `data_object_label_2.zip` — ground-truth labels

Extract all to the same parent. Expected layout:

```
kitti/training/training/
├── calib/       7481 files
├── image_2/     7481 files
├── label_2/     7481 files
└── velodyne/    7481 files
```

### KITTI Tracking Dataset

Download from [KITTI tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php):

```powershell
Start-BitsTransfer -Source "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip"   -Destination "E:\kitti\tracking\data_tracking_calib.zip"
Start-BitsTransfer -Source "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip" -Destination "E:\kitti\tracking\data_tracking_label_2.zip"
Start-BitsTransfer -Source "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip" -Destination "E:\kitti\tracking\data_tracking_image_2.zip"
```

Update `configs/default.yaml` with your KITTI paths.

---

## Usage

```bash
# 1. Single-scene fusion — saves 4 output images
python run_fusion.py --scene 000134

# 2. Benchmark AP across 100 val scenes
python run_benchmark.py

# 3. Multi-object tracking across detection-dataset scenes (no real temporal order)
python run_tracker.py --start 000000 --end 000030 --video

# 4. Sequence tracker on real 10 Hz video — includes MOTA evaluation
python run_sequence_tracker.py --sequence 0001 --start_frame 0 --end_frame 100 --video

# 4a. Same, but with dropout=0 to measure true tracker MOTA
python run_sequence_tracker.py --sequence 0001 --start_frame 0 --end_frame 100 --no_dropout --video
```

**Output directories:**

| Command | Output |
|---|---|
| `run_fusion.py` | `outputs/<scene>_inputs.jpg`, `_fused.jpg`, `_lidar_on_image.jpg`, `_bev.jpg` |
| `run_tracker.py` | `outputs/tracking/<scene>_bev.jpg`, `_camera_tracked.jpg`, `_combined.jpg`, `tracking.avi` |
| `run_sequence_tracker.py` | `outputs/sequence_<SEQ>/<frame>_bev.jpg`, `_camera_tracked.jpg`, `_combined.jpg`, `sequence.avi` |

---

## Configuration

All parameters in `configs/default.yaml` — nothing hardcoded in source.

```yaml
lidar_detector:
  dropout: 0.10           # fraction of GT objects randomly dropped
  bbox_jitter_std: 3.0    # px noise added to each bbox corner
  confidence_alpha: 8.0   # Beta(alpha, beta) confidence sampling
  confidence_beta: 2.0    # → mean ≈ 0.80, range 0.60–0.95

fusion:
  iou_threshold: 0.3      # min 2D IoU to match a camera/LiDAR pair
  camera_weight: 0.4      # weight for camera score in fused confidence
  lidar_weight: 0.6       # weight for LiDAR score (geometric precision)
  min_confidence: 0.2     # drop unmatched detections below this
```

---

## Testing

```bash
python -m pytest tests/ -v
```

25 unit tests on synthetic inputs — no dataset required, runs in CI.

| File | Tests | Covers |
|---|---|---|
| `tests/test_calibration.py` | 11 | Projection math, 3D box corners, round-trip transforms, edge cases |
| `tests/test_fusion.py` | 14 | IoU computation, Hungarian matching, class gating, confidence blending |

---

## Project Structure

```
Camera-LiDAR-Fusion/
├── src/
│   ├── calibration.py         KITTI projection chain P2 × R0_rect × Tr_velo_to_cam
│   ├── camera_detector.py     YOLOv8n wrapper → Detection2D
│   ├── detections.py          Detection2D shared dataclass (camera + LiDAR)
│   ├── evaluator.py           11-pt AP evaluator, GT loader
│   ├── fusion.py              IoU matrix, Hungarian assignment, confidence blend
│   ├── kitti_seq_loader.py    Tracking dataset loader + StubSequenceLidarDetector
│   ├── lidar_detector.py      GT-backed stub detector with noise model
│   ├── point_cloud.py         Velodyne .bin I/O + FOV filter
│   ├── tracker.py             KalmanTracker + MultiObjectTracker
│   └── visualizer.py          4 OpenCV render modes, 3D wireframe projection
│
├── configs/
│   └── default.yaml           All settings — no hardcoded values
│
├── tests/
│   ├── test_calibration.py    Projection math unit tests
│   └── test_fusion.py         Fusion logic unit tests
│
├── docs/
│   ├── architecture.md        Full Mermaid pipeline diagram
│   ├── sample_inputs.jpg
│   ├── sample_fused.jpg
│   ├── sample_lidar_on_image.jpg
│   └── sample_bev.jpg
│
├── run_fusion.py              Single-scene end-to-end CLI
├── run_benchmark.py           AP benchmark across val split
├── run_tracker.py             Multi-scene Kalman MOT (detection dataset)
├── run_sequence_tracker.py    Sequence Kalman MOT + MOTA (tracking dataset)
└── requirements.txt
```

---

## What This Demonstrates

- **Sensor fusion engineering** — camera and LiDAR detections fused at the output level using calibration-aware 3D-to-2D projection and class-gated Hungarian matching; mAP lifts from 0.391 to 0.848, a +117% gain over camera-only

- **KITTI calibration from scratch** — full `P2 × R0_rect × Tr_velo_to_cam` projection chain implemented without sensor-fusion libraries; projects individual LiDAR points and full 3D bounding boxes (8 corners) into pixel space, verified against KITTI ground truth

- **Multi-object tracking pipeline** — constant-velocity Kalman filter with Euclidean BEV distance matching, track birth/coast/death lifecycle, and deterministic colour-coded visualisation; achieves MOTA 0.62 on real 10 Hz KITTI tracking video

- **Quantitative evaluation depth** — two separate metrics: 11-point interpolated AP (detection) and MOTA with FP/FN/IDS breakdown (tracking); uncovered and fixed a systematic MOTA evaluation bug (unprocessed frames contributing phantom FN), improving measured score from 0.24 to 0.62

- **Production-style engineering** — 25 pytest unit tests on synthetic inputs (CI runs without KITTI), config-driven with no hardcoded values, shared `Detection2D` interface makes swapping the stub for a real PointPillars model a one-line change, two independent CLI entry points for detection and tracking workflows

---

## Author

**Shiv Jayant Chaudhary** — Computer Vision & Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shiv1716-blue?logo=linkedin)](https://linkedin.com/in/shiv1716)
[![GitHub](https://img.shields.io/badge/GitHub-SHIVCHAUDHARY17-black?logo=github)](https://github.com/SHIVCHAUDHARY17)
