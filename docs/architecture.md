# Pipeline Architecture

End-to-end data flow from raw KITTI files to detection metrics and tracked outputs.

```mermaid
flowchart TD
    %% ── Inputs ──────────────────────────────────────────────────────────
    subgraph IN[" Inputs "]
        IMG["📷 image_2/&lt;scene&gt;.png\nKITTI left camera image\n1242 × 375 px"]
        BIN["📡 velodyne/&lt;scene&gt;.bin\nVelodyne HDL-64E point cloud\nN × 4  [x, y, z, intensity]"]
        CAL["📄 calib/&lt;scene&gt;.txt\nCalibration matrices\nP0–P3, R0_rect, Tr_velo_to_cam"]
        LBL["🏷️ label_2/&lt;scene&gt;.txt\nGround-truth 2D+3D boxes\nCar / Pedestrian / Cyclist"]
    end

    %% ── Stage 1 : Calibration ───────────────────────────────────────────
    subgraph S1["Stage 1 · KITTICalibration  [src/calibration.py]"]
        C1["Parse P2, R0_rect, Tr_velo_to_cam"]
        C2["Pre-compute 3×4 projection matrix\nP2 × R0_rect_4x4 × Tr_velo_to_cam_4x4"]
        C3["lidar_to_image() · project_box_3d_to_image()"]
        C1 --> C2 --> C3
    end

    %% ── Stage 2 : Camera detector ───────────────────────────────────────
    subgraph S2["Stage 2 · CameraDetector  [src/camera_detector.py]"]
        D1["YOLOv8n inference\nconf ≥ 0.30"]
        D2["COCO class filter\ncar · bus · truck · person · bicycle · motorcycle"]
        D3["Detection2D list\nbbox [x1,y1,x2,y2] · confidence · class_id · source='camera'"]
        D1 --> D2 --> D3
    end

    %% ── Stage 3 : LiDAR detector ────────────────────────────────────────
    subgraph S3["Stage 3 · StubLidarDetector  [src/lidar_detector.py]"]
        L1["Read GT label file\nparse dims_lwh · loc_cam · rotation_y"]
        L2["Apply noise model\n10% dropout · Beta(8,2) confidence · bbox jitter σ=3px"]
        L3["Camera→LiDAR frame\nyaw = −rotation_y − π/2  ·  R.T @ (loc − t)"]
        L4["Project 8 box corners to image\nvia KITTICalibration"]
        L5["Detection2D list\nbbox · confidence · center_3d · dims_lwh · yaw · corners_2d"]
        L1 --> L2 --> L3 --> L4 --> L5
    end

    %% ── Stage 4 : Fusion ────────────────────────────────────────────────
    subgraph S4["Stage 4 · Fusion  [src/fusion.py]"]
        F1["Map to unified taxonomy\nCar=0  Pedestrian=1  Cyclist=2"]
        F2["Per-class 2D IoU matrix\nM_camera × N_lidar"]
        F3["Hungarian assignment\nscipy linear_sum_assignment(−IoU)"]
        F4["IoU ≥ 0.30 → fuse pair\nbbox = 0.4×cam + 0.6×lid\nconfidence = 0.4×cam + 0.6×lid"]
        F5["Unmatched survive if conf ≥ 0.20"]
        F6["FusedDetection list\nsource: fused / camera_only / lidar_only"]
        F1 --> F2 --> F3 --> F4 --> F5 --> F6
    end

    %% ── Stage 5 : Tracker ───────────────────────────────────────────────
    subgraph S5["Stage 5 · MultiObjectTracker  [src/tracker.py]"]
        K1["Predict all tracks\nstate = F × state   [x, y, vx, vy]"]
        K2["Euclidean BEV distance matrix\n+ Hungarian match  threshold=3 m"]
        K3["Kalman update matched tracks\ninnovation · gain · covariance update"]
        K4["Birth unmatched detections\nCoast / kill old tracks  max_age=5"]
        K5["KalmanTracker list\nID · position · velocity · speed · color · trail"]
        K1 --> K2 --> K3 --> K4 --> K5
    end

    %% ── Stage 6 : Visualizer ────────────────────────────────────────────
    subgraph S6["Stage 6 · Visualizer  [src/visualizer.py]"]
        V1["render_inputs()\ncamera=blue  lidar=green  raw detections"]
        V2["render_fused()\nfused=red  cam-only=blue  lid-only=green\n+ 3D wireframe corners"]
        V3["render_lidar_on_image()\ndepth-coloured Velodyne points on photo"]
        V4["render_bev_tracks()\nBEV + point cloud + trails + velocity arrows"]
    end

    %% ── Stage 7 : Evaluator ─────────────────────────────────────────────
    subgraph S7["Stage 7 · Evaluator  [src/evaluator.py  +  run_sequence_tracker.py]"]
        E1["KITTIEvaluator\ngreedy IoU≥0.5 matching\n11-point interpolated AP per class"]
        E2["compute_mota()\nMOTA = 1 − (FP+FN+IDS) / GT\nID-switch tracking via gt→pred map"]
        E3["AP table\nCamera · LiDAR · Fused  ×  3 classes\nmAP: 0.391 / 0.808 / 0.848"]
        E4["MOTA score\n0.24 default  →  0.62 no-dropout\nFP · FN · IDS breakdown"]
    end

    %% ── Edges ───────────────────────────────────────────────────────────
    CAL --> S1
    IMG --> S2
    BIN --> S3
    LBL --> S3
    LBL --> S7

    S1 -->|"lidar_to_image\nproject_box_3d_to_image"| S3
    S1 -->|"calib for 3D projection"| S6

    S2 -->|"Detection2D [camera]"| S4
    S3 -->|"Detection2D [lidar]"| S4

    S4 -->|"FusedDetection list"| S5
    S4 -->|"fused / cam-only / lid-only"| S6
    S4 -->|"camera · lidar · fused dets"| S7

    S5 -->|"active KalmanTracker list"| S6
    S5 -->|"pred_by_frame for MOTA"| S7

    S6 --> V1 & V2 & V3 & V4
    S7 --> E1 & E2 --> E3 & E4
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Late fusion** (detection-level) | Each sensor's detector runs independently — swap any detector without touching fusion or evaluation code |
| **Class-aware IoU matching** | Prevents a camera pedestrian matching a LiDAR car even when boxes overlap |
| **LiDAR confidence weighted 0.6 vs camera 0.4** | LiDAR gives metric-accurate 3D geometry; camera gives semantic richness |
| **Kalman state [x, y, vx, vy]** | Constant-velocity model is sufficient at 10 Hz; higher-order models overfit noise |
| **Stub detector with GT + noise** | Validates fusion and tracking math without a GPU-dependent PointPillars model |
| **Camera-only dets included in MOTA pred** | Tracker skips camera-only (no center_3d); including them in pred avoids phantom FN |
