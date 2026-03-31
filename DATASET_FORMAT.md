# Dataset Format Reference

## 1. Custom (Scala2) Source Format

### Point Cloud — `.bin` files
Binary float32, row-major, shape `(N, 4)`:

| Col | Field     | Axis        | Typical range |
|-----|-----------|-------------|---------------|
| 0   | `x_l`     | Forward     | 0 – 50 m      |
| 1   | `y_l`     | Left        | −40 – 40 m    |
| 2   | `z_l`     | Up          | −1 – 4 m      |
| 3   | intensity | —           | 0 – 1         |

### Labels — `.txt` files
One object per line, **8 space-separated fields**:

```
Class  x_l  y_l  z_l  l  w  h  yaw
```

| Field   | Meaning                                       |
|---------|-----------------------------------------------|
| Class   | `Car` / `Human` / `ForkLift` / `CargoBike` / `ELFplusplus` / `FTS` |
| x_l y_l z_l | **True center** in LiDAR frame (meters)  |
| l w h   | length (forward) / width (lateral) / height   |
| yaw     | Rotation around Z-axis (up), radians          |

---

## 2. PointRCNN Expected Format

The model reads from `data/dataset/KITTI/object/training/`:

```
training/
├── velodyne/      *.bin   — point clouds in CAMERA frame
├── label_2/       *.txt   — KITTI-format labels
├── calib/         *.txt   — calibration files
└── ImageSets/
    ├── train.txt
    └── val.txt
```

### Point Cloud — `velodyne/*.bin`
Same `(N, 4)` float32 layout, but in **KITTI camera frame**:

| Col | Field     | Camera axis | Conversion from LiDAR |
|-----|-----------|-------------|------------------------|
| 0   | `x_cam`   | Right       | `= −y_l`               |
| 1   | `y_cam`   | Down        | `= −z_l`               |
| 2   | `z_cam`   | Forward     | `= x_l`                |
| 3   | intensity | —           | unchanged              |

### Labels — `label_2/*.txt`
One object per line, **15 space-separated fields** (KITTI format):

```
Class  trunc  occl  alpha  x1 y1 x2 y2  h  w  l  x_cam  y_cam_bottom  z_cam  ry
  0      1      2     3    4  5  6   7  8  9  10   11        12          13    14
```

| Field          | Value / Conversion                                      |
|----------------|---------------------------------------------------------|
| trunc / occl   | `0.0` / `0` (not used)                                  |
| alpha          | `−10.0` (not used)                                      |
| x1 y1 x2 y2   | `0 0 50 50` (dummy 2D bbox)                             |
| h w l          | same as source (height / width / length)                |
| x_cam          | `= −y_l`                                               |
| **y_cam_bottom** | **`= −z_l + h/2`** — KITTI bottom-center¹            |
| z_cam          | `= x_l`                                                 |
| ry             | `= normalise(−yaw − π/2)`, rotation around Y-axis       |

¹ The C++ kernel (`pts_in_box3d_cpu`) stores the passed y as `bottom_y` and internally computes `cy = bottom_y − h/2`. Storing true-center here would displace all boxes h/2 upward.

### Calibration — `calib/*.txt`
Bins are **already in camera frame** after conversion, so `Tr_velo_to_cam` must be the identity (otherwise the model applies a second transform and destroys the point cloud).

```
Tr_velo_to_cam: 1 0 0 0  0 1 0 0  0 0 1 0
R0_rect:        1 0 0    0 1 0    0 0 1
P2:             <dummy 3×4 projection — not used>
```

---

## 3. Conversion Pipeline

Run these steps in order on a fresh dataset:

```bash
cd /home/lenovo/venvs/pointrcnn/PointRCNN
source /home/lenovo/venvs/pointrcnn/bin/activate

# Step 1 — Convert LiDAR-frame bins + 8-field labels → camera-frame bins + KITTI labels
python tools/convert_data_to_kitti.py \
    --input_dir  data/dataset_original \
    --output_dir data/dataset

# Step 2 — Build KITTI directory tree (velodyne/, label_2/, calib/, ImageSets/)
python tools/preprocess_dataset.py

# Step 3 — Build ground-truth object database for augmentation
python tools/generate_gt_database_custom.py

# Step 4 — Generate augmented training scenes
python tools/generate_aug_scene_custom.py
```

> **Never patch `label_2/` manually** after running `preprocess_dataset.py`; just re-run Steps 1–4 from source.

---

## 4. Key Invariants

| Check | Expected |
|---|---|
| `velodyne/` Z range | 0 – ~50 m (forward depth) |
| `label_2/` `y_cam_bottom` for a Human at z_l=0 | ≈ +h/2 (positive, ground level) |
| `Tr_velo_to_cam` in every `calib/` file | identity (12 values: `1 0 0 0 0 1 0 0 0 0 1 0`) |
| Box corners from `generate_corners3d()` | bottom face at stored y, top face at y−h |
