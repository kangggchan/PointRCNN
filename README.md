# Train PointRCNN on a Custom Dataset (like `data/dataset`)

This guide explains how to train this repo on a custom LIDAR pointcloud dataset. The dataset used in this project is a combination of [Lidar Warehouse Dataset](https://github.com/anavsgmbh/lidar-warehouse-dataset) and SCALA 2 dataset which we captured by Valeo SCALA 2 Lidar then using an automation labeling pipeline to generate the labels for Car and Human classes. The idea is training a model that have the ability to detect, tracking dynamic objects in a warehouse, then predict its moving direction.

> **⚠️ Important**: This repository has been modified from original PointRCNN to support modern Pytorch and CUDA, custom pointcloud dataset. Custom datasets using Z-axis rotation (yaw) are now properly distinguished from KITTI format which uses Y-axis rotation (ry). See [DATASET_FORMAT.md](DATASET_FORMAT.md) for technical details on coordinate frame handling.

## 1) Supported custom dataset format

Your dataset root should look like:

```text
data/dataset/
  bin/
    000000.bin
    000001.bin
    ...
  labels/
    000000.txt
    000001.txt
    ...
```

Rules:
- Only **paired** files are used (`bin/ID.bin` + `labels/ID.txt`).
- Unpaired files are skipped.
- Label names `box`, `boxes`, `bbox` are ignored.
- Supported classes for training:
  - `Car`
  - `Human`
  - `ForkLift`
  - `CargoBike`

## 2) Label formats accepted

The preprocessing supports either of these formats per line:

### A) Lightweight format (recommended for your data)

```text
ClassName x y z l w h yaw
```

Example:

```text
Car 14.0997 -0.3185 0.1975 4.5086 1.7096 1.7837 -2.1002
Human 10.1259 -2.7427 0.0549 0.9508 0.4599 1.9657 0.1574
```

### B) KITTI full 15/16-field format

If labels are already KITTI-style, they are preserved.

## 3) One-time preprocessing (creates KITTI-style training layout)

Run:

```bash
cd PointRCNN/tools
python preprocess_dataset.py \
  --dataset_root ../data/dataset \
  --classes Car,Human,ForkLift,CargoBike
```

This generates:

```text
data/dataset/KITTI/
  ImageSets/{train,val,smallval,trainval,test}.txt
  object/training/{velodyne,label_2,calib}
  object/testing/{velodyne,calib}
```

Notes:
- Split is auto-created as 80% train / 20% val.
- `smallval.txt` equals `val.txt`.
- Dummy calibration files are generated so existing PointRCNN data flow works.
- 
### Optional:
**Note**: You can use `tools\analyze_aug_scene_points_in_boxes.py` to analyze number of points inside the bounding box per class then use this command to filter the boxes that have less points than a certain number to reduce noise for your imbalance custom dataset

#### Basic: Filter Human objects < 100 points (no backup)
```bas
python filter_aug_labels_by_points.py \
  --aug_label_dir ../data/dataset/KITTI/aug_scene/training/aug_label \
  --aug_pts_dir ../data/dataset/KITTI/aug_scene/training/rectified_data \
  --class_name Human \
  --min_points 100
```

#### With backup: Keep copy of originals before filtering
```bash
python filter_aug_labels_by_points.py \
  --aug_label_dir ../data/dataset/KITTI/aug_scene/training/aug_label \
  --aug_pts_dir ../data/dataset/KITTI/aug_scene/training/rectified_data \
  --class_name Human \
  --min_points 100 \
  --backup_dir ../data/dataset/KITTI/aug_scene/training/aug_label_backup
```

#### Multiple classes: Filter multiple classes at once
```bash
python filter_aug_labels_by_points.py \
  --aug_label_dir ../data/dataset/KITTI/object/training/label_2 \
  --aug_pts_dir ../data/dataset/KITTI/object/training/velodyne \
  --class_name Car Human CargoBike ForkLift \
  --min_points 400 70 60 130 \
  --backup_dir ../data/dataset/KITTI/aug_scene/training/aug_label_backup
```

## 4) Config used for custom training

Default config was updated in `tools/cfgs/default.yaml`:
- `CLASSES: Car,Human,ForkLift,CargoBike`
- `INCLUDE_SIMILAR_TYPE: False`
- `RCNN.CLS_WEIGHT` expanded to 5 entries (background + 4 classes)

## 5) Full pipeline data generation (same flow as original repo)

### A) Generate GT database

```bash
python generate_gt_database.py \
  --root_dir ../data/dataset \
  --split train \
  --class_name Car,Human,ForkLift,CargoBike
```

This generates:
- `gt_database/train_gt_database_3level_multi.pkl`

### B) Generate augmented scenes

**Note**: With `--weighted_sampling` enabled, the augmentation automatically uses class weights from `tools/cfgs/default.yaml` (`RPN.CLS_WEIGHT`). This ensures augmentation balances classes the same way training does. You can optionally override with `--class_weights "Car:1.1,Human:0.5,..."` if needed.

```bash
python generate_aug_scene.py \
  --root_dir ../data/dataset \
  --split train \
  --class_name Car,Human,ForkLift,CargoBike \
  --weighted_sampling \
  --gt_database_dir gt_database/train_gt_database_3level_multi.pkl \
  --aug_times 3
```

This generates:
- `data/dataset/KITTI/aug_scene/training/rectified_data/*.bin`
- `data/dataset/KITTI/aug_scene/training/aug_label/*.txt`
- `data/dataset/KITTI/ImageSets/train_aug.txt`

If you train with augmented split, set in `tools/cfgs/default.yaml`:
- `TRAIN.SPLIT: train_aug`

## 6) Train RPN stage

```bash
cd tools
python train_rcnn.py \
  --cfg_file cfgs/default.yaml \
  --batch_size 8 \
  --train_mode rpn \
  --epochs 200 \
  --data_root ../data/dataset \
  --gt_database ../gt_database/train_gt_database_3level_multi.pkl
```

## 7) Train RCNN stage with online proposals

```bash
cd tools
python train_rcnn.py \
  --cfg_file cfgs/default.yaml \
  --batch_size 4 \
  --train_mode rcnn \
  --epochs 70 \
  --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth \
  --data_root ../data/dataset \
  --gt_database ../gt_database/train_gt_database_3level_multi.pkl
```

#### Optional: CPU proposal sampling

For the offline workflow, the original repo notes the best model was trained with CPU proposal sampling by setting:

- `RCNN.ROI_SAMPLE_JIT: False`

This can improve results, but it uses more CPU worker time.

## 8) Automatic preprocessing behavior in trainer

`tools/train_rcnn.py` now defaults to:
- `--data_root ../data/dataset`
- auto-preprocesses if `data_root/KITTI/ImageSets/<split>.txt` is missing.

If you already preprocessed and want to skip checks:

```bash
python train_rcnn.py ... --skip_preprocess
```

## 9) Practical tips

- Start with smaller batch sizes if GPU memory is limited.
- If you want the strongest RCNN results, prefer the offline RCNN workflow above over the online one.
- If training is unstable, lower LR in `tools/cfgs/default.yaml`.
- Keep class names exactly consistent between labels and config.
- Make sure all `.bin` are `float32` with shape `N x 4` (`x y z intensity`).

## 10) Utilities
### To check class balance inside KITTI folder before generate DB
```bash
python tools/check_class_balance.py
```

### To compute mean of bounding box by class
```bash
python tools/compute_cls_mean_size.py
```

### To visualize a pointcloud frame with label in Open3D:
```bash
python tools/verify_bin_conversion.py
```

### To run PointRCNN inference in Open3D for one frame
Run from the repository root. The script accepts a single `.bin` file and prints a one-frame FPS estimate before opening the viewer.
If a matching GT label file exists next to the frame or in a sibling `label_2/` or `labels/` folder, it is drawn automatically as red bounding boxes.

```bash
python infer_frame_open3d.py \
  --cfg_file cfgs/default.yaml \
  --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --bin_file ../data/dataset/KITTI/aug_scene/training/rectified_data/010210.bin
```

### To run PointRCNN inference on a folder of consecutive frames and measure FPS
Use `--bin_dir` for a folder of frames or `--bin_glob` if you want finer file matching. The script streams the detections in Open3D and prints these metrics at the end:

- `Inference FPS`: model forward pass plus decode and NMS
- `Processing FPS`: file load, preprocessing, inference, and Open3D update
- `Display FPS`: processing FPS plus any playback cap from `--playback_fps`

For KITTI-style camera-frame bins:

```bash
python tools/infer_frame_open3d.py \
  --cfg_file cfgs/default.yaml \
  --ckpt output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --bin_dir data/dataset/KITTI/object/training/velodyne \
  --input_frame camera \
  --score_thresh 0.2 \
  --playback_fps 10 \
  --save_json output/rcnn/default/sequence_fps.json
```

For raw Scala2-format bins described in [DATASET_FORMAT.md](DATASET_FORMAT.md), use `--input_frame lidar`. The script converts each frame to the camera frame internally before inference:

```bash
python infer_frame_open3d.py \
  --cfg_file cfgs/default.yaml \
  --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --bin_dir ../data/scala2/bin \
  --input_frame lidar \
  --score_thresh 0.6 \
  --playback_fps 20 \
  --save_json output/rcnn/default/raw_sequence_fps.json
```

Useful options:

- `--max_frames 200`: stop after a fixed number of frames
- `--stride 2`: use every second frame
- `--loop`: replay the sequence until the Open3D window is closed
- `--bin_pattern "*.bin"`: change the filename filter used with `--bin_dir`
- `--no_cuda`: force CPU inference


### To evaluate an RPN checkpoint or export offline proposals

Run from `tools` when you want proposal recall, saved detections, or offline RCNN features.

```bash
cd tools
python eval_rpn.py \
  --cfg_file cfgs/default.yaml \
  --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth \
  --data_root ../data/dataset \
  --save_result
```

Add `--save_rpn_feature` when preparing the offline RCNN training workflow in section 7B.

### To evaluate an RCNN checkpoint on `bin/` + `labels/`

Run from the repository root. The script scans paired files under `data_root/bin` and `data_root/labels`, then writes a summary CSV under `output/rcnn_eval/` by default.

```bash
python tools/eval_rcnn.py \
  --ckpt output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --data_root data/dataset \
  --iou_thresh 0.5
```

### To benchmark Jetson or edge-device inference

Use this to measure preprocessing, model, decode/NMS, and total latency on a representative frame set.

```bash
python tools/benchmark_jetson.py \
  --cfg_file tools/cfgs/orin_realtime.yaml \
  --ckpt output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --bin_glob "data/dataset/KITTI/object/training/velodyne/*.bin" \
  --max_frames 16 \
  --warmup 20 \
  --repeats 20 \
  --save_json output/benchmark_jetson.json
```

### To auto-label a Scala2 frame range

Run from the repository root. The script reads raw Scala2 `bin/` frames, replaces existing labels that start with `--label_name`, and keeps other label lines untouched.

```bash
python label_scala2_range.py \
  --data_root data/scala2 \
  --start_idx 0 \
  --end_idx 211 \
  --ckpt output/rcnn/default/ckpt/checkpoint_epoch_40.pth \
  --cfg_file tools/cfgs/default.yaml \
  --score_thresh 0.5 \
  --label_name Car
```

### To enable Open3D GUI in WSL:
```bash
unset WAYLAND_DISPLAY
export XDG_SESSION_TYPE=x11
LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe 
```
