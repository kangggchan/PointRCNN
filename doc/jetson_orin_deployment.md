# Jetson Orin Deployment Notes

This repository can run on Jetson Orin with the existing CUDA extensions, but it is not directly ready for full TensorRT INT8 deployment.

## What Works Now

- Native PyTorch inference on Jetson GPU after building the custom extensions.
- Conservative runtime tuning with fewer input points and fewer RPN proposals.
- Latency benchmarking with `tools/benchmark_jetson.py`.

## What Blocks Full INT8 Today

The inference path depends on custom CUDA ops that TensorRT cannot ingest without plugins:

- PointNet++ sampling in `pointnet2_lib/pointnet2/src/sampling.cpp`
- Ball query in `pointnet2_lib/pointnet2/src/ball_query.cpp`
- Grouping in `pointnet2_lib/pointnet2/src/group_points.cpp`
- Three-NN / interpolation in `pointnet2_lib/pointnet2/pointnet2_utils.py`
- ROI pool in `lib/utils/roipool3d/roipool3d_utils.py`
- Rotated NMS / IoU in `lib/utils/iou3d/iou3d_utils.py`

These extensions are also implemented around `float*` tensors today, so they are FP32-only unless you add explicit FP16 or INT8 support in the kernels and bindings.

## Recommended Deployment Path

### Phase 1: Establish an Orin Baseline

On the Jetson device:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks

cd PointRCNN
bash build_and_install.sh
```

Then benchmark with the conservative runtime config:

```bash
cd tools
python benchmark_jetson.py \
  --cfg_file cfgs/orin_realtime.yaml \
  --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_30.pth \
  --bin_glob "../data/dataset/KITTI/object/training/velodyne/*.bin" \
  --max_frames 16 \
  --warmup 20 \
  --repeats 20 \
  --save_json ../results/jetson_orin_baseline.json
```

If accuracy remains acceptable, continue reducing only one knob at a time:

- `RPN.NUM_POINTS`: try `12288`, then `8192`
- `TEST.RPN_POST_NMS_TOP_N`: try `96`, then `64`
- `TEST.RPN_PRE_NMS_TOP_N`: try `1536`

### Phase 2: Ship a Mixed-Precision Native Runtime

If the baseline is close to real-time, the fastest path to deployment is to ship the current PyTorch runtime with native CUDA extensions, plus the reduced runtime config.

This is the lowest engineering risk path because it avoids TensorRT plugin work.

### Phase 3: Full TensorRT INT8

If you still need more speed after Phase 2, a real INT8 deployment requires:

1. Replacing or rewriting the custom PointNet++ / ROI / NMS ops as TensorRT plugins.
2. Building an export graph that separates plugin-backed geometry ops from dense layers.
3. Creating an INT8 calibration dataset from representative warehouse frames.
4. Validating class-wise accuracy and yaw stability after calibration.

This is a significant engineering task, not a config change.

## Practical Recommendation

For this repo, do not start with INT8. Start by proving whether native Orin inference with tuned proposal counts is already fast enough. The custom ops dominate the graph, so INT8 on only the small Conv/MLP heads will not give enough speedup to justify the deployment complexity.

If the baseline is still too slow, the next serious step is plugin work or moving to a TensorRT-friendly 3D detector architecture.

## ROS2 Realtime Node

A minimal ROS2 package is included in `ros2/point_rcnn_ros`.

It subscribes to `sensor_msgs/PointCloud2`, runs PointRCNN inference, and publishes `visualization_msgs/MarkerArray` wireframe boxes that can be visualized in RViz together with the original point cloud.

### Frame Convention

The model in this repo expects point clouds in the project camera frame:

- `x`: right
- `y`: down
- `z`: forward

The ROS2 node supports two input modes:

- `camera`: input topic is already in the model frame
- `ros_lidar`: input topic uses the common ROS LiDAR frame `x forward, y left, z up`

When `input_frame_mode=ros_lidar`, the node converts points into the model frame for inference and converts the predicted box corners back to the original LiDAR frame before publishing markers, so the markers align with the incoming point cloud in RViz.

### Build

Create or reuse a ROS2 workspace and link the package into `src`:

```bash
export POINT_RCNN_ROOT=/path/to/PointRCNN
mkdir -p ~/ros2_ws/src
ln -s $POINT_RCNN_ROOT/ros2/point_rcnn_ros ~/ros2_ws/src/point_rcnn_ros

cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Run

```bash
export POINT_RCNN_ROOT=/path/to/PointRCNN
source ~/ros2_ws/install/setup.bash

ros2 run point_rcnn_ros point_rcnn_node \
  --ros-args \
  -p repo_root:=$POINT_RCNN_ROOT \
  -p input_topic:=/points_raw \
  -p marker_topic:=/point_rcnn/markers \
  -p input_frame_mode:=ros_lidar
```

Or with launch:

```bash
ros2 launch point_rcnn_ros point_rcnn.launch.py \
  repo_root:=$POINT_RCNN_ROOT \
  input_topic:=/points_raw \
  marker_topic:=/point_rcnn/markers \
  input_frame_mode:=ros_lidar
```

### RViz2

In RViz2:

- add a `PointCloud2` display for your LiDAR topic
- add a `MarkerArray` display for `/point_rcnn/markers`

The node publishes wireframe boxes and text labels in the same frame as the incoming point cloud unless `frame_id_override` is set.