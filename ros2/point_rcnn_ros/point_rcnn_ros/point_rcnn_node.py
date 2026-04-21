import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


def _find_repo_root():
    env_root = os.environ.get('POINT_RCNN_ROOT', '').strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    current_file = Path(__file__).resolve()
    candidates.extend(current_file.parents)
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        if (candidate / 'lib' / 'net' / 'point_rcnn.py').exists() and (candidate / 'tools' / 'cfgs').exists():
            return candidate
    raise RuntimeError('Unable to locate PointRCNN repo root. Set POINT_RCNN_ROOT to the repo path.')


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import cfg, cfg_from_file  # noqa: E402
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset  # noqa: E402
from lib.net.point_rcnn import PointRCNN  # noqa: E402
from lib.utils.bbox_transform import decode_bbox_target  # noqa: E402
import lib.utils.kitti_utils as kitti_utils  # noqa: E402
import torch  # noqa: E402


CLASS_COLORS = {
    'background': (0.8, 0.8, 0.8, 0.9),
    'car': (1.0, 0.45, 0.1, 0.95),
    'human': (0.1, 0.85, 0.25, 0.95),
    'forklift': (1.0, 0.2, 0.2, 0.95),
    'cargobike': (0.15, 0.55, 1.0, 0.95),
}

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _torch_load_compat(filename, map_location):
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=map_location)


def get_class_names():
    class_names = KittiRCNNDataset.parse_classes(cfg.CLASSES)
    if not class_names:
        raise ValueError(f'No classes found in cfg.CLASSES: {cfg.CLASSES}')
    return ['Background'] + class_names


def ensure_iou3d_utils(use_cuda: bool):
    try:
        import lib.utils.iou3d.iou3d_utils as iou3d_utils
        return iou3d_utils, use_cuda
    except (ImportError, ModuleNotFoundError):
        return None, False


def _sample_points_like_dataset(pts_cam, pts_intensity, npoints):
    num_pts = len(pts_cam)
    if num_pts == 0:
        ret_pts_cam = np.zeros((npoints, 3), dtype=np.float32)
        ret_pts_intensity = np.zeros((npoints,), dtype=np.float32)
        return ret_pts_cam, ret_pts_intensity

    if npoints < num_pts:
        pts_depth = pts_cam[:, 2]
        near_mask = pts_depth < 40.0
        far_idxs = np.where(~near_mask)[0]
        near_idxs = np.where(near_mask)[0]

        near_need = max(npoints - len(far_idxs), 0)
        if near_need > 0:
            source = near_idxs if len(near_idxs) > 0 else far_idxs
            near_choice = np.random.choice(source, near_need, replace=(near_need > len(source)))
        else:
            near_choice = np.array([], dtype=np.int32)

        choice = np.concatenate((near_choice, far_idxs), axis=0) if len(far_idxs) > 0 else near_choice
        if len(choice) == 0:
            choice = np.arange(0, num_pts, dtype=np.int32)
        if len(choice) < npoints:
            choice = np.concatenate((choice, np.random.choice(choice, npoints - len(choice), replace=True)), axis=0)
        elif len(choice) > npoints:
            choice = np.random.choice(choice, npoints, replace=False)

        np.random.shuffle(choice)
        return pts_cam[choice, :], pts_intensity[choice] - 0.5

    choice = np.arange(0, num_pts, dtype=np.int32)
    if npoints > num_pts:
        choice = np.concatenate((choice, np.random.choice(choice, npoints - num_pts, replace=True)), axis=0)
    np.random.shuffle(choice)
    return pts_cam[choice, :], pts_intensity[choice] - 0.5


def preprocess_points(pts_raw: np.ndarray, npoints: int, use_intensity: bool):
    pts_cam = pts_raw[:, 0:3]
    pts_intensity = pts_raw[:, 3]

    if cfg.PC_REDUCE_BY_RANGE:
        x_range, y_range, z_range = np.asarray(cfg.PC_AREA_SCOPE, dtype=np.float32)
        valid = (
            (pts_cam[:, 0] >= x_range[0]) & (pts_cam[:, 0] <= x_range[1]) &
            (pts_cam[:, 1] >= y_range[0]) & (pts_cam[:, 1] <= y_range[1]) &
            (pts_cam[:, 2] >= z_range[0]) & (pts_cam[:, 2] <= z_range[1])
        )
        pts_cam = pts_cam[valid]
        pts_intensity = pts_intensity[valid]

    ret_pts_cam, ret_pts_intensity = _sample_points_like_dataset(pts_cam, pts_intensity, npoints)
    pts_input = np.concatenate((ret_pts_cam, ret_pts_intensity.reshape(-1, 1)), axis=1) if use_intensity else ret_pts_cam
    return torch.from_numpy(pts_input[np.newaxis, ...]).float()


def _select_class_scores(rcnn_cls, score_thresh: float):
    import torch.nn.functional as F

    if rcnn_cls.shape[1] == 1:
        cls_scores = torch.sigmoid(rcnn_cls.view(-1))
        class_ids = torch.ones_like(cls_scores, dtype=torch.long)
        keep_mask = cls_scores > score_thresh
        raw_scores = rcnn_cls.view(-1)
        return class_ids, cls_scores, raw_scores, keep_mask

    cls_probs = F.softmax(rcnn_cls, dim=1)
    class_ids = torch.argmax(cls_probs, dim=1)
    cls_scores = torch.gather(cls_probs, 1, class_ids.unsqueeze(1)).squeeze(1)
    raw_scores = torch.gather(rcnn_cls, 1, class_ids.unsqueeze(1)).squeeze(1)
    keep_mask = (class_ids > 0) & (cls_scores > score_thresh)
    return class_ids, cls_scores, raw_scores, keep_mask


def transform_points_ros_lidar_to_camera(points_xyz):
    transformed = np.empty_like(points_xyz, dtype=np.float32)
    transformed[:, 0] = -points_xyz[:, 1]
    transformed[:, 1] = -points_xyz[:, 2]
    transformed[:, 2] = points_xyz[:, 0]
    return transformed


def transform_points_camera_to_ros_lidar(points_xyz):
    transformed = np.empty_like(points_xyz, dtype=np.float32)
    transformed[:, 0] = points_xyz[:, 2]
    transformed[:, 1] = -points_xyz[:, 0]
    transformed[:, 2] = -points_xyz[:, 1]
    return transformed


def convert_cloud_to_model_frame(points_xyzi: np.ndarray, frame_mode: str):
    converted = points_xyzi.copy()
    if frame_mode == 'camera':
        return converted
    if frame_mode == 'ros_lidar':
        converted[:, 0:3] = transform_points_ros_lidar_to_camera(converted[:, 0:3])
        return converted
    raise ValueError(f'Unsupported input_frame_mode: {frame_mode}')


def convert_corners_from_model_frame(corners_xyz: np.ndarray, frame_mode: str):
    if frame_mode == 'camera':
        return corners_xyz
    if frame_mode == 'ros_lidar':
        reshaped = corners_xyz.reshape(-1, 3)
        converted = transform_points_camera_to_ros_lidar(reshaped)
        return converted.reshape(corners_xyz.shape)
    raise ValueError(f'Unsupported input_frame_mode: {frame_mode}')


class PointRCNNRosNode(Node):
    def __init__(self):
        super().__init__('point_rcnn_node')

        self.declare_parameter('repo_root', str(REPO_ROOT))
        self.declare_parameter('cfg_file', 'tools/cfgs/orin_realtime.yaml')
        self.declare_parameter('ckpt', 'output/rcnn/default/ckpt/checkpoint_epoch_30.pth')
        self.declare_parameter('input_topic', '/points')
        self.declare_parameter('marker_topic', '/point_rcnn/markers')
        self.declare_parameter('frame_id_override', '')
        self.declare_parameter('input_frame_mode', 'ros_lidar')
        self.declare_parameter('score_thresh', 0.35)
        self.declare_parameter('nms_thresh', 0.1)
        self.declare_parameter('npoints', 16384)
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('line_width', 0.08)
        self.declare_parameter('text_size', 0.45)
        self.declare_parameter('marker_lifetime_sec', 0.25)
        self.declare_parameter('qos_depth', 1)
        self.declare_parameter('log_every_n_frames', 10)

        self.repo_root = Path(self.get_parameter('repo_root').value).expanduser().resolve()
        self.cfg_file = self._resolve_repo_path(self.get_parameter('cfg_file').value)
        self.ckpt = self._resolve_repo_path(self.get_parameter('ckpt').value)
        self.input_topic = self.get_parameter('input_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.frame_id_override = self.get_parameter('frame_id_override').value.strip()
        self.input_frame_mode = self.get_parameter('input_frame_mode').value.strip()
        self.score_thresh = float(self.get_parameter('score_thresh').value)
        self.nms_thresh = float(self.get_parameter('nms_thresh').value)
        self.npoints = int(self.get_parameter('npoints').value)
        self.use_cuda = bool(self.get_parameter('use_cuda').value)
        self.line_width = float(self.get_parameter('line_width').value)
        self.text_size = float(self.get_parameter('text_size').value)
        self.marker_lifetime_sec = float(self.get_parameter('marker_lifetime_sec').value)
        self.log_every_n_frames = int(self.get_parameter('log_every_n_frames').value)

        cfg_from_file(str(self.cfg_file))
        if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
            cfg.RCNN.ROI_SAMPLE_JIT = True
        cfg.RPN.NUM_POINTS = self.npoints
        cfg.RCNN.SCORE_THRESH = self.score_thresh
        cfg.RCNN.NMS_THRESH = self.nms_thresh

        self.class_names = get_class_names()
        self.device = self._select_device(self.use_cuda)
        self.iou3d_utils, self.use_cuda = ensure_iou3d_utils(self.use_cuda)
        self.model, matched_state, current_state = self._build_model()
        self.get_logger().info(
            f'Loaded model from {self.ckpt} with {len(matched_state)}/{len(current_state)} matched keys on {self.device.type}'
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.get_parameter('qos_depth').value),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.marker_publisher = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.subscription = self.create_subscription(PointCloud2, self.input_topic, self._pointcloud_callback, qos)

        self.frame_count = 0
        self.total_latency_ms = 0.0
        self.get_logger().info(
            f'Subscribed to {self.input_topic}, publishing RViz markers on {self.marker_topic}, input_frame_mode={self.input_frame_mode}'
        )

    def _resolve_repo_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.repo_root / path).resolve()

    def _select_device(self, prefer_cuda: bool):
        if prefer_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _build_model(self):
        checkpoint = _torch_load_compat(str(self.ckpt), map_location=self.device)
        model_state = checkpoint.get('model_state', checkpoint)

        model = PointRCNN(num_classes=len(self.class_names), use_xyz=True, mode='TEST')
        if self.device.type == 'cuda':
            model.cuda()
        model.eval()

        current_state = model.state_dict()
        matched_state = {k: v for k, v in model_state.items() if (k in current_state and current_state[k].shape == v.shape)}
        current_state.update(matched_state)
        model.load_state_dict(current_state)
        return model, matched_state, current_state

    def _pointcloud2_to_xyzi(self, msg: PointCloud2) -> np.ndarray:
        dtype = point_cloud2.dtype_from_fields(msg.fields, msg.point_step)
        cloud = np.frombuffer(msg.data, dtype=dtype)
        if msg.is_bigendian:
            cloud = cloud.byteswap().newbyteorder()

        field_names = cloud.dtype.names or ()
        if not {'x', 'y', 'z'}.issubset(field_names):
            raise ValueError('PointCloud2 is missing x/y/z fields')

        x = cloud['x'].astype(np.float32, copy=False)
        y = cloud['y'].astype(np.float32, copy=False)
        z = cloud['z'].astype(np.float32, copy=False)
        if 'intensity' in field_names:
            intensity = cloud['intensity'].astype(np.float32, copy=False)
        else:
            intensity = np.zeros_like(x, dtype=np.float32)

        stacked = np.stack((x, y, z, intensity), axis=1)
        valid = np.isfinite(stacked).all(axis=1)
        return stacked[valid]

    def _run_inference(self, points_model_frame: np.ndarray):
        pts_input_t = preprocess_points(points_model_frame, npoints=self.npoints, use_intensity=cfg.RPN.USE_INTENSITY)
        if self.device.type == 'cuda':
            pts_input_t = pts_input_t.cuda(non_blocking=True)

        with torch.no_grad():
            ret_dict = self.model({'pts_input': pts_input_t})

        roi_boxes3d = ret_dict['rois']
        rcnn_cls = ret_dict['rcnn_cls']
        rcnn_reg = ret_dict['rcnn_reg']

        class_ids, cls_scores, raw_scores, keep_mask = _select_class_scores(rcnn_cls, self.score_thresh)
        if cfg.RCNN.SIZE_RES_ON_ROI:
            anchor_size = roi_boxes3d.view(-1, 7)[:, 3:6]
        else:
            anchor_size = kitti_utils.get_class_anchor_sizes_torch(class_ids.view(-1), device=roi_boxes3d.device)

        pred_boxes3d = decode_bbox_target(
            roi_boxes3d.view(-1, 7),
            rcnn_reg.view(-1, rcnn_reg.shape[-1]),
            loc_scope=cfg.RCNN.LOC_SCOPE,
            loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
            num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
            anchor_size=anchor_size,
            get_xz_fine=True,
            get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
            loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
            loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
            get_ry_fine=True,
        ).view(roi_boxes3d.shape[0], -1, 7)

        if keep_mask.sum() == 0:
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes_kept = pred_boxes3d[0, keep_mask]
        scores_kept = cls_scores[keep_mask]
        raw_scores_kept = raw_scores[keep_mask]
        class_ids_kept = class_ids[keep_mask]

        final_boxes = []
        final_scores = []
        final_class_ids = []
        for class_id in torch.unique(class_ids_kept, sorted=True):
            class_mask = class_ids_kept == class_id
            class_boxes = boxes_kept[class_mask]
            class_scores = scores_kept[class_mask]
            class_raw_scores = raw_scores_kept[class_mask]

            if self.iou3d_utils is None:
                keep_idx = torch.argsort(class_raw_scores, descending=True)
            else:
                boxes_bev = kitti_utils.boxes3d_to_bev_torch(class_boxes)
                keep_idx = self.iou3d_utils.nms_gpu(boxes_bev, class_raw_scores, self.nms_thresh).view(-1)

            final_boxes.append(class_boxes[keep_idx])
            final_scores.append(class_scores[keep_idx])
            final_class_ids.append(
                torch.full((keep_idx.numel(),), int(class_id.item()), dtype=torch.long, device=class_boxes.device)
            )

        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_class_ids = torch.cat(final_class_ids, dim=0)
        sort_order = torch.argsort(final_scores, descending=True)
        return (
            final_boxes[sort_order].cpu().numpy(),
            final_scores[sort_order].cpu().numpy(),
            final_class_ids[sort_order].cpu().numpy(),
        )

    def _build_marker_array(self, stamp, frame_id: str, corners_xyz: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        lifetime = Duration(sec=int(self.marker_lifetime_sec), nanosec=int((self.marker_lifetime_sec % 1.0) * 1e9))
        for index, (corners, score, class_id) in enumerate(zip(corners_xyz, scores, class_ids)):
            class_name = self.class_names[int(class_id)]
            color = CLASS_COLORS.get(class_name.lower(), (1.0, 1.0, 1.0, 0.95))

            line_marker = Marker()
            line_marker.header.frame_id = frame_id
            line_marker.header.stamp = stamp
            line_marker.ns = 'point_rcnn_boxes'
            line_marker.id = index
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = self.line_width
            line_marker.color.r = float(color[0])
            line_marker.color.g = float(color[1])
            line_marker.color.b = float(color[2])
            line_marker.color.a = float(color[3])
            line_marker.lifetime = lifetime

            for start_idx, end_idx in BOX_EDGES:
                start_point = Point(x=float(corners[start_idx, 0]), y=float(corners[start_idx, 1]), z=float(corners[start_idx, 2]))
                end_point = Point(x=float(corners[end_idx, 0]), y=float(corners[end_idx, 1]), z=float(corners[end_idx, 2]))
                line_marker.points.append(start_point)
                line_marker.points.append(end_point)
            marker_array.markers.append(line_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = 'point_rcnn_labels'
            text_marker.id = 10000 + index
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(corners[:, 0].mean())
            text_marker.pose.position.y = float(corners[:, 1].mean())
            text_marker.pose.position.z = float(corners[:, 2].max() + 0.3)
            text_marker.scale.z = self.text_size
            text_marker.color.r = float(color[0])
            text_marker.color.g = float(color[1])
            text_marker.color.b = float(color[2])
            text_marker.color.a = 1.0
            text_marker.text = f'{class_name} {score:.2f}'
            text_marker.lifetime = lifetime
            marker_array.markers.append(text_marker)

        return marker_array

    def _pointcloud_callback(self, msg: PointCloud2):
        start_time = time.perf_counter()
        try:
            points_xyzi = self._pointcloud2_to_xyzi(msg)
            if points_xyzi.shape[0] == 0:
                self.marker_publisher.publish(self._build_marker_array(msg.header.stamp, self._output_frame_id(msg), np.zeros((0, 8, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)))
                return

            model_points = convert_cloud_to_model_frame(points_xyzi, self.input_frame_mode)
            pred_boxes, scores, class_ids = self._run_inference(model_points)
            corners_model = kitti_utils.boxes3d_to_corners3d(pred_boxes) if len(pred_boxes) > 0 else np.zeros((0, 8, 3), dtype=np.float32)
            corners_output = convert_corners_from_model_frame(corners_model, self.input_frame_mode)

            marker_array = self._build_marker_array(msg.header.stamp, self._output_frame_id(msg), corners_output, scores, class_ids)
            self.marker_publisher.publish(marker_array)

            self.frame_count += 1
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.total_latency_ms += latency_ms
            if self.log_every_n_frames > 0 and self.frame_count % self.log_every_n_frames == 0:
                avg_ms = self.total_latency_ms / max(self.frame_count, 1)
                self.get_logger().info(
                    f'Processed {self.frame_count} frames, last={latency_ms:.1f} ms, avg={avg_ms:.1f} ms, detections={len(scores)}'
                )
        except Exception as exc:
            self.get_logger().error(f'Inference callback failed: {exc}')

    def _output_frame_id(self, msg: PointCloud2) -> str:
        return self.frame_id_override if self.frame_id_override else msg.header.frame_id


def main(args=None):
    rclpy.init(args=args)
    node = PointRCNNRosNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
