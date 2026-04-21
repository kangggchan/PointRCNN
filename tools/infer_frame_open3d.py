import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from lib.config import cfg, cfg_from_file
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.kitti_utils as kitti_utils


SCRIPT_DIR = Path(__file__).resolve().parent
EMPTY_COLORS = np.zeros((0, 3), dtype=np.float64)

CLASS_COLORS = {
    'background': (0.8, 0.8, 0.8),
    'car': (1.0, 0.45, 0.1),
    'human': (0.1, 0.85, 0.25),
    'forklift': (1.0, 0.2, 0.2),
    'cargobike': (0.15, 0.55, 1.0),
    'gt': (1.0, 0.0, 0.0),
}

BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def parse_args():
    parser = argparse.ArgumentParser(description='PointRCNN inference for one frame or a frame sequence with Open3D visualization.')
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='Inference config file.')
    parser.add_argument('--ckpt', type=str, default='../output/rcnn/default/ckpt/checkpoint_epoch_40.pth',
                        help='Checkpoint to load.')
    parser.add_argument('--bin_file', type=str, default='../data/dataset/KITTI/aug_scene/training/rectified_data/010210.bin',
                        help='Single .bin file. Ignored when --bin_dir or --bin_glob is provided.')
    parser.add_argument('--bin_dir', type=str, default=None,
                        help='Directory containing consecutive .bin frames.')
    parser.add_argument('--bin_glob', type=str, default=None,
                        help='Glob pattern for consecutive .bin frames.')
    parser.add_argument('--bin_pattern', type=str, default='*.bin',
                        help='Filename pattern used with --bin_dir.')
    parser.add_argument('--input_frame', type=str, choices=['camera', 'lidar'], default='camera',
                        help='Input bin coordinate frame. Use lidar for raw Scala2-format bins.')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='Minimum confidence threshold.')
    parser.add_argument('--nms_thresh', type=float, default=None, help='NMS threshold override.')
    parser.add_argument('--npoints', type=int, default=None, help='Point sampling override.')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to play.')
    parser.add_argument('--stride', type=int, default=1, help='Keep every Nth frame from the sequence.')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup iterations before timed inference.')
    parser.add_argument('--playback_fps', type=float, default=0.0,
                        help='Optional playback cap. Use 0 to run as fast as possible.')
    parser.add_argument('--loop', action='store_true', help='Replay the frame sequence until the Open3D window is closed.')
    parser.add_argument('--save_json', type=str, default=None, help='Optional output JSON path for FPS summary.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    parser.add_argument('--no_axes', action='store_true', help='Hide coordinate axes at origin.')
    parser.add_argument('--axis_size', type=float, default=2.5, help='Coordinate frame axis length.')
    parser.add_argument('--point_size', type=float, default=2.0, help='Open3D point size.')
    return parser.parse_args()


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (SCRIPT_DIR / path).resolve()


def resolve_glob_paths(pattern: str):
    expanded = str(Path(pattern).expanduser())
    matches = sorted(glob.glob(expanded))
    if matches:
        return [Path(match).resolve() for match in matches]

    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        return []

    return [Path(match).resolve() for match in sorted(glob.glob(str((SCRIPT_DIR / pattern_path).resolve())))]


def resolve_frame_paths(args):
    if args.bin_dir:
        bin_dir = resolve_existing_path(args.bin_dir)
        if not bin_dir.exists():
            raise FileNotFoundError(f'Point cloud directory not found: {bin_dir}')
        frame_paths = sorted(path.resolve() for path in bin_dir.glob(args.bin_pattern) if path.is_file())
    elif args.bin_glob:
        frame_paths = resolve_glob_paths(args.bin_glob)
    else:
        frame_paths = [resolve_existing_path(args.bin_file)]

    stride = max(int(args.stride), 1)
    frame_paths = frame_paths[::stride]
    if args.max_frames is not None and args.max_frames > 0:
        frame_paths = frame_paths[:args.max_frames]
    if not frame_paths:
        raise FileNotFoundError('No .bin files matched the provided input')
    return frame_paths


def load_points(bin_path: Path) -> np.ndarray:
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud file not found: {bin_path}')

    raw = np.fromfile(str(bin_path), dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f'Expected float32 point cloud with 4 values per point, got {raw.size} values from {bin_path}')
    return raw.reshape(-1, 4)


def convert_points_to_camera_frame(points: np.ndarray, input_frame: str) -> np.ndarray:
    if input_frame == 'camera':
        return points

    points_camera = points.copy()
    points_camera[:, 0] = -points[:, 1]
    points_camera[:, 1] = -points[:, 2]
    points_camera[:, 2] = points[:, 0]
    return points_camera


def resolve_gt_label_path(bin_path: Path):
    label_name = f'{bin_path.stem}.txt'
    candidates = [
        bin_path.with_suffix('.txt'),
        bin_path.parent / label_name,
        bin_path.parent.parent / 'label_2' / label_name,
        bin_path.parent.parent / 'labels' / label_name,
    ]

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def load_gt_boxes(bin_path: Path):
    label_path = resolve_gt_label_path(bin_path)
    if label_path is None:
        return np.zeros((0, 7), dtype=np.float32), None

    lines = [line.strip() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        return np.zeros((0, 7), dtype=np.float32), label_path

    first_parts = lines[0].split()
    if len(first_parts) >= 15:
        gt_objects = [
            obj for obj in kitti_utils.get_objects_from_label(str(label_path))
            if obj.cls_type.lower() not in {'box', 'boxes', 'bbox'} and obj.cls_type != 'DontCare'
        ]
        return kitti_utils.objs_to_boxes3d(gt_objects), label_path

    gt_boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 8 or parts[0].lower() in {'box', 'boxes', 'bbox'}:
            continue

        _, x_l, y_l, z_l, l, w, h, yaw = parts
        x_l = float(x_l)
        y_l = float(y_l)
        z_l = float(z_l)
        l = float(l)
        w = float(w)
        h = float(h)
        yaw = float(yaw)

        gt_boxes.append([
            -y_l,
            -z_l + (h / 2.0),
            x_l,
            h,
            w,
            l,
            ((-yaw - np.pi / 2.0 + np.pi) % (2.0 * np.pi)) - np.pi,
        ])

    if not gt_boxes:
        return np.zeros((0, 7), dtype=np.float32), label_path

    return np.asarray(gt_boxes, dtype=np.float32), label_path


def _torch_load_compat(filename, map_location):
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=map_location)


def should_use_cuda(no_cuda: bool) -> bool:
    return (not no_cuda) and torch.cuda.is_available()


def ensure_iou3d_utils(use_cuda: bool):
    try:
        import lib.utils.iou3d.iou3d_utils as iou3d_utils
        return iou3d_utils, use_cuda
    except (ImportError, ModuleNotFoundError):
        print('Warning: iou3d_utils not available, falling back to score-sort NMS')
        return None, False


def synchronize_if_needed(use_cuda: bool):
    if use_cuda:
        torch.cuda.synchronize()


def get_class_names():
    class_names = KittiRCNNDataset.parse_classes(cfg.CLASSES)
    if not class_names:
        raise ValueError(f'No classes found in cfg.CLASSES: {cfg.CLASSES}')
    return ['Background'] + class_names


def build_model(ckpt_path: str, class_names: list, device: torch.device, use_cuda: bool):
    checkpoint = _torch_load_compat(ckpt_path, map_location=device)
    model_state = checkpoint.get('model_state', checkpoint)

    model = PointRCNN(num_classes=len(class_names), use_xyz=True, mode='TEST')
    if use_cuda:
        model.cuda()
    model.eval()

    current_state = model.state_dict()
    matched_state = {k: v for k, v in model_state.items() if (k in current_state and current_state[k].shape == v.shape)}
    current_state.update(matched_state)
    model.load_state_dict(current_state)
    return model, matched_state, current_state


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


def preprocess_points(pts_cam_frame: np.ndarray, npoints: int, use_intensity: bool):
    pts_cam = pts_cam_frame[:, 0:3]
    pts_intensity = pts_cam_frame[:, 3]

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


def _select_class_scores(rcnn_cls: torch.Tensor, score_thresh: float):
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


def run_inference(model, pts_input_t, score_thresh, nms_thresh, use_cuda, iou3d_utils):
    if use_cuda:
        pts_input_t = pts_input_t.cuda(non_blocking=True)

    with torch.no_grad():
        ret_dict = model({'pts_input': pts_input_t})

    roi_boxes3d = ret_dict['rois']
    rcnn_cls = ret_dict['rcnn_cls']
    rcnn_reg = ret_dict['rcnn_reg']

    class_ids, cls_scores, raw_scores, keep_mask = _select_class_scores(rcnn_cls, score_thresh)
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

    final_boxes, final_scores, final_class_ids = [], [], []
    for class_id in torch.unique(class_ids_kept, sorted=True):
        class_mask = class_ids_kept == class_id
        class_boxes = boxes_kept[class_mask]
        class_scores = scores_kept[class_mask]
        class_raw_scores = raw_scores_kept[class_mask]

        if iou3d_utils is None:
            keep_idx = torch.argsort(class_raw_scores, descending=True)
        else:
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(class_boxes)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev, class_raw_scores, nms_thresh).view(-1)

        final_boxes.append(class_boxes[keep_idx])
        final_scores.append(class_scores[keep_idx])
        final_class_ids.append(torch.full((keep_idx.numel(),), int(class_id.item()), dtype=torch.long, device=class_boxes.device))

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_class_ids = torch.cat(final_class_ids, dim=0)

    sort_order = torch.argsort(final_scores, descending=True)
    return (
        final_boxes[sort_order].cpu().numpy(),
        final_scores[sort_order].cpu().numpy(),
        final_class_ids[sort_order].cpu().numpy(),
    )


def summarize_ms(values):
    if not values:
        return {'mean_ms': 0.0, 'p50_ms': 0.0, 'p90_ms': 0.0, 'p95_ms': 0.0, 'fps_mean': 0.0}

    arr = np.asarray(values, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        'mean_ms': mean_ms,
        'p50_ms': float(np.percentile(arr, 50)),
        'p90_ms': float(np.percentile(arr, 90)),
        'p95_ms': float(np.percentile(arr, 95)),
        'fps_mean': float(1000.0 / max(mean_ms, 1e-6)),
    }


def build_point_colors(xyz: np.ndarray) -> np.ndarray:
    if xyz.shape[0] == 0:
        return EMPTY_COLORS

    ranges = np.linalg.norm(xyz[:, :2], axis=1)
    scale = max(float(np.percentile(ranges, 95)), 1e-3)
    color_mix = np.clip(ranges / scale, 0.0, 1.0)
    return np.stack([0.2 + 0.7 * color_mix, 0.75 - 0.5 * color_mix, 1.0 - 0.6 * color_mix], axis=1)


def make_box_lineset(boxes_rect: np.ndarray, class_names: list):
    if len(boxes_rect) == 0:
        return None

    corners_rect = kitti_utils.boxes3d_to_corners3d(boxes_rect)
    line_points = corners_rect.reshape(-1, 3).astype(np.float64)

    line_indices = []
    line_colors = []
    for box_idx, class_name in enumerate(class_names):
        color = CLASS_COLORS.get(class_name.lower(), (1.0, 1.0, 1.0))
        point_offset = box_idx * 8
        for start_idx, end_idx in BOX_EDGES:
            line_indices.append([point_offset + start_idx, point_offset + end_idx])
            line_colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    return line_set


def create_visualizer(show_axes: bool, axis_size: float, point_size: float, points: np.ndarray):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointRCNN Open3D Inference', width=1400, height=900)

    point_cloud = o3d.geometry.PointCloud()
    update_point_cloud(point_cloud, points)
    vis.add_geometry(point_cloud)

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0])
        vis.add_geometry(axes)

    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.04, 0.05, 0.09])
    return vis, point_cloud


def update_point_cloud(point_cloud, points: np.ndarray):
    xyz = points[:, :3].astype(np.float64)
    point_colors = build_point_colors(xyz)
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)


def get_scene_center(points: np.ndarray):
    if points.shape[0] == 0:
        return [0.0, 0.0, 0.0]

    xyz = points[:, :3].astype(np.float64)
    return ((xyz.min(axis=0) + xyz.max(axis=0)) * 0.5).tolist()


def set_default_view(vis, points: np.ndarray = None):
    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])
    view_control.set_lookat(get_scene_center(points) if points is not None else [0.0, 0.0, 0.0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)


def visualize(points: np.ndarray, boxes_rect: np.ndarray, scores: np.ndarray, class_names: list,
              show_axes: bool, axis_size: float, point_size: float,
              gt_boxes_rect: np.ndarray = None, gt_label_path: Path = None):
    gt_boxes_rect = np.zeros((0, 7), dtype=np.float32) if gt_boxes_rect is None else gt_boxes_rect

    for idx, (score, class_name) in enumerate(zip(scores, class_names), start=1):
        print(f'  Det {idx:02d}: class={class_name:<11} score={score:.3f}')

    if gt_label_path is not None:
        print(f'GT labels       : {gt_label_path} ({len(gt_boxes_rect)} boxes)')

    print(f'\nVisualizing {len(boxes_rect)} detections (camera-frame view)')
    print('Open3D controls: drag=rotate  Ctrl+drag=pan  scroll=zoom  Q/Esc=quit')

    vis, point_cloud = create_visualizer(show_axes=show_axes, axis_size=axis_size, point_size=point_size, points=points)
    gt_box_lines = make_box_lineset(gt_boxes_rect, ['gt'] * len(gt_boxes_rect))
    box_lines = make_box_lineset(boxes_rect, class_names)
    if gt_box_lines is not None:
        vis.add_geometry(gt_box_lines)
    if box_lines is not None:
        vis.add_geometry(box_lines)
    vis.update_geometry(point_cloud)
    if gt_box_lines is not None:
        vis.update_geometry(gt_box_lines)
    if box_lines is not None:
        vis.update_geometry(box_lines)
    set_default_view(vis, points)
    vis.run()
    vis.destroy_window()


def warmup_model(model, points: np.ndarray, npoints: int, score_thresh: float, nms_thresh: float,
                 use_cuda: bool, iou3d_utils, warmup_iters: int):
    for _ in range(max(int(warmup_iters), 0)):
        pts_input_t = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
        run_inference(model, pts_input_t, score_thresh, nms_thresh, use_cuda, iou3d_utils)
        synchronize_if_needed(use_cuda)


def print_sequence_summary(summary):
    print('\nSequence summary')
    print(f"Frames processed : {summary['frames_processed']}")
    print(f"Elapsed seconds  : {summary['elapsed_seconds']:.2f}")
    print(f"Avg detections   : {summary['avg_detections_per_frame']:.2f}")
    print(f"Inference FPS    : {summary['timings']['inference']['fps_mean']:.2f}")
    print(f"Processing FPS   : {summary['timings']['processing_total']['fps_mean']:.2f}")
    print(f"Display FPS      : {summary['timings']['display_total']['fps_mean']:.2f}")


def run_sequence_visualization(model, frame_paths, class_names, args, npoints, nms_thresh, use_cuda, iou3d_utils):
    first_points = convert_points_to_camera_frame(load_points(frame_paths[0]), args.input_frame)
    warmup_model(model, first_points, npoints, args.score_thresh, nms_thresh, use_cuda, iou3d_utils, args.warmup)

    vis, point_cloud = create_visualizer(
        show_axes=(not args.no_axes),
        axis_size=float(args.axis_size),
        point_size=float(args.point_size),
        points=first_points,
    )
    set_default_view(vis, first_points)
    gt_box_lines = None
    box_lines = None

    print('Open3D controls: drag=rotate  Ctrl+drag=pan  scroll=zoom  Q/Esc=quit')
    if args.playback_fps > 0:
        print(f'Playback is capped at {args.playback_fps:.2f} FPS')

    preprocess_ms_list = []
    inference_ms_list = []
    render_ms_list = []
    process_total_ms_list = []
    display_total_ms_list = []
    det_count_list = []

    frames_processed = 0
    sequence_start = time.perf_counter()
    frame_index = 0

    try:
        while True:
            frame_path = frame_paths[frame_index]
            frame_start = time.perf_counter()

            preprocess_start = time.perf_counter()
            points = convert_points_to_camera_frame(load_points(frame_path), args.input_frame)
            gt_boxes_rect, _ = load_gt_boxes(frame_path)
            pts_input_t = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            inference_start = time.perf_counter()
            pred_boxes_rect, scores, class_ids = run_inference(
                model,
                pts_input_t,
                score_thresh=args.score_thresh,
                nms_thresh=nms_thresh,
                use_cuda=use_cuda,
                iou3d_utils=iou3d_utils,
            )
            synchronize_if_needed(use_cuda)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            detected_names = [class_names[int(class_id)] for class_id in class_ids]

            render_start = time.perf_counter()
            update_point_cloud(point_cloud, points)
            vis.update_geometry(point_cloud)
            if gt_box_lines is not None:
                vis.remove_geometry(gt_box_lines, reset_bounding_box=False)
                gt_box_lines = None
            if box_lines is not None:
                vis.remove_geometry(box_lines, reset_bounding_box=False)
                box_lines = None

            gt_box_lines = make_box_lineset(gt_boxes_rect, ['gt'] * len(gt_boxes_rect))
            if gt_box_lines is not None:
                vis.add_geometry(gt_box_lines, reset_bounding_box=False)
                vis.update_geometry(gt_box_lines)

            box_lines = make_box_lineset(pred_boxes_rect, detected_names)
            if box_lines is not None:
                vis.add_geometry(box_lines, reset_bounding_box=False)
                vis.update_geometry(box_lines)
            keep_running = vis.poll_events()
            vis.update_renderer()
            render_ms = (time.perf_counter() - render_start) * 1000.0

            process_total_ms = (time.perf_counter() - frame_start) * 1000.0

            if args.playback_fps > 0:
                target_seconds = 1.0 / args.playback_fps
                elapsed_seconds = time.perf_counter() - frame_start
                if elapsed_seconds < target_seconds:
                    time.sleep(target_seconds - elapsed_seconds)

            display_total_ms = (time.perf_counter() - frame_start) * 1000.0

            preprocess_ms_list.append(preprocess_ms)
            inference_ms_list.append(inference_ms)
            render_ms_list.append(render_ms)
            process_total_ms_list.append(process_total_ms)
            display_total_ms_list.append(display_total_ms)
            det_count_list.append(len(pred_boxes_rect))
            frames_processed += 1

            inst_process_fps = 1000.0 / max(process_total_ms, 1e-6)
            avg_process_fps = 1000.0 / max(np.mean(process_total_ms_list), 1e-6)
            avg_display_fps = 1000.0 / max(np.mean(display_total_ms_list), 1e-6)
            print(
                f'\rFrame {frames_processed:05d} | sample={frame_path.stem} | det={len(pred_boxes_rect):02d} '
                f'| infer={inference_ms:7.2f} ms | proc_fps={inst_process_fps:6.2f} '
                f'| avg_proc={avg_process_fps:6.2f} | avg_display={avg_display_fps:6.2f}',
                end='',
                flush=True,
            )

            if not keep_running:
                break

            frame_index += 1
            if frame_index >= len(frame_paths):
                if args.loop:
                    frame_index = 0
                else:
                    break
    finally:
        print()
        vis.destroy_window()

    elapsed_seconds = time.perf_counter() - sequence_start
    summary = {
        'frames_processed': int(frames_processed),
        'elapsed_seconds': float(elapsed_seconds),
        'avg_detections_per_frame': float(np.mean(det_count_list)) if det_count_list else 0.0,
        'input_frame': args.input_frame,
        'playback_fps_cap': float(args.playback_fps),
        'timings': {
            'preprocess': summarize_ms(preprocess_ms_list),
            'inference': summarize_ms(inference_ms_list),
            'render': summarize_ms(render_ms_list),
            'processing_total': summarize_ms(process_total_ms_list),
            'display_total': summarize_ms(display_total_ms_list),
        },
        'frames': [str(path) for path in frame_paths],
    }
    return summary


def maybe_save_summary(summary, save_json: str):
    if save_json is None:
        return

    output_path = Path(save_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Saved FPS summary to {output_path}')


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_path = resolve_existing_path(args.cfg_file)
    ckpt_path = resolve_existing_path(args.ckpt)
    frame_paths = resolve_frame_paths(args)

    cfg_from_file(str(cfg_path))
    if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
        cfg.RCNN.ROI_SAMPLE_JIT = True

    class_names = get_class_names()
    npoints = args.npoints if args.npoints is not None else int(cfg.RPN.NUM_POINTS)
    nms_thresh = args.nms_thresh if args.nms_thresh is not None else float(cfg.RCNN.NMS_THRESH)

    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')

    model, matched_state, current_state = build_model(str(ckpt_path), class_names, device, use_cuda)

    print(f'Checkpoint      : {ckpt_path}')
    print(f'Config          : {cfg_path}')
    print(f'Frames matched  : {len(frame_paths)}')
    print(f'Input frame     : {args.input_frame}')
    print(f'Using CUDA      : {use_cuda}')
    print(f'RPN.NUM_POINTS  : {npoints}')
    print(f'RCNN thresholds : score={args.score_thresh} nms={nms_thresh}')
    print(f'Matched keys    : {len(matched_state)}/{len(current_state)}')
    print(f'PC_AREA_SCOPE   : {np.asarray(cfg.PC_AREA_SCOPE).tolist()}')

    is_sequence = args.bin_dir is not None or args.bin_glob is not None or len(frame_paths) > 1 or args.loop
    if is_sequence:
        summary = run_sequence_visualization(model, frame_paths, class_names, args, npoints, nms_thresh, use_cuda, iou3d_utils)
        print_sequence_summary(summary)
        maybe_save_summary(summary, args.save_json)
        return

    points = convert_points_to_camera_frame(load_points(frame_paths[0]), args.input_frame)
    warmup_model(model, points, npoints, args.score_thresh, nms_thresh, use_cuda, iou3d_utils, args.warmup)

    start_time = time.perf_counter()
    pts_input_t = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
    pred_boxes_rect, scores, class_ids = run_inference(
        model,
        pts_input_t,
        score_thresh=args.score_thresh,
        nms_thresh=nms_thresh,
        use_cuda=use_cuda,
        iou3d_utils=iou3d_utils,
    )
    synchronize_if_needed(use_cuda)
    total_ms = (time.perf_counter() - start_time) * 1000.0

    sample_id = frame_paths[0].stem
    gt_boxes_rect, gt_label_path = load_gt_boxes(frame_paths[0])
    print(f'Frame           : {sample_id}')
    print(f'Point cloud     : {frame_paths[0]}')
    print(f'Single-frame FPS: {1000.0 / max(total_ms, 1e-6):.2f}')

    if len(pred_boxes_rect) == 0:
        print('No detections above threshold.')
        visualize(
            points,
            np.zeros((0, 7), dtype=np.float32),
            scores,
            [],
            (not args.no_axes),
            float(args.axis_size),
            float(args.point_size),
            gt_boxes_rect=gt_boxes_rect,
            gt_label_path=gt_label_path,
        )
        return

    detected_names = [class_names[int(class_id)] for class_id in class_ids]
    visualize(
        points,
        pred_boxes_rect,
        scores,
        detected_names,
        (not args.no_axes),
        float(args.axis_size),
        float(args.point_size),
        gt_boxes_rect=gt_boxes_rect,
        gt_label_path=gt_label_path,
    )


if __name__ == '__main__':
    main()
