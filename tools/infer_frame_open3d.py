import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import argparse
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


CLASS_COLORS = {
    'background': (0.8, 0.8, 0.8),
    'car': (1.0, 0.45, 0.1),
    'human': (0.1, 0.85, 0.25),
    'forklift': (1.0, 0.2, 0.2),
    'cargobike': (0.15, 0.55, 1.0),
}

BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def parse_args():
    parser = argparse.ArgumentParser(description='Simple PointRCNN frame inference + Open3D visualization.')
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='Inference config file.')
    parser.add_argument('--ckpt', type=str, default='../output/rcnn/default/ckpt/checkpoint_epoch_25.pth',
                        help='Checkpoint to load.')
    parser.add_argument('--bin_file', type=str, default='../data/dataset/KITTI/aug_scene/training/rectified_data/010212.bin',
                        help='Direct path to a .bin file in camera frame.')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='Minimum confidence threshold.')
    parser.add_argument('--nms_thresh', type=float, default=None, help='NMS threshold override.')
    parser.add_argument('--npoints', type=int, default=None, help='Point sampling override.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    parser.add_argument('--no_axes', action='store_true', help='Hide coordinate axes at origin.')
    parser.add_argument('--axis_size', type=float, default=2.5, help='Coordinate frame axis length.')
    return parser.parse_args()


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
    # Calibration-free path: bins are expected to already be in camera frame.
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
        pts_input_t = pts_input_t.cuda()

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


def make_box_lineset(corners: np.ndarray, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    line_set.colors = o3d.utility.Vector3dVector([list(color)] * len(BOX_EDGES))
    return line_set


def visualize(points: np.ndarray, boxes_rect: np.ndarray, scores: np.ndarray, class_names: list,
              show_axes: bool, axis_size: float):
    xyz = points[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    ranges = np.linalg.norm(xyz[:, :2], axis=1)
    color_mix = np.clip(ranges / max(np.percentile(ranges, 95), 1e-3), 0.0, 1.0)
    point_colors = np.stack([0.2 + 0.7 * color_mix, 0.75 - 0.5 * color_mix, 1.0 - 0.6 * color_mix], axis=1)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    geometries = [point_cloud]
    if show_axes:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0]))

    corners_rect = kitti_utils.boxes3d_to_corners3d(boxes_rect) if len(boxes_rect) > 0 else np.zeros((0, 8, 3), dtype=np.float32)
    for idx, (corners, score, class_name) in enumerate(zip(corners_rect, scores, class_names), start=1):
        color = CLASS_COLORS.get(class_name.lower(), (1.0, 1.0, 1.0))
        geometries.append(make_box_lineset(corners, color))
        print(f'  Det {idx:02d}: class={class_name:<11} score={score:.3f}')

    print(f'\nVisualizing {len(corners_rect)} detections (camera-frame view)')
    print('Open3D controls: drag=rotate  Ctrl+drag=pan  scroll=zoom  Q/Esc=quit')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointRCNN Simple Inference', width=1400, height=900)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.04, 0.05, 0.09])

    for geometry in geometries:
        vis.add_geometry(geometry)

    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_from_file(args.cfg_file)
    if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
        cfg.RCNN.ROI_SAMPLE_JIT = True

    class_names = get_class_names()
    npoints = args.npoints if args.npoints is not None else int(cfg.RPN.NUM_POINTS)
    nms_thresh = args.nms_thresh if args.nms_thresh is not None else float(cfg.RCNN.NMS_THRESH)

    bin_path = Path(args.bin_file).expanduser().resolve()
    if not bin_path.exists():
        raise FileNotFoundError(f'Point cloud file not found: {bin_path}')
    sample_id = bin_path.stem
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')

    model, matched_state, current_state = build_model(args.ckpt, class_names, device, use_cuda)
    pts_input_t = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)

    pred_boxes_rect, scores, class_ids = run_inference(
        model,
        pts_input_t,
        score_thresh=args.score_thresh,
        nms_thresh=nms_thresh,
        use_cuda=use_cuda,
        iou3d_utils=iou3d_utils,
    )

    print(f'Checkpoint : {Path(args.ckpt).resolve()}')
    print(f'Frame      : {sample_id}')
    print(f'Point cloud: {bin_path}')
    print(f'Classes    : {class_names[1:]}')
    print(f'Matched keys: {len(matched_state)}/{len(current_state)}')
    print(f'PC_AREA_SCOPE: {np.asarray(cfg.PC_AREA_SCOPE).tolist()}')

    if len(pred_boxes_rect) == 0:
        print('No detections above threshold.')
        visualize(points, np.zeros((0, 7), dtype=np.float32), scores, [], (not args.no_axes), float(args.axis_size))
        return

    detected_names = [class_names[int(class_id)] for class_id in class_ids]
    visualize(points, pred_boxes_rect, scores, detected_names, (not args.no_axes), float(args.axis_size))


if __name__ == '__main__':
    main()
