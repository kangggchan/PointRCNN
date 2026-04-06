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
import lib.utils.calibration as calibration_module
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


def _is_identity_calibration(calib) -> bool:
    return (
        np.allclose(calib.R0, np.eye(3, dtype=np.float32), atol=1e-6)
        and np.allclose(
            calib.V2C,
            np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            atol=1e-6,
        )
    )


def _resolve_pc_area_scope(class_names, calib):
    if _is_identity_calibration(calib):
        if [name.lower() for name in class_names] == ['car']:
            return np.array([[-40.0, 40.0], [-3.0, 3.0], [-70.4, 70.4]], dtype=np.float32)
        return np.array([[-50.0, 50.0], [-3.0, 3.0], [-50.0, 50.0]], dtype=np.float32)
    return np.asarray(cfg.PC_AREA_SCOPE, dtype=np.float32)


def _sample_points_like_dataset(pts_rect, pts_intensity, npoints):
    num_pts = len(pts_rect)
    if num_pts == 0:
        ret_pts_rect = np.zeros((npoints, 3), dtype=np.float32)
        ret_pts_intensity = np.zeros((npoints,), dtype=np.float32)
        return ret_pts_rect, ret_pts_intensity

    if npoints < num_pts:
        pts_depth = pts_rect[:, 2]
        pts_near_flag = pts_depth < 40.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]

        near_need = max(npoints - len(far_idxs_choice), 0)
        if near_need > 0:
            if len(near_idxs) > 0:
                near_idxs_choice = np.random.choice(near_idxs, near_need, replace=(near_need > len(near_idxs)))
            else:
                near_idxs_choice = np.random.choice(
                    far_idxs_choice,
                    near_need,
                    replace=(near_need > len(far_idxs_choice)),
                )
        else:
            near_idxs_choice = np.array([], dtype=np.int32)

        if len(far_idxs_choice) > 0:
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
        else:
            choice = near_idxs_choice

        if len(choice) < npoints:
            pad_choice = np.random.choice(choice, npoints - len(choice), replace=True)
            choice = np.concatenate((choice, pad_choice), axis=0)
        elif len(choice) > npoints:
            choice = np.random.choice(choice, npoints, replace=False)

        np.random.shuffle(choice)
        ret_pts_rect = pts_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5
        return ret_pts_rect, ret_pts_intensity

    choice = np.arange(0, num_pts, dtype=np.int32)
    if npoints > num_pts:
        extra_choice = np.random.choice(choice, npoints - num_pts, replace=True)
        choice = np.concatenate((choice, extra_choice), axis=0)
    np.random.shuffle(choice)

    ret_pts_rect = pts_rect[choice, :]
    ret_pts_intensity = pts_intensity[choice] - 0.5
    return ret_pts_rect, ret_pts_intensity


def _rect_to_lidar(points_rect: np.ndarray, calib) -> np.ndarray:
    v2c_4x4 = np.eye(4, dtype=np.float32)
    v2c_4x4[:3, :4] = calib.V2C

    r0_4x4 = np.eye(4, dtype=np.float32)
    r0_4x4[:3, :3] = calib.R0

    rect_to_lidar = np.linalg.inv(r0_4x4 @ v2c_4x4)
    points_hom = np.hstack([points_rect, np.ones((points_rect.shape[0], 1), dtype=np.float32)])
    return (points_hom @ rect_to_lidar.T)[:, :3]


def should_use_cuda(no_cuda: bool) -> bool:
    """Determine whether to use CUDA based on availability and arguments."""
    if no_cuda or not torch.cuda.is_available():
        return False
    return True


def ensure_iou3d_utils(use_cuda: bool):
    """Load iou3d_utils and return it with updated use_cuda flag."""
    try:
        import lib.utils.iou3d.iou3d_utils as iou3d_utils
        return iou3d_utils, use_cuda
    except (ImportError, ModuleNotFoundError):
        print('Warning: iou3d_utils not available, falling back to CPU NMS')
        return None, False


def adapt_cfg_from_checkpoint(model_state):
    """Adapt config from checkpoint (e.g., num_classes based on final layer)."""
    # Check if the classification layer exists and adapt num_classes if needed
    cls_layer_keys = [k for k in model_state.keys() if 'rcnn_net.cls_layer' in k]
    if cls_layer_keys:
        for key in cls_layer_keys:
            if 'weight' in key and key.endswith('.weight'):
                # Last conv layer weight has shape [num_classes, ...]
                num_classes = model_state[key].shape[0]
                if hasattr(cfg, 'NUM_CLASSES'):
                    cfg.NUM_CLASSES = num_classes
                break


def preprocess(pts_lidar_raw: np.ndarray, calib, img_shape, npoints: int, use_intensity: bool, class_names):
    """
    Preprocess point cloud: transform to rect coords, filter by range, sample points.
    
    Args:
        pts_lidar_raw: (N, 4) array of [x, y, z, intensity]
        calib: Calibration object
        img_shape: (height, width, channels)
        npoints: Number of points to sample
        use_intensity: Whether to include intensity in output
    
    Returns:
        pts_input_t: torch tensor of shape (1, npoints, 3 or 4)
        pts_rect: (N', 3 or 4) preprocessed points
    """
    pts_rect = calib.lidar_to_rect(pts_lidar_raw[:, 0:3])

    pc_range = _resolve_pc_area_scope(class_names, calib)
    x_range, y_range, z_range = pc_range[0], pc_range[1], pc_range[2]

    valid_mask = (
        (pts_rect[:, 0] >= x_range[0]) & (pts_rect[:, 0] <= x_range[1]) &
        (pts_rect[:, 1] >= y_range[0]) & (pts_rect[:, 1] <= y_range[1]) &
        (pts_rect[:, 2] >= z_range[0]) & (pts_rect[:, 2] <= z_range[1])
    )
    pts_rect = pts_rect[valid_mask]

    pts_intensity = pts_lidar_raw[valid_mask, 3]
    ret_pts_rect, ret_pts_intensity = _sample_points_like_dataset(pts_rect, pts_intensity, npoints)

    if use_intensity:
        pts_input = np.concatenate((ret_pts_rect, ret_pts_intensity.reshape(-1, 1)), axis=1)
    else:
        pts_input = ret_pts_rect

    pts_input_t = torch.from_numpy(pts_input[np.newaxis, ...]).float()

    return pts_input_t, pts_rect


def boxes_rect_to_lidar_corners(boxes_rect: np.ndarray, calib) -> np.ndarray:
    """
    Convert 3D boxes in rect coords to corner points in lidar coords.
    
    Args:
        boxes_rect: (N, 7) array of [x, y, z, h, w, l, ry]
        calib: Calibration object
    
    Returns:
        corners_lidar: (N, 8, 3) corner points in lidar coords
    """
    corners_rect = kitti_utils.boxes3d_to_corners3d(boxes_rect)  # (N, 8, 3)
    corners_lidar = _rect_to_lidar(corners_rect.reshape(-1, 3), calib).reshape(boxes_rect.shape[0], 8, 3)
    return corners_lidar


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a PointRCNN checkpoint on one dataset frame and visualize detections in Open3D.'
    )
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml',
                        help='Training/evaluation config file.')
    parser.add_argument('--ckpt', type=str, default='../output/rcnn/default/ckpt/checkpoint_epoch_4.pth',
                        help='Checkpoint to load.')
    parser.add_argument('--data_root', type=str, default='../data/dataset',
                        help='Dataset root used for training.')
    parser.add_argument('--frame_id', type=str, default='000355',
                        help='Frame id inside the dataset, for example 000010 or 10.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'smallval', 'trainval', 'test'],
                        help='KITTI split to resolve the frame from when using data_root.')
    parser.add_argument('--bin_file', type=str, default=None,
                        help='Optional direct path to a .bin file. Overrides data_root/frame_id lookup.')
    parser.add_argument('--calib_file', type=str, default=None,
                        help='Optional direct path to calibration .txt. Overrides dataset lookup.')
    parser.add_argument('--img_height', type=int, default=720,
                        help='Image height used by the projection filter when no image exists.')
    parser.add_argument('--img_width', type=int, default=1280,
                        help='Image width used by the projection filter when no image exists.')
    parser.add_argument('--score_thresh', type=float, default=0.1,
                        help='Minimum class confidence kept after RCNN scoring.')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='Optional NMS threshold override. Defaults to cfg.RCNN.NMS_THRESH.')
    parser.add_argument('--npoints', type=int, default=None,
                        help='Optional point sampling override. Defaults to cfg.RPN.NUM_POINTS.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Force CPU inference.')
    return parser.parse_args()


def _torch_load_compat(filename, map_location):
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=map_location)


def normalize_frame_id(frame_id: str) -> str:
    frame_text = str(frame_id).strip()
    if frame_text.isdigit():
        return f'{int(frame_text):06d}'
    stem = Path(frame_text).stem
    if stem.isdigit():
        return f'{int(stem):06d}'
    raise ValueError(f'Invalid frame id: {frame_id}')


def resolve_frame_paths(data_root: str, frame_id: str, split: str, bin_override: str, calib_override: str):
    if bin_override is not None:
        bin_path = Path(bin_override).expanduser().resolve()
        if not bin_path.exists():
            raise FileNotFoundError(f'Point cloud file not found: {bin_path}')
        calib_path = None
        if calib_override is not None:
            calib_path = Path(calib_override).expanduser().resolve()
            if not calib_path.exists():
                raise FileNotFoundError(f'Calibration file not found: {calib_path}')
        return bin_path, calib_path, bin_path.stem

    sample_id = normalize_frame_id(frame_id)
    root = Path(data_root).expanduser().resolve()
    object_split = 'testing' if split == 'test' else 'training'

    candidate_bins = [
        root / 'KITTI' / 'object' / object_split / 'velodyne' / f'{sample_id}.bin',
        root / 'bin' / f'{sample_id}.bin',
    ]
    bin_path = next((path for path in candidate_bins if path.exists()), None)
    if bin_path is None:
        checked = '\n'.join(str(path) for path in candidate_bins)
        raise FileNotFoundError(f'Unable to find frame {sample_id}. Checked:\n{checked}')

    if calib_override is not None:
        calib_path = Path(calib_override).expanduser().resolve()
        if not calib_path.exists():
            raise FileNotFoundError(f'Calibration file not found: {calib_path}')
    else:
        candidate_calibs = [
            root / 'KITTI' / 'object' / object_split / 'calib' / f'{sample_id}.txt',
        ]
        calib_path = next((path for path in candidate_calibs if path.exists()), None)

    return bin_path, calib_path, sample_id


def resolve_image_shape(data_root: str, sample_id: str, split: str, default_height: int, default_width: int):
    root = Path(data_root).expanduser().resolve()
    object_split = 'testing' if split == 'test' else 'training'
    image_path = root / 'KITTI' / 'object' / object_split / 'image_2' / f'{sample_id}.png'
    if not image_path.exists():
        return default_height, default_width, 3

    from PIL import Image
    image = Image.open(image_path)
    width, height = image.size
    return height, width, 3


def get_class_names():
    class_names = KittiRCNNDataset.parse_classes(cfg.CLASSES)
    if not class_names:
        raise ValueError(f'No classes found in cfg.CLASSES: {cfg.CLASSES}')
    return ['Background'] + class_names


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
    batch_size = roi_boxes3d.shape[0]

    class_ids, cls_scores, raw_scores, keep_mask = _select_class_scores(rcnn_cls, score_thresh)
    if cfg.RCNN.SIZE_RES_ON_ROI:
        anchor_size = roi_boxes3d.view(-1, 7)[:, 3:6]
    else:
        anchor_size = kitti_utils.get_class_anchor_sizes_torch(class_ids.view(-1), device=roi_boxes3d.device)

    pred_boxes3d = decode_bbox_target(
        roi_boxes3d.view(-1, 7),
        rcnn_reg.view(-1, rcnn_reg.shape[-1]),
        anchor_size=anchor_size,
        loc_scope=cfg.RCNN.LOC_SCOPE,
        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
        get_xz_fine=True,
        get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine=True,
    ).view(batch_size, -1, 7)

    if keep_mask.sum() == 0:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

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

        if iou3d_utils is None:
            keep_idx = torch.argsort(class_raw_scores, descending=True)
        else:
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(class_boxes)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev, class_raw_scores, nms_thresh).view(-1)

        final_boxes.append(class_boxes[keep_idx])
        final_scores.append(class_scores[keep_idx])
        final_class_ids.append(torch.full((keep_idx.numel(),), int(class_id.item()), dtype=torch.long,
                                          device=class_boxes.device))

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_class_ids = torch.cat(final_class_ids, dim=0)

    sort_order = torch.argsort(final_scores, descending=True)
    final_boxes = final_boxes[sort_order].cpu().numpy()
    final_scores = final_scores[sort_order].cpu().numpy()
    final_class_ids = final_class_ids[sort_order].cpu().numpy()
    return final_boxes, final_scores, final_class_ids


def make_box_lineset(corners: np.ndarray, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    line_set.colors = o3d.utility.Vector3dVector([list(color)] * len(BOX_EDGES))
    return line_set


def visualize(pts_lidar_raw: np.ndarray, boxes_lidar: np.ndarray, scores: np.ndarray, class_names: list):
    xyz = pts_lidar_raw[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    ranges = np.linalg.norm(xyz[:, :2], axis=1)
    color_mix = np.clip(ranges / max(np.percentile(ranges, 95), 1e-3), 0.0, 1.0)
    point_colors = np.stack([0.2 + 0.7 * color_mix, 0.75 - 0.5 * color_mix, 1.0 - 0.6 * color_mix], axis=1)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)

    geometries = [point_cloud]
    for idx, (corners, score, class_name) in enumerate(zip(boxes_lidar, scores, class_names), start=1):
        color = CLASS_COLORS.get(class_name.lower(), (1.0, 1.0, 1.0))
        geometries.append(make_box_lineset(corners, color))
        print(f'  Det {idx:02d}: class={class_name:<11} score={score:.3f}')

    print(f'\nVisualizing {len(boxes_lidar)} detections')
    print('Open3D controls: drag=rotate  Ctrl+drag=pan  scroll=zoom  Q/Esc=quit')

    o3d.visualization.draw_geometries(
        geometries,
        window_name='PointRCNN Frame Inference',
        width=1400,
        height=900,
    )


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_from_file(args.cfg_file)
    class_names = get_class_names()
    num_classes = len(class_names)
    npoints = args.npoints if args.npoints is not None else int(cfg.RPN.NUM_POINTS)
    nms_thresh = args.nms_thresh if args.nms_thresh is not None else float(cfg.RCNN.NMS_THRESH)

    bin_path, calib_path, sample_id = resolve_frame_paths(
        args.data_root,
        args.frame_id,
        args.split,
        args.bin_file,
        args.calib_file,
    )
    image_shape = resolve_image_shape(args.data_root, sample_id, args.split, args.img_height, args.img_width)

    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')

    checkpoint = _torch_load_compat(args.ckpt, map_location=device)
    model_state = checkpoint.get('model_state', checkpoint)
    adapt_cfg_from_checkpoint(model_state)

    model = PointRCNN(num_classes=num_classes, use_xyz=True, mode='TEST')
    if use_cuda:
        model.cuda()
    model.eval()

    current_state = model.state_dict()
    matched_state = {key: value for key, value in model_state.items() if key in current_state and current_state[key].shape == value.shape}
    current_state.update(matched_state)
    model.load_state_dict(current_state)

    if 'rcnn_net.cls_layer.3.conv.weight' in current_state and 'rcnn_net.cls_layer.3.conv.weight' not in matched_state:
        raise RuntimeError('Checkpoint classification head does not match cfg.CLASSES. Check cfg_file and checkpoint.')

    calib = calibration_module.Calibration(str(calib_path)) if calib_path is not None else None
    if calib is None:
        raise FileNotFoundError('No calibration file was found. Supply --calib_file or use a preprocessed dataset root.')

    pts_lidar_raw = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    pts_input_t, _ = preprocess(
        pts_lidar_raw,
        calib,
        img_shape=image_shape,
        npoints=npoints,
        use_intensity=cfg.RPN.USE_INTENSITY,
        class_names=class_names[1:],
    )

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
    print(f'Calibration: {calib_path}')
    print(f'Classes    : {class_names[1:]}')
    print(f'Matched keys: {len(matched_state)}/{len(current_state)}')

    if len(pred_boxes_rect) == 0:
        print('No detections above threshold.')
        visualize(pts_lidar_raw, np.zeros((0, 8, 3), dtype=np.float32), scores, [])
        return

    detected_names = [class_names[int(class_id)] for class_id in class_ids]
    corners_lidar = boxes_rect_to_lidar_corners(pred_boxes_rect, calib)
    visualize(pts_lidar_raw, corners_lidar, scores, detected_names)


if __name__ == '__main__':
    main()