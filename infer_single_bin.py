"""
infer_single_bin.py
====================
Run PointRCNN inference on a single KITTI-format .bin file and visualise the
detected cars with Open3D.

Usage example
-------------
cd /path/to/PointRCNN/tools

python infer_single_bin.py \
    --bin_file  /path/to/000042.bin          \
    --ckpt      ../PointRCNN.pth             \
    --calib_file /path/to/000042.txt         \
    --cfg_file  cfgs/default.yaml            \
    --score_thresh 0.3

If --calib_file is omitted a default KITTI calibration is used as fallback
(accuracy may be reduced).

Input format (.bin)
-------------------
Standard KITTI Velodyne format:
  dtype  : float32
  shape  : (N, 4)   columns = [x, y, z, intensity]  in LiDAR coordinates
  → No conversion needed; only the calibration-based LiDAR→rect transform
    plus range/image-projection filtering is applied before feeding the model.
"""

import os
import sys
import argparse


def _bootstrap_paths():
    """Ensure PointRCNN modules are importable from both root and tools runs."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        this_dir,
        os.path.join(this_dir, 'tools')
    ]

    for candidate in candidate_dirs:
        init_file = os.path.join(candidate, '_init_path.py')
        if os.path.isfile(init_file):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            import _init_path  # noqa: F401
            return

    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    datasets_dir = os.path.join(this_dir, 'lib', 'datasets')
    net_dir = os.path.join(this_dir, 'lib', 'net')
    if datasets_dir not in sys.path:
        sys.path.insert(0, datasets_dir)
    if net_dir not in sys.path:
        sys.path.insert(0, net_dir)


_bootstrap_paths()

import numpy as np
import torch
import open3d as o3d
import types

from lib.config import cfg, cfg_from_file
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.kitti_utils as kitti_utils
import lib.utils.calibration as calibration_module


def should_use_cuda(force_cpu: bool) -> bool:
    """
    Return True only when CUDA is both requested and usable.
    Falls back to CPU automatically for unsupported GPU architectures.
    """
    if force_cpu:
        print('[INFO] --no_cuda specified. Using CPU inference.')
        return False

    if not torch.cuda.is_available():
        print('[WARNING] CUDA is not available. Using CPU inference.')
        return False

    try:
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        target_arch = f'sm_{cap_major}{cap_minor}'
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and target_arch not in supported_arches:
            print('[WARNING] CUDA device architecture is not supported by this '
                  f'PyTorch build ({target_arch} not in {sorted(supported_arches)}).')
            print('[WARNING] Falling back to CPU inference.')
            return False

        # Small runtime probe to catch other CUDA runtime init issues.
        _ = torch.zeros(1, device='cuda')
    except Exception as err:
        print(f'[WARNING] CUDA runtime check failed: {err}')
        print('[WARNING] Falling back to CPU inference.')
        return False

    return True


def _nms_cpu_bev(boxes_bev: torch.Tensor, scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """
    CPU NMS fallback for boxes in BEV format [x1, y1, x2, y2, ry].
    Rotation term is ignored in fallback mode.
    """
    if boxes_bev.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes_bev.device)

    x1 = boxes_bev[:, 0]
    y1 = boxes_bev[:, 1]
    x2 = boxes_bev[:, 2]
    y2 = boxes_bev[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = torch.argsort(scores, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = inter / torch.clamp(union, min=1e-6)

        order = rest[iou <= thresh]

    return torch.stack(keep).long()


def ensure_iou3d_utils(use_cuda: bool):
    """
    Load CUDA iou3d utils when available.
    If unavailable and running CPU mode, inject a minimal CPU fallback module.
    """
    try:
        import lib.utils.iou3d.iou3d_utils as iou3d_utils
        return iou3d_utils, use_cuda
    except Exception as err:
        if use_cuda:
            print('[WARNING] CUDA iou3d extension failed to import. '
                  'Automatically switching to CPU inference mode.')
            print(f'[WARNING] Original CUDA extension error: {err}')
        else:
            print('[WARNING] CUDA iou3d extension not available in CPU mode; '
                  'using CPU NMS fallback (axis-aligned BEV).')

        fallback = types.ModuleType('lib.utils.iou3d.iou3d_utils')

        def nms_gpu(boxes, scores, thresh):
            return _nms_cpu_bev(boxes, scores, thresh)

        def nms_normal_gpu(boxes, scores, thresh):
            return _nms_cpu_bev(boxes, scores, thresh)

        def boxes_iou3d_gpu(_boxes_a, _boxes_b):
            raise NotImplementedError('boxes_iou3d_gpu is unavailable in CPU fallback mode')

        fallback.nms_gpu = nms_gpu
        fallback.nms_normal_gpu = nms_normal_gpu
        fallback.boxes_iou3d_gpu = boxes_iou3d_gpu

        sys.modules['lib.utils.iou3d.iou3d_utils'] = fallback
        return fallback, False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='PointRCNN inference on a single KITTI .bin + Open3D visualisation')

    parser.add_argument('--bin_file', type=str, default="../data/scala2/scala2_data/frame_000027.bin",
                        help='Path to the KITTI-format LiDAR .bin file '
                             '(float32, shape N×4: x,y,z,intensity in LiDAR coords)')
    parser.add_argument('--ckpt', type=str, default='../output/rcnn/default/ckpt/checkpoint_epoch_70.pth',
                        help='Path to the PointRCNN checkpoint (.pth)')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/default.yaml',
                        help='Config YAML file (default: tools/cfgs/default.yaml)')
    parser.add_argument('--calib_file', type=str, default=None,
                        help='Path to the matching KITTI calibration .txt file. '
                             'Required for the correct LiDAR→rect coordinate '
                             'transform.  If omitted a typical KITTI calib is '
                             'used as fallback (accuracy may be reduced).')
    parser.add_argument('--img_height', type=int, default=375,
                        help='Camera image height for projection filter '
                             '(default=375, KITTI standard)')
    parser.add_argument('--img_width', type=int, default=1242,
                        help='Camera image width for projection filter '
                             '(default=1242, KITTI standard)')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Min confidence score to keep a detection (default=0.3)')
    parser.add_argument('--nms_thresh', type=float, default=0.1,
                        help='NMS IoU threshold (default=0.1)')
    parser.add_argument('--npoints', type=int, default=16384,
                        help='Points to sample – must match training (default=16384)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Force CPU inference (not recommended)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Default calibration fallback
# Approximate KITTI sequence-0000 calibration used when no file is supplied.
# ---------------------------------------------------------------------------

_DEFAULT_CALIB = {
    'P2': np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02,  4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02,  2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00,  2.745884e-03],
    ], dtype=np.float32),
    'P3': np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, -3.395242e+02],
        [0.000000e+00, 7.215377e+02, 1.728540e+02,  2.199616e+00],
        [0.000000e+00, 0.000000e+00, 1.000000e+00,  2.729905e-03],
    ], dtype=np.float32),
    'R0': np.array([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01],
    ], dtype=np.float32),
    'Tr_velo2cam': np.array([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
    ], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rect_to_lidar(pts_rect: np.ndarray, calib) -> np.ndarray:
    """
    Inverse of calib.lidar_to_rect():
      lidar_to_rect:   pts_rect = pts_lidar_hom @ V2C.T @ R0.T

    Step 1 – undo R0  → unrectified camera coords (R0 is orthogonal, inv = R0.T)
    Step 2 – undo V2C → LiDAR coords
    """
    # Step 1
    pts_cam = (calib.R0.T @ pts_rect.T).T          # (N, 3)

    # Step 2:  pts_cam = pts_lidar @ R_vc.T + t_vc
    #          pts_lidar = (pts_cam - t_vc) @ inv(R_vc).T
    R_vc = calib.V2C[:, :3]                         # (3, 3)
    t_vc = calib.V2C[:, 3]                          # (3,)
    pts_lidar = (pts_cam - t_vc) @ np.linalg.inv(R_vc).T  # (N, 3)
    return pts_lidar


# ---------------------------------------------------------------------------
# Preprocessing  (mirrors kitti_rcnn_dataset.get_rpn_sample for TEST mode)
# ---------------------------------------------------------------------------

def preprocess(pts_lidar_raw: np.ndarray, calib, img_shape, npoints, use_intensity):
    """
    Replicate the exact preprocessing used during training / evaluation.

    Parameters
    ----------
    pts_lidar_raw : (N, 4) float32   x,y,z,intensity in LiDAR coords
    calib         : Calibration object
    img_shape     : (H, W, C)
    npoints       : target number of points
    use_intensity : whether to append intensity to pts_input (cfg.RPN.USE_INTENSITY)

    Returns
    -------
    pts_input_t      : torch.Tensor (1, npoints, C)  ready for model.forward()
    pts_rect_sampled : np.ndarray   (npoints, 3)     sampled rect-coord points
    """
    pts_xyz       = pts_lidar_raw[:, :3]
    pts_intensity = pts_lidar_raw[:, 3]

    # ── 1. LiDAR → rectified camera coordinates ──────────────────────────
    pts_rect = calib.lidar_to_rect(pts_xyz)   # (N, 3)

    # ── 2. Image-projection filter (keeps only forward-facing points) ─────
    pts_img, pts_depth = calib.rect_to_img(pts_rect)
    H, W = img_shape[0], img_shape[1]
    val = (
        (pts_img[:, 0] >= 0) & (pts_img[:, 0] < W) &
        (pts_img[:, 1] >= 0) & (pts_img[:, 1] < H) &
        (pts_depth >= 0)
    )

    # ── 3. PC_AREA_SCOPE range filter (in rect coords) ────────────────────
    if cfg.PC_REDUCE_BY_RANGE:
        xr, yr, zr = cfg.PC_AREA_SCOPE
        px, py, pz = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
        val = val & (
            (px >= xr[0]) & (px <= xr[1]) &
            (py >= yr[0]) & (py <= yr[1]) &
            (pz >= zr[0]) & (pz <= zr[1])
        )

    pts_rect      = pts_rect[val]
    pts_intensity = pts_intensity[val]
    n             = len(pts_rect)
    print(f'  After filtering: {n} / {len(pts_lidar_raw)} points remain')

    # ── 4. Sample to npoints ──────────────────────────────────────────────
    if n == 0:
        pts_rect_s = np.zeros((npoints, 3), dtype=np.float32)
        pts_intensity_s = np.zeros((npoints,), dtype=np.float32)
    else:
        if npoints < n:
            depth = pts_rect[:, 2]
            near_mask = depth < 40.0
            far_idxs = np.where(~near_mask)[0]
            near_idxs = np.where(near_mask)[0]

            near_need = max(npoints - len(far_idxs), 0)
            if near_need > 0:
                if len(near_idxs) > 0:
                    near_choice = np.random.choice(near_idxs, near_need,
                                                   replace=(near_need > len(near_idxs)))
                else:
                    near_choice = np.random.choice(far_idxs, near_need,
                                                   replace=(near_need > len(far_idxs)))
            else:
                near_choice = np.array([], dtype=np.int32)

            if len(far_idxs) > 0:
                choice = np.concatenate([near_choice, far_idxs])
            else:
                choice = near_choice

            if len(choice) < npoints:
                pad_choice = np.random.choice(choice, npoints - len(choice), replace=True)
                choice = np.concatenate([choice, pad_choice])
            elif len(choice) > npoints:
                choice = np.random.choice(choice, npoints, replace=False)
        else:
            choice = np.arange(n, dtype=np.int32)
            if npoints > n:
                extra = np.random.choice(choice, npoints - n, replace=True)
                choice = np.concatenate([choice, extra])

        np.random.shuffle(choice)
        pts_rect_s = pts_rect[choice]                # (npoints, 3)
        pts_intensity_s = pts_intensity[choice]      # (npoints,)

    pts_intensity_s = pts_intensity_s - 0.5          # shift → [-0.5, 0.5]

    # ── 5. Build model input ──────────────────────────────────────────────
    # Default config: RPN.USE_INTENSITY = False  →  pts_input = pts_rect (N,3)
    if use_intensity:
        pts_input = np.concatenate(
            [pts_rect_s, pts_intensity_s.reshape(-1, 1)], axis=1)  # (N, 4)
    else:
        pts_input = pts_rect_s                    # (N, 3)

    pts_input_t = torch.from_numpy(pts_input).float().unsqueeze(0)   # (1, N, C)
    return pts_input_t, pts_rect_s


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, pts_input_t, score_thresh, nms_thresh, use_cuda, iou3d_utils):
    """
    Run RPN + RCNN forward pass and return final boxes + scores.

    Returns
    -------
    final_boxes  : (D, 7) numpy  [x, y, z, h, w, l, ry]  in RECT camera coords
    final_scores : (D,)   numpy  confidence in [0, 1]
    """
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0])
    if use_cuda:
        pts_input_t = pts_input_t.cuda()
        MEAN_SIZE   = MEAN_SIZE.cuda()

    with torch.no_grad():
        ret = model({'pts_input': pts_input_t})

    # Shapes with batch_size = 1:
    #   ret['rois']     : (1, M, 7)   RPN proposals  in rect coords
    #   ret['rcnn_cls'] : (M, 1)      raw logit per proposal
    #   ret['rcnn_reg'] : (M, C)      regression residuals
    roi_boxes3d = ret['rois']       # (1, M, 7)
    rcnn_cls    = ret['rcnn_cls']   # (M, 1)
    rcnn_reg    = ret['rcnn_reg']   # (M, C)
    batch_size  = roi_boxes3d.shape[0]

    # ── Decode refined bounding boxes ─────────────────────────────────────
    pred_boxes3d = decode_bbox_target(
        roi_boxes3d.view(-1, 7),
        rcnn_reg.view(-1, rcnn_reg.shape[-1]),
        anchor_size    = MEAN_SIZE,
        loc_scope      = cfg.RCNN.LOC_SCOPE,
        loc_bin_size   = cfg.RCNN.LOC_BIN_SIZE,
        num_head_bin   = cfg.RCNN.NUM_HEAD_BIN,
        get_xz_fine    = True,
        get_y_by_bin   = cfg.RCNN.LOC_Y_BY_BIN,
        loc_y_scope    = cfg.RCNN.LOC_Y_SCOPE,
        loc_y_bin_size = cfg.RCNN.LOC_Y_BIN_SIZE,
        get_ry_fine    = True,
    ).view(batch_size, -1, 7)       # (1, M, 7)

    raw_scores  = rcnn_cls.view(batch_size, -1)     # (1, M)
    norm_scores = torch.sigmoid(raw_scores)         # (1, M)

    # ── Score filter ──────────────────────────────────────────────────────
    keep_flag = norm_scores[0] > score_thresh       # (M,)
    if keep_flag.sum() == 0:
        print('No detections above score threshold.')
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes_f  = pred_boxes3d[0, keep_flag]           # (K, 7)
    scores_f = raw_scores[0, keep_flag]             # (K,)  raw logits for NMS ordering

    # ── NMS ───────────────────────────────────────────────────────────────
    boxes_bev = kitti_utils.boxes3d_to_bev_torch(boxes_f)   # (K, 5)
    keep_idx  = iou3d_utils.nms_gpu(boxes_bev, scores_f, nms_thresh).view(-1)

    final_boxes  = boxes_f[keep_idx].cpu().numpy()                      # (D, 7)
    final_scores = torch.sigmoid(scores_f[keep_idx]).cpu().numpy()      # (D,)
    return final_boxes, final_scores


def _infer_reg_channel(loc_xz_fine: bool) -> int:
    per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
    if loc_xz_fine:
        reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
    else:
        reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
    reg_channel += 1
    return reg_channel


def adapt_cfg_from_checkpoint(model_state) -> bool:
    """
    Try to reconcile config with checkpoint head dimensions.
    Returns True if cfg was changed.
    """
    key = 'rpn.rpn_reg_layer.2.conv.weight'
    if key not in model_state:
        return False

    ckpt_reg_out = int(model_state[key].shape[0])
    expected_now = _infer_reg_channel(cfg.RPN.LOC_XZ_FINE)
    if ckpt_reg_out == expected_now:
        return False

    expected_if_toggle = _infer_reg_channel(not cfg.RPN.LOC_XZ_FINE)
    if ckpt_reg_out == expected_if_toggle:
        old_val = cfg.RPN.LOC_XZ_FINE
        cfg.RPN.LOC_XZ_FINE = not cfg.RPN.LOC_XZ_FINE
        print('[INFO] Auto-adjusted cfg to match checkpoint: '
              f'RPN.LOC_XZ_FINE {old_val} -> {cfg.RPN.LOC_XZ_FINE} '
              f'(checkpoint reg channels: {ckpt_reg_out}).')
        return True

    print('[WARNING] Checkpoint/model reg-head mismatch remains: '
          f'checkpoint={ckpt_reg_out}, '
          f'expected(current_cfg)={expected_now}, '
          f'expected(toggle_LOC_XZ_FINE)={expected_if_toggle}.')
    return False


# ---------------------------------------------------------------------------
# Open3D visualisation helpers
# ---------------------------------------------------------------------------

# 12 edges of a 3-D bounding box
_BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],   # top face
    [0, 4], [1, 5], [2, 6], [3, 7],   # vertical pillars
]


def boxes_rect_to_lidar_corners(boxes3d_rect: np.ndarray, calib) -> np.ndarray:
    """
    Convert 3-D boxes in rect-camera coords to 8 corner points in LiDAR coords.

    boxes3d_rect : (N, 7)  [x, y, z, h, w, l, ry]  rect camera coords
    Returns      : (N, 8, 3)  LiDAR coords
    """
    corners_rect = kitti_utils.boxes3d_to_corners3d(boxes3d_rect, rotate=True)  # (N, 8, 3)
    N            = corners_rect.shape[0]
    flat_lidar   = rect_to_lidar(corners_rect.reshape(-1, 3), calib)             # (N*8, 3)
    return flat_lidar.reshape(N, 8, 3)


def make_box_lineset(corners: np.ndarray, color=(0.0, 1.0, 0.0)):
    """Create an Open3D LineSet from 8 corner points (LiDAR coords)."""
    ls         = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(corners)
    ls.lines   = o3d.utility.Vector2iVector(_BOX_EDGES)
    ls.colors  = o3d.utility.Vector3dVector([list(color)] * len(_BOX_EDGES))
    return ls


def visualize(pts_lidar_raw: np.ndarray,
              corners_lidar: np.ndarray,
              scores: np.ndarray):
    """
    Display the raw point cloud (LiDAR coords) with green bounding boxes.

    pts_lidar_raw  : (N, 4)   full raw point cloud
    corners_lidar  : (D, 8, 3) box corners in LiDAR coords
    scores         : (D,)
    """
    xyz = pts_lidar_raw[:, :3]

    # Point cloud – colour by height
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    z          = xyz[:, 2]
    t          = np.clip((z - z.min()) / (np.ptp(z) + 1e-6), 0, 1)
    rgb        = np.zeros((len(xyz), 3), dtype=np.float32)
    rgb[:, 0]  = t          # red   = high
    rgb[:, 2]  = 1.0 - t    # blue  = low
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geoms = [pcd]
    for i, corners in enumerate(corners_lidar):
        geoms.append(make_box_lineset(corners, color=(0.0, 1.0, 0.0)))
        print(f'  Box {i + 1:2d}  score={scores[i]:.3f}')

    print(f'\n{len(corners_lidar)} car(s) visualised.')
    print('Open3D controls: drag=rotate  Ctrl+drag=pan  scroll=zoom  Q/Esc=quit')

    o3d.visualization.draw_geometries(
        geoms,
        window_name='PointRCNN – Car Detection',
        width=1280, height=720,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(1024)

    # ── Config ────────────────────────────────────────────────────────────
    cfg_from_file(args.cfg_file)
    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    if not use_cuda:
        print('[INFO] Running on CPU (inference will be slower).')

    try:
        from lib.net.point_rcnn import PointRCNN
    except Exception as err:
        err_text = str(err)
        if ('pointnet2_cuda' in err_text) or ('undefined symbol' in err_text):
            print('\n[ERROR] PointNet2 CUDA extension failed to load.')
            print('[ERROR] This repository implementation depends on PointNet2 CUDA ops,')
            print('        so model import cannot continue until extensions are rebuilt.')
            print('\n[Next steps]')
            print('  1) Ensure torch/cuda build and toolchain are aligned.')
            print('  2) Rebuild extensions in this environment:')
            print('     pip uninstall -y pointnet2 iou3d roipool3d')
            print('     cd /home/lenovo/venvs/pointrcnn/PointRCNN')
            print('     find . -type d \\( -name build -o -name "*.egg-info" \\) -exec rm -rf {} +')
            print('     bash build_and_install.sh')
            print('  3) Retry this script.')
            raise SystemExit(1)
        raise

    # ── Calibration ───────────────────────────────────────────────────────
    if args.calib_file:
        assert os.path.exists(args.calib_file), \
            f'Calibration file not found: {args.calib_file}'
        calib = calibration_module.Calibration(args.calib_file)
        print(f'Calibration  : {args.calib_file}')
    else:
        calib = calibration_module.Calibration(_DEFAULT_CALIB)
        print('[WARNING] --calib_file not supplied. '
              'Using default KITTI calibration as fallback.')
        print('          Provide the matching .txt file for accurate results.')

    # ── Load point cloud ──────────────────────────────────────────────────
    assert os.path.exists(args.bin_file), f'File not found: {args.bin_file}'
    pts_lidar_raw = np.fromfile(args.bin_file, dtype=np.float32).reshape(-1, 4)
    print(f'Point cloud  : {pts_lidar_raw.shape[0]} points  ← {args.bin_file}')

    # ── Pre-process ───────────────────────────────────────────────────────
    print('Pre-processing …')
    pts_input_t, _pts_rect = preprocess(
        pts_lidar_raw,
        calib,
        img_shape     = (args.img_height, args.img_width, 3),
        npoints       = args.npoints,
        use_intensity = cfg.RPN.USE_INTENSITY,
    )

    # ── Load checkpoint ───────────────────────────────────────────────────
    assert os.path.isfile(args.ckpt), f'Checkpoint not found: {args.ckpt}'
    ckpt        = torch.load(args.ckpt, map_location=device)
    model_state = ckpt.get('model_state', ckpt)   # support both save formats

    # Try to auto-fix common config/checkpoint mismatch (e.g., LOC_XZ_FINE).
    adapt_cfg_from_checkpoint(model_state)

    # ── Build model ───────────────────────────────────────────────────────
    print('Building model …')
    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    if use_cuda:
        model.cuda()
    model.eval()

    cur_state   = model.state_dict()
    update      = {k: v for k, v in model_state.items() if k in cur_state}
    cur_state.update(update)
    model.load_state_dict(cur_state)
    print(f'Checkpoint   : {args.ckpt}  ({len(update)}/{len(cur_state)} keys matched)')

    # ── Inference ─────────────────────────────────────────────────────────
    print('Running inference …')
    pred_boxes3d_rect, scores = run_inference(
        model, pts_input_t,
        score_thresh = args.score_thresh,
        nms_thresh   = args.nms_thresh,
        use_cuda     = use_cuda,
        iou3d_utils  = iou3d_utils,
    )

    # ── Print detections ──────────────────────────────────────────────────
    print(f'\nDetected {len(pred_boxes3d_rect)} car(s) '
          f'(score > {args.score_thresh}, NMS thresh = {args.nms_thresh}):')
    for i, (box, s) in enumerate(zip(pred_boxes3d_rect, scores)):
        x, y, z, h, w, l, ry = box
        print(f'  Car {i + 1:2d}: '
              f'xyz=({x:6.2f}, {y:6.2f}, {z:6.2f})  '
              f'hwl=({h:.2f}, {w:.2f}, {l:.2f})  '
              f'ry={ry:.2f} rad  '
              f'score={s:.3f}')

    # ── Convert boxes to LiDAR corners for Open3D ─────────────────────────
    if len(pred_boxes3d_rect) > 0:
        corners_lidar = boxes_rect_to_lidar_corners(pred_boxes3d_rect, calib)
    else:
        corners_lidar = np.zeros((0, 8, 3), dtype=np.float32)

    # ── Open3D visualisation ──────────────────────────────────────────────
    print('\nOpening Open3D window …')
    visualize(pts_lidar_raw, corners_lidar, scores)


if __name__ == '__main__':
    main()
