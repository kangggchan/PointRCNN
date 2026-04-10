import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lib.config import cfg, cfg_from_file, cfg_from_list
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN
from lib.utils.bbox_transform import decode_bbox_target
import lib.utils.kitti_utils as kitti_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark PointRCNN inference on Jetson or other edge GPUs.')
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='Inference config file.')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint to load.')
    parser.add_argument('--bin_file', type=str, default=None, help='Single .bin frame in camera frame.')
    parser.add_argument('--bin_glob', type=str, default=None, help='Glob for multiple .bin frames.')
    parser.add_argument('--max_frames', type=int, default=32, help='Maximum number of frames to benchmark.')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations before timing.')
    parser.add_argument('--repeats', type=int, default=50, help='Timed iterations per frame.')
    parser.add_argument('--npoints', type=int, default=None, help='Override cfg.RPN.NUM_POINTS at runtime.')
    parser.add_argument('--score_thresh', type=float, default=None, help='Override cfg.RCNN.SCORE_THRESH.')
    parser.add_argument('--nms_thresh', type=float, default=None, help='Override cfg.RCNN.NMS_THRESH.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    parser.add_argument('--save_json', type=str, default=None, help='Optional output JSON path for benchmark results.')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='Extra config overrides, for example TEST.RPN_POST_NMS_TOP_N 128')
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


def _cuda_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def _cpu_ms(fn):
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def run_inference_timed(model, pts_input_t, score_thresh, nms_thresh, use_cuda, iou3d_utils):
    timer = _cuda_ms if use_cuda else _cpu_ms
    timings = {}

    if use_cuda:
        pts_input_t, timings['h2d_ms'] = timer(lambda: pts_input_t.cuda(non_blocking=True))
    else:
        timings['h2d_ms'] = 0.0

    with torch.no_grad():
        ret_dict, timings['model_ms'] = timer(lambda: model({'pts_input': pts_input_t}))

        def decode_and_nms():
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
                return 0

            boxes_kept = pred_boxes3d[0, keep_mask]
            scores_kept = cls_scores[keep_mask]
            raw_scores_kept = raw_scores[keep_mask]
            class_ids_kept = class_ids[keep_mask]

            kept_count = 0
            for class_id in torch.unique(class_ids_kept, sorted=True):
                class_mask = class_ids_kept == class_id
                class_boxes = boxes_kept[class_mask]
                class_raw_scores = raw_scores_kept[class_mask]

                if iou3d_utils is None:
                    keep_idx = torch.argsort(class_raw_scores, descending=True)
                else:
                    boxes_bev = kitti_utils.boxes3d_to_bev_torch(class_boxes)
                    keep_idx = iou3d_utils.nms_gpu(boxes_bev, class_raw_scores, nms_thresh).view(-1)
                kept_count += int(keep_idx.numel())

            _ = scores_kept
            return kept_count

        det_count, timings['decode_nms_ms'] = timer(decode_and_nms)

    timings['total_ms'] = timings['h2d_ms'] + timings['model_ms'] + timings['decode_nms_ms']
    return det_count, timings


def summarize_ms(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean_ms': float(arr.mean()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p90_ms': float(np.percentile(arr, 90)),
        'p95_ms': float(np.percentile(arr, 95)),
        'fps_mean': float(1000.0 / max(arr.mean(), 1e-6)),
    }


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
        cfg.RCNN.ROI_SAMPLE_JIT = True
    if args.npoints is not None:
        cfg.RPN.NUM_POINTS = int(args.npoints)
    if args.score_thresh is not None:
        cfg.RCNN.SCORE_THRESH = float(args.score_thresh)
    if args.nms_thresh is not None:
        cfg.RCNN.NMS_THRESH = float(args.nms_thresh)

    class_names = get_class_names()
    npoints = int(cfg.RPN.NUM_POINTS)

    frame_paths = []
    if args.bin_file:
        frame_paths = [Path(args.bin_file).expanduser().resolve()]
    elif args.bin_glob:
        frame_paths = [Path(p).resolve() for p in sorted(glob.glob(args.bin_glob))[:args.max_frames]]
    else:
        raise ValueError('Specify either --bin_file or --bin_glob')

    if not frame_paths:
        raise FileNotFoundError('No .bin files matched the provided input')

    points_by_frame = []
    for frame_path in frame_paths:
        if not frame_path.exists():
            raise FileNotFoundError(f'Point cloud file not found: {frame_path}')
        points_by_frame.append(np.fromfile(str(frame_path), dtype=np.float32).reshape(-1, 4))

    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    model, matched_state, current_state = build_model(args.ckpt, class_names, device, use_cuda)

    print(f'Checkpoint      : {Path(args.ckpt).resolve()}')
    print(f'Config          : {Path(args.cfg_file).resolve()}')
    print(f'Frames          : {len(frame_paths)}')
    print(f'Using CUDA      : {use_cuda}')
    print(f'RPN.NUM_POINTS  : {cfg.RPN.NUM_POINTS}')
    print(f'TEST pre/post   : {cfg.TEST.RPN_PRE_NMS_TOP_N}/{cfg.TEST.RPN_POST_NMS_TOP_N}')
    print(f'RCNN thresholds : score={cfg.RCNN.SCORE_THRESH} nms={cfg.RCNN.NMS_THRESH}')
    print(f'Matched keys    : {len(matched_state)}/{len(current_state)}')

    warmup_frame = points_by_frame[0]
    for _ in range(args.warmup):
        pts_input_t = preprocess_points(warmup_frame, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
        run_inference_timed(model, pts_input_t, cfg.RCNN.SCORE_THRESH, cfg.RCNN.NMS_THRESH, use_cuda, iou3d_utils)

    preprocess_ms_list = []
    h2d_ms_list = []
    model_ms_list = []
    decode_nms_ms_list = []
    total_ms_list = []
    det_count_list = []

    for frame_idx, points in enumerate(points_by_frame):
        for repeat_idx in range(args.repeats):
            _, preprocess_ms = _cpu_ms(lambda: preprocess_points(points, npoints=npoints,
                                                                 use_intensity=cfg.RPN.USE_INTENSITY))
            pts_input_t = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
            det_count, timings = run_inference_timed(
                model,
                pts_input_t,
                cfg.RCNN.SCORE_THRESH,
                cfg.RCNN.NMS_THRESH,
                use_cuda,
                iou3d_utils,
            )

            preprocess_ms_list.append(preprocess_ms)
            h2d_ms_list.append(timings['h2d_ms'])
            model_ms_list.append(timings['model_ms'])
            decode_nms_ms_list.append(timings['decode_nms_ms'])
            total_ms_list.append(preprocess_ms + timings['total_ms'])
            det_count_list.append(det_count)

        print(f'Frame {frame_idx + 1:02d}/{len(points_by_frame)}: avg total {np.mean(total_ms_list[-args.repeats:]):.2f} ms')

    summary = {
        'preprocess': summarize_ms(preprocess_ms_list),
        'h2d': summarize_ms(h2d_ms_list) if use_cuda else {'mean_ms': 0.0, 'p50_ms': 0.0, 'p90_ms': 0.0, 'p95_ms': 0.0, 'fps_mean': 0.0},
        'model': summarize_ms(model_ms_list),
        'decode_nms': summarize_ms(decode_nms_ms_list),
        'end_to_end': summarize_ms(total_ms_list),
        'avg_detection_count': float(np.mean(det_count_list)),
        'frames': [str(path) for path in frame_paths],
        'config': {
            'cfg_file': str(Path(args.cfg_file).resolve()),
            'ckpt': str(Path(args.ckpt).resolve()),
            'rpn_num_points': int(cfg.RPN.NUM_POINTS),
            'test_rpn_pre_nms_top_n': int(cfg.TEST.RPN_PRE_NMS_TOP_N),
            'test_rpn_post_nms_top_n': int(cfg.TEST.RPN_POST_NMS_TOP_N),
            'rcnn_score_thresh': float(cfg.RCNN.SCORE_THRESH),
            'rcnn_nms_thresh': float(cfg.RCNN.NMS_THRESH),
            'use_cuda': bool(use_cuda),
        },
    }

    print('\nLatency summary')
    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        output_path = Path(args.save_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Saved benchmark JSON to {output_path}')


if __name__ == '__main__':
    main()