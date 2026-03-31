import os
import sys
import argparse
import importlib.util

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from lib.net.point_rcnn import PointRCNN
from lib.config import cfg, cfg_from_file
import lib.utils.calibration as calibration_module


def load_infer_helpers():
    infer_path = os.path.join(THIS_DIR, 'infer_single_bin.py')
    if not os.path.isfile(infer_path):
        raise FileNotFoundError(f'infer_single_bin.py not found: {infer_path}')

    spec = importlib.util.spec_from_file_location('infer_single_bin_runtime', infer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description='Batch auto-label scala2 frames with PointRCNN')
    parser.add_argument('--data_root', type=str, default='data/scala2', help='Root folder containing bin/ and labels/')
    parser.add_argument('--bin_subdir', type=str, default='bin', help='Subfolder for .bin files')
    parser.add_argument('--label_subdir', type=str, default='labels', help='Subfolder for .txt label files')

    parser.add_argument('--start_idx', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=211, help='End frame index (inclusive)')

    parser.add_argument('--ckpt', type=str, default='PointRCNN.pth', help='Checkpoint path')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/default.yaml', help='Config file path')

    parser.add_argument('--calib_file', type=str, default=None,
                        help='Single calib file to use for all frames (optional)')
    parser.add_argument('--img_height', type=int, default=375, help='Image height for projection filter')
    parser.add_argument('--img_width', type=int, default=1242, help='Image width for projection filter')

    parser.add_argument('--score_thresh', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.1, help='NMS threshold')
    parser.add_argument('--npoints', type=int, default=16384, help='Points sampled per frame')
    parser.add_argument('--label_name', type=str, default='Car', help='Label token to write at line start')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU mode')
    return parser.parse_args()


def frame_name(idx: int) -> str:
    return f'frame_{idx:06d}'


def rect_boxes_to_lidar_labels(boxes_rect: np.ndarray, calib, boxes_rect_to_lidar_corners_fn) -> np.ndarray:
    if boxes_rect.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    corners_lidar = boxes_rect_to_lidar_corners_fn(boxes_rect, calib)
    labels = np.zeros((boxes_rect.shape[0], 7), dtype=np.float32)

    for i, corners in enumerate(corners_lidar):
        center = corners.mean(axis=0)

        length_vec = corners[0] - corners[3]
        width_vec = corners[0] - corners[1]
        height_vec = corners[4] - corners[0]

        length = float(np.linalg.norm(length_vec))
        width = float(np.linalg.norm(width_vec))
        height = float(np.linalg.norm(height_vec))
        yaw = float(np.arctan2(length_vec[1], length_vec[0]))

        labels[i] = np.array([
            center[0], center[1], center[2],
            length, width, height, yaw,
        ], dtype=np.float32)

    return labels


def update_label_file_in_place(label_file: str, label_name: str, boxes: np.ndarray):
    prefix = f'{label_name} '

    with open(label_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    existing_lines = [line for line in original_lines if not line.startswith(prefix)]
    removed = len(original_lines) - len(existing_lines)

    for box in boxes:
        x, y, z, l, w, h, yaw = box
        existing_lines.append(
            f'{label_name} {x:.4f} {y:.4f} {z:.4f} {l:.4f} {w:.4f} {h:.4f} {yaw:.4f}\n'
        )

    with open(label_file, 'w', encoding='utf-8') as f:
        f.writelines(existing_lines)

    return boxes.shape[0], removed


def main():
    args = parse_args()
    np.random.seed(1024)

    infer_helpers = load_infer_helpers()

    cfg_from_file(args.cfg_file)

    use_cuda = infer_helpers.should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = infer_helpers.ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    if not use_cuda:
        print('[INFO] Running labeling on CPU.')

    if args.calib_file:
        assert os.path.exists(args.calib_file), f'Calibration file not found: {args.calib_file}'
        calib = calibration_module.Calibration(args.calib_file)
        print(f'[INFO] Using calibration file: {args.calib_file}')
    else:
        calib = calibration_module.Calibration(infer_helpers._DEFAULT_CALIB)
        print('[WARNING] No calib file provided; using default KITTI calibration fallback.')

    assert os.path.isfile(args.ckpt), f'Checkpoint not found: {args.ckpt}'
    ckpt = torch.load(args.ckpt, map_location=device)
    model_state = ckpt.get('model_state', ckpt)
    infer_helpers.adapt_cfg_from_checkpoint(model_state)

    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    if use_cuda:
        model.cuda()
    model.eval()

    cur_state = model.state_dict()
    update = {k: v for k, v in model_state.items() if k in cur_state}
    cur_state.update(update)
    model.load_state_dict(cur_state)
    print(f'[INFO] Loaded checkpoint: {args.ckpt} ({len(update)}/{len(cur_state)} keys matched)')

    bin_dir = os.path.join(args.data_root, args.bin_subdir)
    label_dir = os.path.join(args.data_root, args.label_subdir)
    assert os.path.isdir(bin_dir), f'Bin directory not found: {bin_dir}'
    assert os.path.isdir(label_dir), f'Label directory not found: {label_dir}'

    total_frames = 0
    processed_frames = 0
    skipped_missing = 0
    total_written = 0
    total_removed = 0

    for idx in range(args.start_idx, args.end_idx + 1):
        total_frames += 1
        name = frame_name(idx)
        bin_file = os.path.join(bin_dir, f'{name}.bin')
        label_file = os.path.join(label_dir, f'{name}.txt')

        if not os.path.isfile(bin_file):
            print(f'[SKIP] Missing bin file: {bin_file}')
            skipped_missing += 1
            continue
        if not os.path.isfile(label_file):
            print(f'[SKIP] Missing label file (not creating new file): {label_file}')
            skipped_missing += 1
            continue

        pts_lidar_raw = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        pts_input_t, _ = infer_helpers.preprocess(
            pts_lidar_raw,
            calib,
            img_shape=(args.img_height, args.img_width, 3),
            npoints=args.npoints,
            use_intensity=cfg.RPN.USE_INTENSITY,
        )

        pred_boxes3d_rect, scores = infer_helpers.run_inference(
            model=model,
            pts_input_t=pts_input_t,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            use_cuda=use_cuda,
            iou3d_utils=iou3d_utils,
        )

        pred_boxes_lidar = rect_boxes_to_lidar_labels(
            pred_boxes3d_rect,
            calib,
            infer_helpers.boxes_rect_to_lidar_corners,
        )

        written, removed = update_label_file_in_place(label_file, args.label_name, pred_boxes_lidar)
        processed_frames += 1
        total_written += written
        total_removed += removed
        print(f'[OK] {name}: removed {removed} old {args.label_name} labels, wrote {written} labels')

    print('\n===== Summary =====')
    print(f'Requested frame range : {args.start_idx:06d} -> {args.end_idx:06d}')
    print(f'Total frames scanned  : {total_frames}')
    print(f'Frames processed      : {processed_frames}')
    print(f'Frames skipped        : {skipped_missing}')
    print(f'Old labels removed    : {total_removed}')
    print(f'Labels written total  : {total_written}')


if __name__ == '__main__':
    main()
