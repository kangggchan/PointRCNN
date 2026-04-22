import argparse
from pathlib import Path

import numpy as np
import torch

from lib.config import cfg, cfg_from_file
from tools.infer_frame_open3d import (
    build_model,
    convert_points_to_camera_frame,
    ensure_iou3d_utils,
    get_class_names,
    preprocess_points,
    resolve_existing_path,
    run_inference,
    should_use_cuda,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Auto-label a range of Scala2 frames with PointRCNN predictions.')
    parser.add_argument('--data_root', type=str, default='data/scala2', help='Root folder containing bin/ and labels/.')
    parser.add_argument('--bin_subdir', type=str, default='bin', help='Subfolder for .bin files.')
    parser.add_argument('--label_subdir', type=str, default='labels', help='Subfolder for .txt label files.')
    parser.add_argument('--start_idx', type=int, default=0, help='Start frame index, inclusive.')
    parser.add_argument('--end_idx', type=int, default=211, help='End frame index, inclusive.')
    parser.add_argument('--ckpt', type=str, default='PointRCNN.pth', help='Checkpoint path.')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/default.yaml', help='Config file path.')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='Detection score threshold.')
    parser.add_argument('--nms_thresh', type=float, default=0.1, help='NMS threshold.')
    parser.add_argument('--npoints', type=int, default=None, help='Override cfg.RPN.NUM_POINTS.')
    parser.add_argument('--label_name', type=str, default='Car', help='Label token written for each predicted box.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU mode.')
    return parser.parse_args()


def frame_name(idx: int) -> str:
    return f'frame_{idx:06d}'


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def rect_boxes_to_lidar_labels(boxes_rect: np.ndarray) -> np.ndarray:
    if boxes_rect.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    labels = np.zeros((boxes_rect.shape[0], 7), dtype=np.float32)
    heights = boxes_rect[:, 3]

    labels[:, 0] = boxes_rect[:, 2]
    labels[:, 1] = -boxes_rect[:, 0]
    labels[:, 2] = -boxes_rect[:, 1] + (heights / 2.0)
    labels[:, 3] = boxes_rect[:, 5]
    labels[:, 4] = boxes_rect[:, 4]
    labels[:, 5] = boxes_rect[:, 3]
    labels[:, 6] = wrap_angle(-boxes_rect[:, 6] - (np.pi / 2.0))
    return labels.astype(np.float32)


def update_label_file_in_place(label_file: Path, label_name: str, boxes: np.ndarray):
    prefix = f'{label_name} '
    original_lines = label_file.read_text(encoding='utf-8').splitlines(keepends=True)

    existing_lines = [line for line in original_lines if not line.startswith(prefix)]
    removed = len(original_lines) - len(existing_lines)

    for box in boxes:
        x, y, z, l, w, h, yaw = box
        existing_lines.append(
            f'{label_name} {x:.4f} {y:.4f} {z:.4f} {l:.4f} {w:.4f} {h:.4f} {yaw:.4f}\n'
        )

    label_file.write_text(''.join(existing_lines), encoding='utf-8')
    return boxes.shape[0], removed


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_path = resolve_existing_path(args.cfg_file)
    ckpt_path = resolve_existing_path(args.ckpt)
    data_root = resolve_existing_path(args.data_root)

    cfg_from_file(str(cfg_path))
    if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
        cfg.RCNN.ROI_SAMPLE_JIT = True

    npoints = int(args.npoints) if args.npoints is not None else int(cfg.RPN.NUM_POINTS)
    class_names = get_class_names()
    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    model, matched_state, current_state = build_model(str(ckpt_path), class_names, device, use_cuda)

    bin_dir = data_root / args.bin_subdir
    label_dir = data_root / args.label_subdir
    if not bin_dir.exists():
        raise FileNotFoundError(f'Bin directory not found: {bin_dir}')
    if not label_dir.exists():
        raise FileNotFoundError(f'Label directory not found: {label_dir}')

    print(f'Checkpoint      : {ckpt_path}')
    print(f'Config          : {cfg_path}')
    print(f'Data root       : {data_root}')
    print(f'Frame range     : {args.start_idx:06d} -> {args.end_idx:06d}')
    print(f'Using CUDA      : {use_cuda}')
    print(f'RPN.NUM_POINTS  : {npoints}')
    print(f'Matched keys    : {len(matched_state)}/{len(current_state)}')

    total_frames = 0
    processed_frames = 0
    skipped_missing = 0
    total_written = 0
    total_removed = 0

    for idx in range(args.start_idx, args.end_idx + 1):
        total_frames += 1
        name = frame_name(idx)
        bin_path = bin_dir / f'{name}.bin'
        label_path = label_dir / f'{name}.txt'

        if not bin_path.exists():
            print(f'[SKIP] Missing bin file: {bin_path}')
            skipped_missing += 1
            continue
        if not label_path.exists():
            print(f'[SKIP] Missing label file (not creating new file): {label_path}')
            skipped_missing += 1
            continue

        points_lidar = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        points_camera = convert_points_to_camera_frame(points_lidar, 'lidar')
        pts_input_t = preprocess_points(points_camera, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)

        pred_boxes_rect, _, _ = run_inference(
            model=model,
            pts_input_t=pts_input_t,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            use_cuda=use_cuda,
            iou3d_utils=iou3d_utils,
        )

        pred_boxes_lidar = rect_boxes_to_lidar_labels(pred_boxes_rect)
        written, removed = update_label_file_in_place(label_path, args.label_name, pred_boxes_lidar)
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
