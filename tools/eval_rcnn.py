import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from infer_frame_open3d import (
    build_model,
    ensure_iou3d_utils,
    get_class_names,
    preprocess_points,
    resolve_existing_path,
    run_inference,
    should_use_cuda,
)
from lib.config import cfg, cfg_from_file, cfg_from_list
import lib.utils.kitti_utils as kitti_utils


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate an RCNN checkpoint on a custom dataset with paired bin/ and labels/ files.'
    )
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='Config file path.')
    parser.add_argument('--ckpt', type=str, default='../output/rcnn/default/ckpt/checkpoint_epoch_40.pth',
                        help='Checkpoint to evaluate.')
    parser.add_argument('--data_root', type=str, default='../data/dataset',
                        help='Dataset root containing bin/ and labels/.')
    parser.add_argument('--score_thresh', type=float, default=0.7, help='Inference score threshold.')
    parser.add_argument('--nms_thresh', type=float, default=None, help='Inference NMS threshold override.')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold used for TP/FP matching.')
    parser.add_argument('--npoints', type=int, default=None, help='Override cfg.RPN.NUM_POINTS.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for the summary CSV.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='Extra config overrides, same format as other scripts.')
    return parser.parse_args()


def compute_iou3d(pred_boxes, gt_boxes, use_cuda, iou3d_utils):
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    if use_cuda and iou3d_utils is not None:
        pred_tensor = torch.from_numpy(pred_boxes).cuda(non_blocking=True).float()
        gt_tensor = torch.from_numpy(gt_boxes).cuda(non_blocking=True).float()
        return iou3d_utils.boxes_iou3d_gpu(pred_tensor, gt_tensor).cpu().numpy()

    pred_corners = kitti_utils.boxes3d_to_corners3d(pred_boxes)
    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes)
    return kitti_utils.get_iou3d(pred_corners, gt_corners)


def summarize_class(records, gt_count, pred_count):
    if records:
        records.sort(key=lambda item: item[0], reverse=True)
        tp = np.cumsum([item[1] for item in records], dtype=np.float32)
        fp = np.cumsum([1 - item[1] for item in records], dtype=np.float32)
        precision_curve = tp / np.maximum(tp + fp, 1e-6)
        recall_curve = tp / max(gt_count, 1)

        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], precision_curve, [0.0]))
        for idx in range(mpre.size - 2, -1, -1):
            mpre[idx] = max(mpre[idx], mpre[idx + 1])
        change = np.where(mrec[1:] != mrec[:-1])[0]

        tp_total = int(tp[-1])
        fp_total = int(fp[-1])
        precision = float(tp_total / max(pred_count, 1))
        recall = float(tp_total / max(gt_count, 1))
        ap = float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))
    else:
        tp_total = 0
        fp_total = pred_count
        precision = 0.0
        recall = 0.0
        ap = 0.0

    return {
        'tp': tp_total,
        'fp': fp_total,
        'fn': max(gt_count - tp_total, 0),
        'precision': precision,
        'recall': recall,
        'ap_3d': ap,
    }


def main():
    args = parse_args()
    np.random.seed(1024)

    cfg_path = resolve_existing_path(args.cfg_file)
    ckpt_path = resolve_existing_path(args.ckpt)
    data_root = resolve_existing_path(args.data_root)

    cfg_from_file(str(cfg_path))
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if cfg.RCNN.ENABLED and not cfg.RCNN.ROI_SAMPLE_JIT:
        cfg.RCNN.ROI_SAMPLE_JIT = True

    class_names_with_bg = get_class_names()
    class_names = class_names_with_bg[1:]
    npoints = int(args.npoints) if args.npoints is not None else int(cfg.RPN.NUM_POINTS)
    nms_thresh = float(args.nms_thresh) if args.nms_thresh is not None else float(cfg.RCNN.NMS_THRESH)

    use_cuda = should_use_cuda(args.no_cuda)
    iou3d_utils, use_cuda = ensure_iou3d_utils(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')

    model, matched_state, current_state = build_model(str(ckpt_path), class_names_with_bg, device, use_cuda)

    bin_dir = data_root / 'bin'
    label_dir = data_root / 'labels'
    if not bin_dir.exists():
        raise FileNotFoundError(f'Bin folder not found: {bin_dir}')
    if not label_dir.exists():
        raise FileNotFoundError(f'Label folder not found: {label_dir}')

    sample_ids = []
    for bin_path in sorted(bin_dir.glob('*.bin')):
        sample_id = bin_path.stem
        if (label_dir / f'{sample_id}.txt').exists():
            sample_ids.append(sample_id)

    if not sample_ids:
        raise FileNotFoundError(f'No matching .bin/.txt pairs found under {data_root}')

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (SCRIPT_DIR.parent / 'output' / 'rcnn_eval' / Path(args.cfg_file).stem / data_root.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        class_name: {'gt_count': 0, 'pred_count': 0, 'records': []}
        for class_name in class_names
    }

    print(f'Checkpoint      : {ckpt_path}')
    print(f'Config          : {cfg_path}')
    print(f'Bin folder      : {bin_dir}')
    print(f'Label folder    : {label_dir}')
    print(f'Samples         : {len(sample_ids)}')
    print(f'Using CUDA      : {use_cuda}')
    print(f'RPN.NUM_POINTS  : {npoints}')
    print(f'RCNN thresholds : score={args.score_thresh} nms={nms_thresh}')
    print(f'IoU threshold   : {args.iou_thresh}')
    print(f'Matched keys    : {len(matched_state)}/{len(current_state)}')

    start_time = time.perf_counter()

    for index, sample_id in enumerate(sample_ids, start=1):
        bin_path = bin_dir / f'{sample_id}.bin'
        label_path = label_dir / f'{sample_id}.txt'

        points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        pts_input = preprocess_points(points, npoints=npoints, use_intensity=cfg.RPN.USE_INTENSITY)
        pred_boxes, pred_scores, pred_class_ids = run_inference(
            model,
            pts_input,
            score_thresh=args.score_thresh,
            nms_thresh=nms_thresh,
            use_cuda=use_cuda,
            iou3d_utils=iou3d_utils,
        )

        gt_objects = [obj for obj in kitti_utils.get_objects_from_label(str(label_path)) if obj.cls_id > 0]
        gt_boxes = np.zeros((len(gt_objects), 7), dtype=np.float32)
        gt_class_ids = np.zeros((len(gt_objects),), dtype=np.int64)
        for gt_idx, obj in enumerate(gt_objects):
            gt_boxes[gt_idx, 0:3] = obj.pos
            gt_boxes[gt_idx, 3] = obj.h
            gt_boxes[gt_idx, 4] = obj.w
            gt_boxes[gt_idx, 5] = obj.l
            gt_boxes[gt_idx, 6] = obj.ry
            gt_class_ids[gt_idx] = obj.cls_id

        for class_id, class_name in enumerate(class_names, start=1):
            gt_mask = gt_class_ids == class_id
            pred_mask = pred_class_ids == class_id
            gt_boxes_cls = gt_boxes[gt_mask]
            pred_boxes_cls = pred_boxes[pred_mask]
            pred_scores_cls = pred_scores[pred_mask]

            stats[class_name]['gt_count'] += int(gt_boxes_cls.shape[0])
            stats[class_name]['pred_count'] += int(pred_boxes_cls.shape[0])

            if pred_boxes_cls.shape[0] == 0:
                continue

            order = np.argsort(-pred_scores_cls)
            pred_boxes_cls = pred_boxes_cls[order]
            pred_scores_cls = pred_scores_cls[order]

            if gt_boxes_cls.shape[0] == 0:
                stats[class_name]['records'].extend((float(score), 0) for score in pred_scores_cls.tolist())
                continue

            iou_matrix = compute_iou3d(pred_boxes_cls, gt_boxes_cls, use_cuda, iou3d_utils)
            matched_gt = np.zeros(gt_boxes_cls.shape[0], dtype=bool)

            for pred_idx, score in enumerate(pred_scores_cls):
                best_gt = int(np.argmax(iou_matrix[pred_idx]))
                is_tp = iou_matrix[pred_idx, best_gt] >= args.iou_thresh and not matched_gt[best_gt]
                if is_tp:
                    matched_gt[best_gt] = True
                stats[class_name]['records'].append((float(score), 1 if is_tp else 0))

        if index % 20 == 0 or index == len(sample_ids):
            print(f'Processed {index}/{len(sample_ids)} samples')

    elapsed = time.perf_counter() - start_time
    csv_path = output_dir / f'metrics_iou_{args.iou_thresh:.2f}.csv'

    rows = []
    for class_name in class_names:
        class_stats = stats[class_name]
        summary = summarize_class(
            class_stats['records'],
            class_stats['gt_count'],
            class_stats['pred_count'],
        )
        rows.append({
            'class_name': class_name,
            'gt_count': class_stats['gt_count'],
            'pred_count': class_stats['pred_count'],
            'tp': summary['tp'],
            'fp': summary['fp'],
            'fn': summary['fn'],
            'precision': summary['precision'],
            'recall': summary['recall'],
            'ap_3d': summary['ap_3d'],
        })

    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            'class_name', 'gt_count', 'pred_count', 'tp', 'fp', 'fn', 'precision', 'recall', 'ap_3d'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print('')
    print('Per-class metrics')
    print('class         GT   Pred   TP   FP   FN   Precision   Recall   AP_3D@%.2f' % args.iou_thresh)
    for row in rows:
        print('%-12s %4d %6d %4d %4d %4d %10.4f %8.4f %10.4f' % (
            row['class_name'],
            row['gt_count'],
            row['pred_count'],
            row['tp'],
            row['fp'],
            row['fn'],
            row['precision'],
            row['recall'],
            row['ap_3d'],
        ))

    print('')
    print(f'Elapsed seconds : {elapsed:.2f}')
    print(f'Output CSV      : {csv_path}')


if __name__ == '__main__':
    main()
