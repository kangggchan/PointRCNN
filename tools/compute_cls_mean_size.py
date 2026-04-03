#!/usr/bin/env python
"""
Compute per-class mean box size (h, w, l) from KITTI-format labels.

Supports train_aug split by resolving labels exactly like dataset loading:
- sample_id < 10000: use KITTI/object/training/label_2
- sample_id >= 10000: use KITTI/aug_scene/training/aug_label

Usage:
  python tools/compute_cls_mean_size.py \
      --data_root data/dataset \
      --split train_aug \
      --classes Car,Human,ForkLift,CargoBike
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_classes(class_string):
    return [c.strip() for c in class_string.split(',') if c.strip()]


def parse_label_file(label_file):
    rows = []
    if not label_file.exists():
        return rows

    with label_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                rows.append(
                    {
                        'class_name': parts[0],
                        'h': float(parts[8]),
                        'w': float(parts[9]),
                        'l': float(parts[10]),
                    }
                )
            except (ValueError, IndexError):
                continue
    return rows


def resolve_label_path(kitti_root, split, sample_id):
    sid = int(sample_id)
    if split == 'train_aug':
        if sid < 10000:
            return kitti_root / 'object' / 'training' / 'label_2' / f'{sid:06d}.txt'
        return kitti_root / 'aug_scene' / 'training' / 'aug_label' / f'{sid:06d}.txt'

    # Non-train_aug default behavior
    return kitti_root / 'object' / 'training' / 'label_2' / f'{sid:06d}.txt'


def main():
    parser = argparse.ArgumentParser(description='Compute class mean size for PointRCNN config')
    parser.add_argument('--data_root', type=str, default='data/dataset', help='Root containing KITTI/')
    parser.add_argument('--split', type=str, default='train_aug', help='Split name in KITTI/ImageSets')
    parser.add_argument('--classes', type=str, default='Car,Human,ForkLift,CargoBike')
    parser.add_argument('--out_csv', type=str, default='output/cls_mean_size_stats.csv')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    kitti_root = data_root / 'KITTI'
    split_file = kitti_root / 'ImageSets' / f'{args.split}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'Split file not found: {split_file}')

    classes = parse_classes(args.classes)
    class_set = set(classes)

    sample_ids = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]

    dims_by_class = defaultdict(list)
    missing_labels = 0

    for sid in sample_ids:
        label_file = resolve_label_path(kitti_root, args.split, sid)
        if not label_file.exists():
            missing_labels += 1
            continue

        for obj in parse_label_file(label_file):
            if obj['class_name'] not in class_set:
                continue
            dims_by_class[obj['class_name']].append([obj['h'], obj['w'], obj['l']])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_name', 'num_boxes', 'mean_h', 'mean_w', 'mean_l', 'median_h', 'median_w', 'median_l'])

        print('=' * 90)
        print('Per-class box size stats (h, w, l)')
        print('=' * 90)
        print(f'Split file       : {split_file}')
        print(f'Samples in split : {len(sample_ids)}')
        print(f'Missing labels   : {missing_labels}')
        print('')

        mean_rows = []
        for cls in classes:
            arr = np.array(dims_by_class.get(cls, []), dtype=np.float64)
            if arr.size == 0:
                writer.writerow([cls, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                print(f'{cls:12s} count=0 (no objects found)')
                mean_rows.append([np.nan, np.nan, np.nan])
                continue

            mean_vals = arr.mean(axis=0)
            med_vals = np.median(arr, axis=0)
            writer.writerow([
                cls,
                int(arr.shape[0]),
                float(mean_vals[0]),
                float(mean_vals[1]),
                float(mean_vals[2]),
                float(med_vals[0]),
                float(med_vals[1]),
                float(med_vals[2]),
            ])
            print(
                f"{cls:12s} count={arr.shape[0]:6d} "
                f"mean=[{mean_vals[0]:.6f}, {mean_vals[1]:.6f}, {mean_vals[2]:.6f}] "
                f"median=[{med_vals[0]:.6f}, {med_vals[1]:.6f}, {med_vals[2]:.6f}]"
            )
            mean_rows.append(mean_vals.tolist())

    print('')
    print(f'CSV saved to: {out_csv}')

    # Ready-to-paste numpy block in class order
    print('')
    print('Paste-ready CLS_MEAN_SIZE:')
    print('__C.CLS_MEAN_SIZE = np.array([')
    for cls, vals in zip(classes, mean_rows):
        if any(np.isnan(vals)):
            print(f'    [np.nan, np.nan, np.nan],  # {cls} (missing)')
        else:
            print(f'    [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}],  # {cls}')
    print('], dtype=np.float32)')


if __name__ == '__main__':
    main()
