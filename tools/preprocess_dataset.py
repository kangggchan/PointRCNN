import os
import math
import shutil
import argparse
import numpy as np
from pathlib import Path


TARGET_CLASSES = ('Car', 'Human', 'ForkLift', 'CargoBike')
IGNORED_LABELS = {'box', 'boxes', 'Box'}


def _normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_dummy_calib(calib_path):
    fx, fy = 700.0, 700.0
    cx, cy = 640.0, 360.0
    p_mat = f"{fx:.6f} 0.000000 {cx:.6f} 0.000000 0.000000 {fy:.6f} {cy:.6f} 0.000000 0.000000 0.000000 1.000000 0.000000"
    lines = [
        f"P0: {p_mat}\n",
        f"P1: {p_mat}\n",
        f"P2: {p_mat}\n",
        f"P3: {p_mat}\n",
        "R0_rect: 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000\n",
        # Identity transform: bins are already in camera frame (converted by convert_bin_to_kitti.py).
        # x_cam=-y_l, y_cam=-z_l+h/2, z_cam=x_l was applied before preprocess, so lidar_to_rect() must be a no-op.
        "Tr_velo_to_cam: 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000\n",
        "Tr_imu_to_velo: 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 1.000000 0.000000\n",
    ]
    with open(calib_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def _lidar_label_to_kitti_line(parts):
    cls_name = parts[0]
    x_l, y_l, z_l, l, w, h, yaw = [float(v) for v in parts[1:8]]

    # Approximate LiDAR(center) -> KITTI camera(bottom-center)
    x_cam = -y_l
    y_cam = -z_l + (h / 2.0)
    z_cam = x_l
    ry = _normalize_angle(-yaw - math.pi / 2.0)

    trunc = 0.0
    occl = 0
    alpha = -10.0
    x1, y1, x2, y2 = 0.0, 0.0, 50.0, 50.0

    return (
        f"{cls_name} {trunc:.2f} {occl:d} {alpha:.2f} "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.3f} {w:.3f} {l:.3f} {x_cam:.3f} {y_cam:.3f} {z_cam:.3f} {ry:.6f}"
    )


def _convert_label_file(src_path, dst_path, allowed_classes):
    """
    Convert label file format (custom to KITTI).
    
    Does NOT filter by point count - use filter_aug_labels_by_points.py for that.
    
    Args:
        src_path: Source label file path
        dst_path: Destination label file path
        allowed_classes: Tuple of allowed class names
    """
    kept_lines = []

    with open(src_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            cls_name = parts[0]
            cls_lower = cls_name.lower()

            if cls_lower in IGNORED_LABELS:
                continue
            if cls_name not in allowed_classes:
                continue

            kitti_line = None

            # Already KITTI-like label format (15+ fields)
            if len(parts) >= 15:
                kitti_line = ' '.join(parts[:16])

            # Custom lightweight label format: Class x y z l w h yaw
            elif len(parts) == 8:
                kitti_line = _lidar_label_to_kitti_line(parts)

            if kitti_line:
                kept_lines.append(kitti_line)

    with open(dst_path, 'w', encoding='utf-8') as f:
        if kept_lines:
            f.write('\n'.join(kept_lines) + '\n')


def _write_split(split_path, sample_ids):
    with open(split_path, 'w', encoding='utf-8') as f:
        if sample_ids:
            f.write('\n'.join(sample_ids) + '\n')


def preprocess_dataset(dataset_root, classes=None, logger=None):
    if classes is None:
        classes = TARGET_CLASSES

    dataset_root = Path(dataset_root)
    raw_bin_dir = dataset_root / 'bin'
    raw_label_dir = dataset_root / 'labels'

    if not raw_bin_dir.exists() or not raw_label_dir.exists():
        raise FileNotFoundError(f"Expected raw dataset folders at {raw_bin_dir} and {raw_label_dir}")

    kitti_root = dataset_root / 'KITTI'
    training_root = kitti_root / 'object' / 'training'
    testing_root = kitti_root / 'object' / 'testing'
    image_sets_root = kitti_root / 'ImageSets'

    train_velodyne = training_root / 'velodyne'
    train_label2 = training_root / 'label_2'
    train_calib = training_root / 'calib'
    train_image2 = training_root / 'image_2'

    test_velodyne = testing_root / 'velodyne'
    test_calib = testing_root / 'calib'
    test_image2 = testing_root / 'image_2'

    for d in [train_velodyne, train_label2, train_calib, train_image2, test_velodyne, test_calib, test_image2, image_sets_root]:
        _ensure_dir(d)

    bin_files = {p.stem: p for p in sorted(raw_bin_dir.glob('*.bin'))}
    label_files = {p.stem: p for p in sorted(raw_label_dir.glob('*.txt'))}
    paired_ids = sorted(set(bin_files.keys()) & set(label_files.keys()))

    if logger:
        logger.info('Preprocessing dataset at %s', str(dataset_root))
        logger.info('Found %d paired samples', len(paired_ids))

    for idx, sample_id in enumerate(paired_ids):
        dst_bin = train_velodyne / f'{sample_id}.bin'
        dst_label = train_label2 / f'{sample_id}.txt'
        dst_calib = train_calib / f'{sample_id}.txt'

        shutil.copy2(bin_files[sample_id], dst_bin)
        _convert_label_file(label_files[sample_id], dst_label, classes)
        _write_dummy_calib(dst_calib)

        if idx % 500 == 0 and logger:
            logger.info('Preprocessed %d/%d samples', idx + 1, len(paired_ids))

    # Keep testing folder usable for split=test if needed
    for sample_id in paired_ids:
        shutil.copy2(train_velodyne / f'{sample_id}.bin', test_velodyne / f'{sample_id}.bin')
        shutil.copy2(train_calib / f'{sample_id}.txt', test_calib / f'{sample_id}.txt')

    num_total = len(paired_ids)
    if num_total == 0:
        raise RuntimeError('No paired .bin/.txt samples found in dataset root')

    num_train = max(int(num_total * 0.8), 1)
    if num_total > 1:
        num_train = min(num_train, num_total - 1)

    train_ids = paired_ids[:num_train]
    val_ids = paired_ids[num_train:] if num_total > 1 else paired_ids
    smallval_ids = val_ids if len(val_ids) > 0 else train_ids

    _write_split(image_sets_root / 'train.txt', train_ids)
    _write_split(image_sets_root / 'val.txt', val_ids)
    _write_split(image_sets_root / 'smallval.txt', smallval_ids)
    _write_split(image_sets_root / 'trainval.txt', paired_ids)
    _write_split(image_sets_root / 'test.txt', val_ids if len(val_ids) > 0 else paired_ids)

    if logger:
        logger.info('Preprocess done: train=%d val=%d total=%d', 
                    len(train_ids), len(val_ids), num_total)

    return {
        'total': num_total,
        'train': len(train_ids),
        'val': len(val_ids),
        'kitti_root': str(kitti_root)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data/dataset into KITTI-style training layout')
    parser.add_argument('--dataset_root', type=str, default='../data/dataset_2', help='dataset root with bin/ and labels/')
    parser.add_argument('--classes', type=str,
                        default='Car,Human,ForkLift,CargoBike',
                        help='comma-separated classes to keep')
    args = parser.parse_args()

    classes = tuple([c.strip() for c in args.classes.split(',') if c.strip()])
    stats = preprocess_dataset(args.dataset_root, classes=classes)
    print('Preprocess done:', stats)
