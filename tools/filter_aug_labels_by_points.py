#!/usr/bin/env python3
"""
Filter augmented labels by minimum point count inside bounding boxes (IN-PLACE).

This script removes label entries (objects) that have fewer than a specified
number of points inside their bounding box. It uses the same box convention as
the training code and analyze_aug_scene_points_in_boxes.py:
- Box location is bottom center
- Rotation is around Y-axis
- Local X uses length, local Z uses width
- Local Y spans from 0 at the bottom face to -h at the top face

Modifies the original label files directly. Optionally backs up before filtering.

Usage:
    python filter_aug_labels_by_points.py \\
        --aug_label_dir ../data/dataset/KITTI/aug_scene/training/aug_label \\
        --aug_pts_dir ../data/dataset/KITTI/aug_scene/training/rectified_data \\
        --class_name Human \\
        --min_points 100 \\
        --backup_dir ../data/dataset/KITTI/aug_scene/training/aug_label_backup
"""

import os
import sys
import argparse
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
import shutil


def setup_logging(log_file=None):
    """Setup logging to file and console."""
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_point_cloud(bin_path):
    """Load point cloud from .bin file (N x 4: [x, y, z, intensity])."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Return only xyz


def count_points_in_box(points, bbox_params):
    """
    Count points inside a repo-convention KITTI 3D box.

    This repository follows the same convention as Object3d.generate_corners3d():
    - Box location (x, y, z) is bottom center.
    - Dimensions are stored as (h, w, l).
    - Local X axis uses length, local Z axis uses width.
    - Local Y spans from 0 at the bottom face to -h at the top face.
    - Rotation ry is around Y axis.
    
    Args:
        points: (N, 3) array of point coordinates in KITTI camera frame
        bbox_params: (x, y_bottom, z, length, width, height, rotation_angle_ry)
    
    Returns:
        count: number of points inside the box
    """
    x, y, z, length, width, height, rotation_angle = bbox_params
    
    # Translate points relative to box center (x, y, z)
    translated = points - np.array([x, y, z], dtype=np.float32)
    
    # Inverse rotation: rotate points by -ry around Y-axis to align with box axes
    # ry is rotation around Y-axis: [cos  0 -sin] [  1  sin  0]  [cos  0 -sin]
    #                               [  0  1   0 ] [  0  cos  0]=[  0  1   0 ]
    #                               [sin  0  cos] [ -sin  0  cos] [sin  0  cos]
    cos_ry = np.cos(rotation_angle)
    sin_ry = np.sin(rotation_angle)
    
    # Rotation matrix around Y-axis (for inverse rotation)
    rot_inv = np.array([
        [cos_ry, 0.0, -sin_ry],
        [0.0, 1.0, 0.0],
        [sin_ry, 0.0, cos_ry]
    ], dtype=np.float32)
    
    # Apply inverse rotation
    local = translated @ rot_inv.T
    
    # Match lib/utils/object3d.py:
    # x_corners use length, z_corners use width, y_corners go from 0 to -h.
    inside_x = np.abs(local[:, 0]) <= (length / 2.0)
    inside_z = np.abs(local[:, 2]) <= (width / 2.0)
    inside_y = (local[:, 1] <= 0.0) & (local[:, 1] >= -height)
    
    return np.sum(inside_x & inside_y & inside_z)


def parse_kitti_label(line):
    """Parse a KITTI format label line."""
    parts = line.strip().split()
    
    if len(parts) < 15:
        return None
    
    obj = {
        'cls_type': parts[0],
        'truncation': float(parts[1]),
        'occlusion': int(parts[2]),
        'alpha': float(parts[3]),
        'box2d': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
        'h': float(parts[8]),
        'w': float(parts[9]),
        'l': float(parts[10]),
        'x': float(parts[11]),
        'y': float(parts[12]),
        'z': float(parts[13]),
        'ry': float(parts[14]),
        'score': float(parts[15]) if len(parts) > 15 else -1.0
    }
    
    return obj


def format_kitti_label(obj):
    """Format object back to KITTI label string."""
    label_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % (
        obj['cls_type'],
        obj['truncation'],
        int(obj['occlusion']),
        obj['alpha'],
        obj['box2d'][0], obj['box2d'][1], obj['box2d'][2], obj['box2d'][3],
        obj['h'], obj['w'], obj['l'],
        obj['x'], obj['y'], obj['z'],
        obj['ry']
    )
    
    if obj['score'] >= 0:
        label_str += ' %.6f' % obj['score']
    
    return label_str


def filter_label_file(label_path, pts_path, filter_config, logger):
    """
    Filter a single label file based on point counts.
    
    Args:
        label_path: path to label file
        pts_path: path to point cloud file
        filter_config: dict with {class_name: min_points}
        logger: logging instance
    
    Returns:
        (kept_labels, filtered_labels, stats_dict)
    """
    
    # Load point cloud
    if not os.path.exists(pts_path):
        logger.warning(f'Point cloud not found: {pts_path}')
        return [], [], {}
    
    points = load_point_cloud(pts_path)
    
    # Load labels
    kept_labels = []
    filtered_labels = []
    stats = defaultdict(int)
    
    with open(label_path, 'r') as f:
        for line in f:
            obj = parse_kitti_label(line)
            if obj is None:
                continue
            
            class_name = obj['cls_type']
            
            # Check if this class should be filtered
            if class_name not in filter_config:
                kept_labels.append(line.rstrip('\n'))
                continue
            
            min_points = filter_config[class_name]
            
            # Count points in bounding box
            bbox_params = (obj['x'], obj['y'], obj['z'],
                          obj['l'], obj['w'], obj['h'],
                          obj['ry'])
            
            point_count = count_points_in_box(points, bbox_params)
            
            stats[f'{class_name}_checked'] += 1
            
            if point_count >= min_points:
                kept_labels.append(line.rstrip('\n'))
                stats[f'{class_name}_kept'] += 1
            else:
                filtered_labels.append(line.rstrip('\n'))
                stats[f'{class_name}_removed'] += 1
    
    return kept_labels, filtered_labels, dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description='Filter augmented labels by minimum point count in bounding box (IN-PLACE)'
    )
    parser.add_argument(
        '--aug_label_dir',
        required=True,
        help='Directory to augmented labels (aug_label) - will be modified in-place'
    )
    parser.add_argument(
        '--aug_pts_dir',
        required=True,
        help='Directory to augmented point clouds (rectified_data)'
    )
    parser.add_argument(
        '--class_name',
        required=True,
        nargs='+',
        help='Class name(s) to filter (e.g., Human CargoBike)'
    )
    parser.add_argument(
        '--min_points',
        required=True,
        nargs='+',
        type=int,
        help='Minimum points required (order matches class_name)'
    )
    parser.add_argument(
        '--backup_dir',
        default=None,
        help='Optional backup directory for original labels before filtering'
    )
    parser.add_argument(
        '--keep_removed_log',
        action='store_true',
        default=True,
        help='Keep log of removed entries'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.class_name) != len(args.min_points):
        print('ERROR: Number of class_name and min_points must match')
        sys.exit(1)
    
    # Create filter config
    filter_config = dict(zip(args.class_name, args.min_points))
    
    # Setup logging
    log_file = None
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
        log_file = os.path.join(args.backup_dir, 'filtering_log.txt')
    
    logger = setup_logging(log_file)
    
    logger.info('='*80)
    logger.info('AUGMENTED LABEL FILTERING (IN-PLACE)')
    logger.info('='*80)
    logger.info(f'Label directory: {args.aug_label_dir}')
    logger.info(f'Point cloud directory: {args.aug_pts_dir}')
    logger.info(f'Filter config: {filter_config}')
    if args.backup_dir:
        logger.info(f'Backup directory: {args.backup_dir}')
    
    # Backup original labels if requested
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
        for label_file in os.listdir(args.aug_label_dir):
            if label_file.endswith('.txt'):
                src = os.path.join(args.aug_label_dir, label_file)
                dst = os.path.join(args.backup_dir, label_file)
                shutil.copy2(src, dst)
        logger.info(f'Backed up original labels to: {args.backup_dir}')
    
    # Process all label files
    total_stats = defaultdict(int)
    num_files = 0
    errors = 0
    
    label_files = sorted([f for f in os.listdir(args.aug_label_dir) if f.endswith('.txt')])
    logger.info(f'Found {len(label_files)} label files to process\n')
    
    removed_entries_log = {}  # Track removed entries if requested
    
    for label_file in label_files:
        label_path = os.path.join(args.aug_label_dir, label_file)
        file_id = os.path.splitext(label_file)[0]
        pts_path = os.path.join(args.aug_pts_dir, f'{file_id}.bin')
        
        try:
            kept, removed, stats = filter_label_file(label_path, pts_path, filter_config, logger)
            
            if stats:
                num_files += 1
                
                # Update total stats
                for key, val in stats.items():
                    total_stats[key] += val
                
                # Write filtered labels back to original file (IN-PLACE)
                with open(label_path, 'w') as f:
                    for line in kept:
                        f.write(line + '\n')
                
                # Keep track of removed entries
                if removed:
                    removed_entries_log[label_file] = removed
                
                # Log per-file stats
                if any(v > 0 for v in stats.values()):
                    logger.info(f'{label_file}:')
                    for class_name, min_pts in filter_config.items():
                        checked = stats.get(f'{class_name}_checked', 0)
                        kept_cnt = stats.get(f'{class_name}_kept', 0)
                        removed_cnt = stats.get(f'{class_name}_removed', 0)
                        if checked > 0:
                            logger.info(f'  {class_name} (>{min_pts}pts): {kept_cnt}/{checked} kept, {removed_cnt} removed')
        
        except Exception as e:
            logger.error(f'Error processing {label_file}: {str(e)}')
            errors += 1
    
    # Log summary statistics
    logger.info('\n' + '='*80)
    logger.info('FILTERING SUMMARY')
    logger.info('='*80)
    logger.info(f'Files processed: {num_files}')
    logger.info(f'Errors: {errors}')
    
    for class_name, min_pts in filter_config.items():
        checked = total_stats.get(f'{class_name}_checked', 0)
        kept = total_stats.get(f'{class_name}_kept', 0)
        removed = total_stats.get(f'{class_name}_removed', 0)
        
        if checked > 0:
            removal_pct = 100 * removed / checked
            logger.info(f'\n{class_name} (min {min_pts} points):')
            logger.info(f'  Total checked: {checked}')
            logger.info(f'  Kept: {kept} ({100*kept/checked:.1f}%)')
            logger.info(f'  Removed: {removed} ({removal_pct:.1f}%)')
    
    # Save removed entries log if requested
    if args.keep_removed_log and removed_entries_log:
        if args.backup_dir:
            removed_log_file = os.path.join(args.backup_dir, 'removed_entries.txt')
        else:
            removed_log_file = os.path.join(args.aug_label_dir, '..', 'removed_entries_log.txt')
        
        os.makedirs(os.path.dirname(removed_log_file), exist_ok=True)
        with open(removed_log_file, 'w') as f:
            for label_file, entries in removed_entries_log.items():
                f.write(f'\n# From: {label_file}\n')
                for entry in entries:
                    f.write(entry + '\n')
        logger.info(f'\nRemoved entries log saved to: {removed_log_file}')
    
    logger.info(f'\n✓ Labels modified IN-PLACE in: {args.aug_label_dir}')
    logger.info('='*80)
    
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
