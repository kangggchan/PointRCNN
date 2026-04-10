#!/usr/bin/env python
"""Minimal Open3D viewer for project-format point clouds and labels."""

import argparse
import os
import sys

import numpy as np
import open3d as o3d


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import lib.utils.kitti_utils as kitti_utils


BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

CLASS_COLORS = {
    'human': [0.15, 0.95, 0.35],
    'car': [1.0, 0.42, 0.12],
    'forklift': [1.0, 1.0, 0.0],
    'cargobike': [0.15, 0.65, 1.0],
}


def get_box_color(class_name):
    return CLASS_COLORS.get(class_name.lower(), [0.7, 0.7, 0.7])


def parse_kitti_label_file(label_file):
    class_names = []
    boxes = []

    if not os.path.exists(label_file):
        return class_names, np.zeros((0, 7), dtype=np.float32)

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 15:
                continue

            try:
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                ry = float(parts[14])
            except (ValueError, IndexError):
                continue

            class_names.append(parts[0])
            boxes.append([x, y, z, h, w, l, ry])

    if not boxes:
        return class_names, np.zeros((0, 7), dtype=np.float32)

    return class_names, np.asarray(boxes, dtype=np.float32)


def make_box_lineset(corners, color):
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    lineset.colors = o3d.utility.Vector3dVector([list(color)] * len(BOX_EDGES))
    return lineset


def build_point_colors(points_xyz, intensity):
    xy_range = np.linalg.norm(points_xyz[:, :2], axis=1)
    z = points_xyz[:, 2]

    range_norm = np.clip(xy_range / max(np.percentile(xy_range, 95), 1e-6), 0.0, 1.0)
    z_norm = np.clip((z - float(z.min())) / max(float(np.ptp(z)), 1e-6), 0.0, 1.0)
    i_norm = np.clip((intensity - float(intensity.min())) / max(float(np.ptp(intensity)), 1e-6), 0.0, 1.0)

    colors = np.zeros((points_xyz.shape[0], 3), dtype=np.float64)
    colors[:, 0] = 0.15 + 0.85 * z_norm
    colors[:, 1] = 0.20 + 0.75 * (1.0 - range_norm)
    colors[:, 2] = 0.30 + 0.70 * i_norm
    return np.clip(colors, 0.0, 1.0)


def visualize(bin_file, label_file, class_filter=None, show_axes=True, axis_size=2.5):
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f'Point cloud file not found: {bin_file}')
    if not os.path.exists(label_file):
        raise FileNotFoundError(f'Label file not found: {label_file}')

    points_raw = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points_xyz = points_raw[:, :3]
    intensity = points_raw[:, 3]

    class_names, boxes3d = parse_kitti_label_file(label_file)
    corners3d = kitti_utils.boxes3d_to_corners3d(boxes3d) if len(boxes3d) > 0 else np.zeros((0, 8, 3), dtype=np.float32)

    print(f'Point cloud: {bin_file}')
    print(f'Label file : {label_file}')
    print(f'Points     : {len(points_xyz)}')
    print(f'Boxes      : {len(corners3d)}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Project Format Viewer', width=1280, height=720)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.04, 0.05, 0.09])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(build_point_colors(points_xyz, intensity))
    vis.add_geometry(pcd)

    if show_axes:
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0.0, 0.0, 0.0]))

    for cls_name, corners in zip(class_names, corners3d):
        if class_filter and cls_name != class_filter:
            continue
        vis.add_geometry(make_box_lineset(corners, get_box_color(cls_name)))

    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)

    print('Controls: drag=rotate, Ctrl+drag=pan, scroll=zoom, Q/Esc=quit')
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Minimal project-format bin/label Open3D viewer')
    parser.add_argument('--bin_file', type=str, default='../data/dataset/KITTI/aug_scene/training/rectified_data/010311.bin', help='Path to .bin file (camera frame Nx4 float32).')
    parser.add_argument('--label_file', type=str, default='../output/rpn/default/eval/epoch_200/train_aug/detections/data/010311.txt', help='Path to KITTI label .txt file.')
    parser.add_argument('--class_filter', type=str, default=None, help='Optional class filter (exact label text).')
    parser.add_argument('--no_axes', action='store_true', help='Hide coordinate axes at origin.')
    parser.add_argument('--axis_size', type=float, default=2.5, help='Coordinate frame axis length in meters.')
    args = parser.parse_args()

    visualize(
        bin_file=args.bin_file,
        label_file=args.label_file,
        class_filter=args.class_filter,
        show_axes=(not args.no_axes),
        axis_size=float(args.axis_size),
    )


if __name__ == '__main__':
    main()
