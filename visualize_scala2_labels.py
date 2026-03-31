import os
import sys
import time
import argparse
import importlib.util
import numpy as np
import open3d as o3d

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import lib.utils.calibration as calibration_module


_BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

CLASS_COLORS = {
    'human': [0.15, 0.95, 0.35],
    'car': [1.0, 0.42, 0.12],
    'cyclist': [0.15, 0.65, 1.0],
}


def build_point_colors(points: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    xy_range = np.linalg.norm(points[:, :2], axis=1)
    z = points[:, 2]

    range_norm = np.clip(xy_range / max(np.percentile(xy_range, 95), 1e-6), 0.0, 1.0)
    z_min = float(z.min())
    z_span = max(float(np.ptp(z)), 1e-6)
    z_norm = np.clip((z - z_min) / z_span, 0.0, 1.0)

    i_min = float(intensity.min())
    i_span = max(float(intensity.max() - i_min), 1e-6)
    intensity_norm = np.clip((intensity - i_min) / i_span, 0.0, 1.0)

    colors = np.zeros((points.shape[0], 3), dtype=np.float64)
    colors[:, 0] = 0.15 + 0.85 * z_norm
    colors[:, 1] = 0.20 + 0.75 * (1.0 - range_norm)
    colors[:, 2] = 0.30 + 0.70 * intensity_norm
    return np.clip(colors, 0.0, 1.0)


def load_infer_helpers():
    infer_path = os.path.join(THIS_DIR, 'infer_single_bin.py')
    if not os.path.isfile(infer_path):
        raise FileNotFoundError(f'infer_single_bin.py not found: {infer_path}')

    spec = importlib.util.spec_from_file_location('infer_single_bin_runtime', infer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.boxes_rect_to_lidar_corners, module._DEFAULT_CALIB


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize scala2 labels in Open3D as a frame player')
    parser.add_argument('--data_root', type=str, default='data/dataset', help='Root containing bin/ and labels/')
    parser.add_argument('--bin_subdir', type=str, default='bin', help='Subfolder for point clouds')
    parser.add_argument('--label_subdir', type=str, default='labels', help='Subfolder for label txt files')
    parser.add_argument('--start_idx', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=3000, help='End frame index (inclusive)')
    parser.add_argument('--fps', type=float, default=24.0, help='Playback FPS')
    parser.add_argument('--point_size', type=float, default=2.0, help='Point size in viewer')
    parser.add_argument('--label_filter', type=str, default=None,
                        help='Optional class filter, e.g. Car. If None, draw all labels')
    parser.add_argument('--coord_mode', type=str, default='lidar', choices=['rect', 'lidar'],
                        help='Label coordinate mode for labels formatted as x y z l w h yaw: rect=PointRCNN rect camera coords, lidar=direct LiDAR coords')
    parser.add_argument('--calib_file', type=str, default=None,
                        help='Calibration file for rect->lidar conversion when coord_mode=rect')
    return parser.parse_args()


def frame_name(idx: int) -> str:
    return f'{idx:06d}'


def parse_label_file(label_file: str, class_filter: str = None):
    boxes = []
    if not os.path.isfile(label_file):
        return boxes

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 8:
                continue

            cls = parts[0]
            if class_filter is not None and cls != class_filter:
                continue

            try:
                x, y, z, l, w, h, yaw = map(float, parts[1:])
            except ValueError:
                continue

            boxes.append((cls, x, y, z, l, w, h, yaw))
    return boxes


def box_to_corners(x, y, z, l, w, h, yaw) -> np.ndarray:
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    local = np.array([
        [dx, dy, -dz],
        [dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx, dy, -dz],
        [dx, dy, dz],
        [dx, -dy, dz],
        [-dx, -dy, dz],
        [-dx, dy, dz],
    ], dtype=np.float64)
    rot = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return (rot @ local.T).T + np.array([x, y, z], dtype=np.float64)


def make_lineset_from_box(x, y, z, l, w, h, yaw, color=(0.0, 1.0, 0.0)):
    corners = box_to_corners(x, y, z, l, w, h, yaw)
    return make_lineset_from_corners(corners, color=color)


def make_lineset_from_corners(corners: np.ndarray, color=(0.0, 1.0, 0.0)):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(_BOX_EDGES))
    return ls


def load_frame(bin_file: str, label_file: str, class_filter: str = None):
    raw = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = raw[:, :3]
    intensity = raw[:, 3].astype(np.float64)
    colors = build_point_colors(points, intensity)
    boxes = parse_label_file(label_file, class_filter)
    return points, colors, boxes


def main():
    args = parse_args()

    bin_dir = os.path.join(args.data_root, args.bin_subdir)
    label_dir = os.path.join(args.data_root, args.label_subdir)

    if not os.path.isdir(bin_dir):
        raise FileNotFoundError(f'Bin directory not found: {bin_dir}')
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f'Label directory not found: {label_dir}')

    calib = None
    boxes_rect_to_lidar_corners_fn = None
    if args.coord_mode == 'rect':
        boxes_rect_to_lidar_corners_fn, default_calib = load_infer_helpers()
        if args.calib_file:
            if not os.path.isfile(args.calib_file):
                raise FileNotFoundError(f'Calibration file not found: {args.calib_file}')
            calib = calibration_module.Calibration(args.calib_file)
            print(f'[INFO] coord_mode=rect, using calibration: {args.calib_file}')
        else:
            calib = calibration_module.Calibration(default_calib)
            print('[WARNING] coord_mode=rect but --calib_file not provided; using default KITTI fallback calibration.')

    frame_period = 1.0 / max(args.fps, 1e-6)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='scala2 label playback', width=1280, height=720)

    render_option = vis.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = np.array([0.04, 0.05, 0.09])

    pcd = o3d.geometry.PointCloud()
    # Don't add empty pcd yet — add after first frame so camera fits real data

    current_lines = []
    first_frame = True

    print(f'Playing frames {args.start_idx:06d} -> {args.end_idx:06d} at {args.fps:.2f} FPS')
    print('Controls: drag=rotate, Ctrl+drag=pan, scroll=zoom, Q/Esc=quit')

    for idx in range(args.start_idx, args.end_idx + 1):
        t0 = time.time()

        name = frame_name(idx)
        bin_file = os.path.join(bin_dir, f'{name}.bin')
        label_file = os.path.join(label_dir, f'{name}.txt')

        if not os.path.isfile(bin_file):
            print(f'[SKIP] Missing bin: {bin_file}')
            continue

        points, colors, boxes = load_frame(bin_file, label_file, args.label_filter)

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first_frame:
            vis.add_geometry(pcd)          # first add — camera auto-fits to real data
            first_frame = False
        else:
            vis.update_geometry(pcd)

        for ls in current_lines:
            vis.remove_geometry(ls, reset_bounding_box=False)
        current_lines = []

        for cls, x, y, z, l, w, h, yaw in boxes:
            color = CLASS_COLORS.get(cls.lower(), [1.0, 1.0, 1.0])
            if args.coord_mode == 'rect':
                box_rect = np.array([[x, y, z, h, w, l, yaw]], dtype=np.float64)
                corners_lidar = boxes_rect_to_lidar_corners_fn(box_rect, calib)[0]
                ls = make_lineset_from_corners(corners_lidar, color=color)
            else:
                ls = make_lineset_from_box(x, y, z, l, w, h, yaw, color=color)
            vis.add_geometry(ls, reset_bounding_box=False)
            current_lines.append(ls)

        vis.poll_events()
        vis.update_renderer()

        print(f'[FRAME] {name}  points={points.shape[0]}  boxes={len(boxes)}')

        elapsed = time.time() - t0
        sleep_time = frame_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print('Playback finished. Close the window to exit.')
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()


if __name__ == '__main__':
    main()
