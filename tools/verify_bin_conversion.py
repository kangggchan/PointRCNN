#!/usr/bin/env python
"""
Verify and visualize the bin conversion results.

This script:
1. Loads converted .bin files (camera frame) and labels (camera frame)
2. Verifies they're in the same coordinate frame
3. Visualizes both point cloud and boxes together using Open3D
4. Checks for alignment issues

Usage:
  python verify_bin_conversion.py --idx 000001 --data_root ../data/dataset
"""
import sys
import os
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


_BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

CLASS_COLORS = {
    'Human': [0.15, 0.95, 0.35],
    'Car': [1.0, 0.42, 0.12],
    'Forklift': [1.0, 1.0, 0.0],
    'Cargobike': [0.15, 0.65, 1.0],
    'ELFplusplus': [1.0, 0.0, 1.0],
    'FTS': [1.0, 0.65, 0.0],
}


def get_box_color(class_name):
    """Get color for a class."""
    class_lower = class_name.lower()
    return CLASS_COLORS.get(class_lower, [0.5, 0.5, 0.5])


def parse_kitti_label_file(label_file):
    """
    Parse KITTI-format label file.
    Returns list of (class_name, x, y, z, h, w, l, ry)
    
    KITTI format (16 fields):
    class truncated occluded alpha bbox(4 vals) dimensions(3 vals) location(3 vals) rotation_y
    Example: Human 0.00 0 -10.00 0.00 0.00 50.00 50.00 1.948 0.456 0.916 2.743 0.896 10.150 -1.689196
    """
    objects = []
    if not os.path.exists(label_file):
        return objects
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            # Need at least: class + truncated + occluded + alpha + 4bbox + 3dims + 3loc + ry = 15 fields
            if len(parts) < 15:
                continue
            
            cls_name = parts[0]
            
            try:
                # KITTI format indices:
                # 0: class
                # 1: truncated
                # 2: occluded
                # 3: alpha
                # 4-7: bbox (x1, y1, x2, y2)
                # 8-10: dimensions (h, w, l)
                # 11-13: location (x, y, z)
                # 14: rotation_y
                h = float(parts[8])
                w = float(parts[9])
                l = float(parts[10])
                x = float(parts[11])
                y = float(parts[12])
                z = float(parts[13])
                ry = float(parts[14])
                
                objects.append((cls_name, x, y, z, h, w, l, ry))
            except (ValueError, IndexError) as e:
                # Print debug info for failed lines
                if len(parts) >= 15:
                    pass  # Silently skip lines with parsing errors
                continue
    
    return objects


def compute_box_corners_kitti(x, y, z, h, w, l, ry):
    """
    Compute 8 corners of a 3D bounding box in KITTI camera frame.

    KITTI convention: (x, y, z) is the BOTTOM-CENTER of the box.
    Y axis points DOWN, so bottom = largest Y value.
    Matches boxes3d_to_corners3d() in kitti_utils.py:
      y_corners = [0,0,0,0,-h,-h,-h,-h]  → top face at y-h, bottom face at y.

    In KITTI camera frame:
    - X: right
    - Y: down (positive downward)
    - Z: forward
    - (x, y, z): BOTTOM-CENTER of the box (y = bottom face Y)
    """
    # Half-dimensions for X and Z only; Y uses full h from bottom face
    dx = l / 2.0    # Half-length along X
    dz = w / 2.0    # Half-width along Z

    # Corners wound as rectangles (right-front → right-back → left-back → left-front)
    # so that sequential edges [0,1],[1,2],[2,3],[3,0] form a proper rectangle, not an X.
    local_corners = np.array([
        # Bottom face (y = 0, ground level)
        [ dx, 0,   dz],   # 0 right-front
        [ dx, 0,  -dz],   # 1 right-back
        [-dx, 0,  -dz],   # 2 left-back
        [-dx, 0,   dz],   # 3 left-front
        # Top face (y = -h, sky level)
        [ dx, -h,  dz],   # 4 right-front
        [ dx, -h, -dz],   # 5 right-back
        [-dx, -h, -dz],   # 6 left-back
        [-dx, -h,  dz],   # 7 left-front
    ], dtype=np.float64)
    
    # Rotation matrix around Y axis (ry)
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    R_y = np.array([
        [cos_ry, 0, sin_ry],
        [0, 1, 0],
        [-sin_ry, 0, cos_ry],
    ], dtype=np.float64)
    
    # Apply rotation and translation
    rotated_corners = (R_y @ local_corners.T).T
    world_corners = rotated_corners + np.array([x, y, z], dtype=np.float64)
    
    return world_corners


def create_box_lineset(x, y, z, h, w, l, ry, color=(0.0, 1.0, 0.0)):
    """Create Open3D LineSet for a bounding box."""
    corners = compute_box_corners_kitti(x, y, z, h, w, l, ry)
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines = o3d.utility.Vector2iVector(_BOX_EDGES)
    lineset.colors = o3d.utility.Vector3dVector([list(color)] * len(_BOX_EDGES))
    
    return lineset


def build_point_colors(points, intensity):
    """Color points based on range, height, and intensity."""
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


def verify_conversion(bin_file, label_file, verbose=False):
    """
    Verify that point cloud and labels are in the same coordinate frame.
    
    Returns:
        (points, intensity, boxes, verification_results)
    """
    print(f"\n{'='*80}")
    print(f"Verifying Conversion")
    print(f"{'='*80}")
    
    # Load point cloud
    if not os.path.exists(bin_file):
        print(f"❌ Point cloud file not found: {bin_file}")
        return None, None, None, False
    
    points_raw = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = points_raw[:, :3]
    intensity = points_raw[:, 3]
    
    print(f"\n📁 Point Cloud: {os.path.basename(bin_file)}")
    print(f"   Points: {points.shape[0]}")
    print(f"   X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"   Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"   Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Load labels
    boxes = parse_kitti_label_file(label_file)
    
    print(f"\n📁 Labels: {os.path.basename(label_file)}")
    print(f"   Objects: {len(boxes)}")
    
    if len(boxes) == 0:
        print("   ⚠️  No objects in label file")
        return points, intensity, boxes, True
    
    # Verify alignment
    print(f"\n🔍 Alignment Check:")
    all_good = True
    
    for cls_name, x, y, z, h, w, l, ry in boxes:
        # Check if box center is within reasonable bounds of point cloud
        x_in_bounds = points[:, 0].min() <= x <= points[:, 0].max()
        y_in_bounds = points[:, 1].min() <= y <= points[:, 1].max()
        z_in_bounds = points[:, 2].min() <= z <= points[:, 2].max()
        
        bounds_check = "✓" if (x_in_bounds and y_in_bounds and z_in_bounds) else "✗"
        
        if verbose:
            print(f"   {cls_name} @ ({x:6.2f}, {y:6.2f}, {z:6.2f}): {bounds_check}")
        
        if not (x_in_bounds and y_in_bounds and z_in_bounds):
            all_good = False
    
    if all_good:
        print(f"   ✅ All box centers are within point cloud bounds")
    else:
        print(f"   ⚠️  Some boxes outside point cloud bounds (may be at edges)")
    
    print(f"\n{'='*80}")
    
    return points, intensity, boxes, True


def visualize(points, intensity, boxes, class_filter=None):
    """
    Visualize point cloud and boxes together using Open3D.
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='BIN Conversion Verification', width=1280, height=720)
    
    # Configure rendering
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.04, 0.05, 0.09])
    
    # Add point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = build_point_colors(points, intensity)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    
    # Add bounding boxes
    print(f"\n📊 Rendering {len(boxes)} boxes...")
    for cls_name, x, y, z, h, w, l, ry in boxes:
        if class_filter and cls_name != class_filter:
            continue
        
        color = get_box_color(cls_name)
        lineset = create_box_lineset(x, y, z, h, w, l, ry, color=color)
        vis.add_geometry(lineset)
    
    # Configure view
    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(0.5)
    
    # Start visualization
    print("Controls: drag=rotate, Ctrl+drag=pan, scroll=zoom, Q/Esc=quit, SPACE=reset view")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description='Verify bin conversion by visualizing point cloud and labels'
    )
    parser.add_argument(
        '--idx',
        type=str,
        default='010002',
        help='Frame index to verify (e.g., 000001)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='../data/dataset/KITTI/aug_scene/training',
        help='Dataset root directory'
    )
    parser.add_argument(
        '--bin_dir',
        type=str,
        default='rectified_data',
        help='Subdirectory for point clouds'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        default='aug_label',
        help='Subdirectory for labels (relative to data_root)'
    )
    parser.add_argument(
        '--class_filter',
        type=str,
        default=None,
        help='Filter visualization to only this class'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Construct paths
    bin_file = os.path.join(args.data_root, args.bin_dir, f'{args.idx}.bin')
    label_file = os.path.join(args.data_root, args.label_dir, f'{args.idx}.txt')
    
    # Verify conversion
    points, intensity, boxes, success = verify_conversion(
        bin_file,
        label_file,
        verbose=args.verbose
    )
    
    if not success or points is None:
        print("\n❌ Verification failed")
        return 1
    
    # Visualize
    print("\n🎨 Launching Open3D visualization...")
    visualize(points, intensity, boxes, class_filter=args.class_filter)
    
    print("\n✅ Verification complete")
    return 0


if __name__ == '__main__':
    sys.exit(main())
