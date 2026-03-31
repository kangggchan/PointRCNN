#!/usr/bin/env python
"""
Convert dataset from custom Scala2 LiDAR format to KITTI camera frame.

Input (dataset_original):  LiDAR-frame bins + 8-field labels
Output (dataset):          Camera-frame bins + 15-field KITTI labels

Point cloud transform:
  x_cam = -y_l        (right)
  y_cam = -z_l        (down, intensity unchanged)
  z_cam =  x_l        (forward/depth)

Label transform:
  x_cam        = -y_l
  y_cam_bottom = -z_l + h/2   (KITTI bottom-center convention)
  z_cam        =  x_l
  ry           = normalise(-yaw - pi/2)

See DATASET_FORMAT.md for the full format specification.
"""
import sys
import os
import numpy as np
import argparse
import math
from pathlib import Path


def _normalize_angle(angle):
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def transform_lidar_to_camera(points_lidar):
    """
    Transform points from LiDAR frame to KITTI camera frame.
    
    Args:
        points_lidar: (N, 4) array with [x_l, y_l, z_l, intensity]
    
    Returns:
        points_camera: (N, 4) array with [x_cam, y_cam, z_cam, intensity]
    """
    points_camera = np.zeros_like(points_lidar)
    
    # Apply coordinate transformation
    points_camera[:, 0] = -points_lidar[:, 1]  # x_cam = -y_l
    points_camera[:, 1] = -points_lidar[:, 2]  # y_cam = -z_l
    points_camera[:, 2] = points_lidar[:, 0]   # z_cam = x_l
    points_camera[:, 3] = points_lidar[:, 3]   # intensity unchanged
    
    return points_camera


def convert_label_custom_to_kitti(label_line):
    """
    Convert a single label line from custom LiDAR format to KITTI format.
    
    Custom format: Class x_l y_l z_l l w h yaw
    KITTI format:  Class Trunc Occl Alpha BBox(4) h w l x_cam y_cam z_cam ry
    
    Args:
        label_line: String with 8 fields (class + 7 values)
    
    Returns:
        kitti_line: String in KITTI format (16 fields)
    """
    parts = label_line.strip().split()
    
    if len(parts) != 8:
        return None
    
    cls_name = parts[0]
    try:
        x_l, y_l, z_l, l, w, h, yaw = [float(v) for v in parts[1:8]]
    except ValueError:
        return None
    
    # LiDAR true-center → KITTI camera bottom-center.
    # The C++ kernel (pts_in_box3d_cpu) treats the stored y as bottom_y and
    # computes cy = bottom_y - h/2, so we must store the bottom face, not the
    # geometric center.  bottom_y = true_center_y + h/2 = (-z_l) + h/2.
    x_cam = -y_l
    y_cam = -z_l + (h / 2.0)  # KITTI bottom-center: true_center_y + h/2
    z_cam = x_l
    
    # Rotation angle transformation: Z-axis yaw → Y-axis ry
    ry = _normalize_angle(-yaw - math.pi / 2.0)
    
    # KITTI format fields
    trunc = 0.0
    occl = 0
    alpha = -10.0
    x1, y1, x2, y2 = 0.0, 0.0, 50.0, 50.0
    
    # Construct KITTI line: cls trunc occl alpha bbox(4) h w l xyz ry
    kitti_line = (
        f"{cls_name} {trunc:.2f} {occl:d} {alpha:.2f} "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"{h:.3f} {w:.3f} {l:.3f} {x_cam:.3f} {y_cam:.3f} {z_cam:.3f} {ry:.6f}"
    )
    
    return kitti_line


def convert_bin_file(input_path, output_path, verbose=False):
    """
    Convert a single .bin file from LiDAR to camera frame.
    
    Args:
        input_path: Path to input .bin file (LiDAR frame)
        output_path: Path to output .bin file (camera frame)
        verbose: Print conversion details
    """
    # Read point cloud
    points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)
    
    if verbose:
        print(f"  Input: {points.shape[0]} points")
        print(f"    Range X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"    Range Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"    Range Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Transform to camera frame
    points_camera = transform_lidar_to_camera(points)
    
    if verbose:
        print(f"  Output (camera frame): {points_camera.shape[0]} points")
        print(f"    Range X: [{points_camera[:, 0].min():.2f}, {points_camera[:, 0].max():.2f}]")
        print(f"    Range Y: [{points_camera[:, 1].min():.2f}, {points_camera[:, 1].max():.2f}]")
        print(f"    Range Z: [{points_camera[:, 2].min():.2f}, {points_camera[:, 2].max():.2f}]")
    
    # Write converted point cloud
    points_camera.astype(np.float32).tofile(output_path)
    
    return points.shape[0]


def convert_label_file(input_path, output_path, verbose=False):
    """
    Convert a label file from custom LiDAR format to KITTI format.
    
    Args:
        input_path: Path to input label file (custom format)
        output_path: Path to output label file (KITTI format)
        verbose: Print conversion details
    
    Returns:
        Number of objects converted
    """
    kitti_lines = []
    
    if not os.path.exists(input_path):
        return 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            kitti_line = convert_label_custom_to_kitti(line)
            if kitti_line:
                kitti_lines.append(kitti_line)
    
    if verbose:
        print(f"  Converted {len(kitti_lines)} objects")
    
    # Write KITTI format labels
    with open(output_path, 'w', encoding='utf-8') as f:
        if kitti_lines:
            f.write('\n'.join(kitti_lines) + '\n')
    
    return len(kitti_lines)



def convert_dataset(input_dir, output_dir, verbose=False):
    """
    Convert entire dataset from custom LiDAR format to KITTI camera frame.
    
    Creates output_dir structure:
      output_dir/
        bin/      (converted .bin files)
        labels/   (converted labels)
    
    Args:
        input_dir: Source dataset directory (should contain bin/ and labels/)
        output_dir: Output dataset directory (will be created)
        verbose: Print detailed progress
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Subdirectories
    input_bin_dir = input_dir / 'bin'
    input_label_dir = input_dir / 'labels'
    output_bin_dir = output_dir / 'bin'
    output_label_dir = output_dir / 'labels'
    
    # Verify inputs
    if not input_bin_dir.exists():
        print(f"❌ Input bin directory not found: {input_bin_dir}")
        return 0, 0
    
    if not input_label_dir.exists():
        print(f"❌ Input label directory not found: {input_label_dir}")
        return 0, 0
    
    # Create output directories
    output_bin_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Converting Custom LiDAR Dataset to KITTI Camera Frame")
    print("=" * 80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all .bin files
    bin_files = sorted(input_bin_dir.glob('*.bin'))
    label_files = sorted(input_label_dir.glob('*.txt'))
    
    if not bin_files:
        print(f"❌ No .bin files found in {input_bin_dir}")
        return 0, 0
    
    print(f"Converting {len(bin_files)} .bin files and {len(label_files)} label files...")
    print("-" * 80)
    
    total_points = 0
    total_objects = 0
    
    # Convert .bin files
    print(f"\n{'FILE':<20} {'POINTS':<10} {'STATUS'}")
    print("-" * 40)
    
    for i, bin_file in enumerate(bin_files):
        try:
            output_file = output_bin_dir / bin_file.name
            num_points = convert_bin_file(bin_file, output_file, verbose=verbose)
            total_points += num_points
            
            status = "✓"
            print(f"{bin_file.name:<20} {num_points:<10} {status}")
            
        except Exception as e:
            print(f"{bin_file.name:<20} {'ERROR':<10} ❌ {e}")
    
    # Convert label files
    print(f"\n{'FILE':<20} {'OBJECTS':<10} {'STATUS'}")
    print("-" * 40)
    
    for i, label_file in enumerate(label_files):
        try:
            output_file = output_label_dir / label_file.name
            num_objects = convert_label_file(label_file, output_file, verbose=verbose)
            total_objects += num_objects
            
            status = "✓"
            print(f"{label_file.name:<20} {num_objects:<10} {status}")
            
        except Exception as e:
            print(f"{label_file.name:<20} {'ERROR':<10} ❌ {e}")
    
    print("\n" + "=" * 80)
    print("✅ Conversion Complete!")
    print("=" * 80)
    print(f"  .bin files:    {len(bin_files)}")
    print(f"  Total points:  {total_points:,}")
    print(f"  Label files:   {len(label_files)}")
    print(f"  Total objects: {total_objects:,}")
    print(f"\n📁 Output: {output_dir}")
    print(f"  ├── bin/    ({len(list(output_bin_dir.glob('*.bin')))} files)")
    print(f"  └── labels/ ({len(list(output_label_dir.glob('*.txt')))} files)")
    
    return len(bin_files), len(label_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert custom LiDAR dataset to KITTI camera frame format'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='../data/dataset_original',
        help='Source dataset directory with raw Scala2 data (bin/ and labels/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/dataset',
        help='Output directory for camera-frame data (will be created)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed conversion info'
    )
    
    args = parser.parse_args()
    
    # Run conversion
    convert_dataset(args.input_dir, args.output_dir, verbose=args.verbose)
