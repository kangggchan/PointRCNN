#!/usr/bin/env python
"""
Analyze number of LiDAR points inside 3D boxes for KITTI-style aug_scene/training data.

This script scans label and bin files, computes points-in-box counts for each object,
and writes:
- per-box CSV
- per-class summary CSV
- histogram + boxplot + distance-vs-count scatter plots

Usage examples:
  python tools/analyze_aug_scene_points_in_boxes.py \
      --data_root data/dataset/KITTI/aug_scene/training \
      --out_dir output/aug_scene_point_analysis

  python tools/analyze_aug_scene_points_in_boxes.py \
      --data_root data/dataset/KITTI/aug_scene/training \
      --max_frames 500
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def parse_kitti_label_file(label_file):
    """Return list of dicts with KITTI 3D box fields."""
    objects = []
    if not os.path.exists(label_file):
        return objects

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 15:
                continue

            try:
                objects.append(
                    {
                        "class_name": parts[0],
                        "h": float(parts[8]),
                        "w": float(parts[9]),
                        "l": float(parts[10]),
                        "x": float(parts[11]),
                        "y": float(parts[12]),
                        "z": float(parts[13]),
                        "ry": float(parts[14]),
                    }
                )
            except (ValueError, IndexError):
                continue

    return objects


def points_in_kitti_box_mask(points_xyz, x, y, z, h, w, l, ry):
    """
    Return boolean mask for points inside a KITTI box.

    KITTI convention used here:
    - Box location (x, y, z) is bottom center.
    - Dimensions are (h, w, l).
    - Rotation ry is around Y axis.
    """
    translated = points_xyz - np.array([x, y, z], dtype=np.float64)

    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    rot_inv = np.array(
        [[cos_ry, 0.0, -sin_ry], [0.0, 1.0, 0.0], [sin_ry, 0.0, cos_ry]],
        dtype=np.float64,
    )

    local = translated @ rot_inv.T

    inside_x = np.abs(local[:, 0]) <= (l / 2.0)
    inside_z = np.abs(local[:, 2]) <= (w / 2.0)
    inside_y = (local[:, 1] <= 0.0) & (local[:, 1] >= -h)
    return inside_x & inside_y & inside_z


def safe_float(v):
    return float(v) if np.isfinite(v) else float("nan")


def write_per_box_csv(rows, out_path):
    fieldnames = [
        "frame_id",
        "box_index",
        "class_name",
        "x",
        "y",
        "z",
        "h",
        "w",
        "l",
        "ry",
        "distance",
        "volume",
        "point_count",
        "density_pts_per_m3",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_class_summary_csv(rows, out_path):
    fieldnames = [
        "class_name",
        "num_boxes",
        "mean_points",
        "p25_points",
        "p50_points",
        "p75_points",
        "min_points",
        "max_points",
        "mean_density",
        "median_density",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, out_dir):
    import matplotlib.pyplot as plt

    if not rows:
        return

    counts = np.array([r["point_count"] for r in rows], dtype=np.float64)
    dists = np.array([r["distance"] for r in rows], dtype=np.float64)
    classes = sorted({r["class_name"] for r in rows})

    # 1) Global histogram
    fig = plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, color="#2E7D32", edgecolor="white", alpha=0.9)
    plt.title("Points per Box Distribution (All Classes)")
    plt.xlabel("Point count inside 3D box")
    plt.ylabel("Number of boxes")
    plt.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_points_per_box.png"), dpi=180)
    plt.close(fig)

    # 2) Boxplot by class
    fig = plt.figure(figsize=(12, 6))
    class_series = []
    class_labels = []
    for cls in classes:
        arr = np.array([r["point_count"] for r in rows if r["class_name"] == cls], dtype=np.float64)
        if arr.size == 0:
            continue
        class_series.append(arr)
        class_labels.append(cls)

    if class_series:
        plt.boxplot(class_series, labels=class_labels, showfliers=False)
        plt.title("Points per Box by Class")
        plt.xlabel("Class")
        plt.ylabel("Point count")
        plt.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "boxplot_points_per_class.png"), dpi=180)
    plt.close(fig)

    # 3) Distance vs count scatter
    fig = plt.figure(figsize=(10, 6))
    for cls in classes:
        xs = np.array([r["distance"] for r in rows if r["class_name"] == cls], dtype=np.float64)
        ys = np.array([r["point_count"] for r in rows if r["class_name"] == cls], dtype=np.float64)
        if xs.size == 0:
            continue
        plt.scatter(xs, ys, s=10, alpha=0.5, label=cls)

    plt.title("Distance vs Points Inside Box")
    plt.xlabel("Box center distance (m)")
    plt.ylabel("Point count")
    plt.grid(alpha=0.2)
    if len(classes) <= 12:
        plt.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_distance_vs_points.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze points inside KITTI-format 3D boxes")
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data/dataset/KITTI/aug_scene/training",
        help="Path containing rectified_data/ and aug_label/",
    )
    parser.add_argument("--bin_dir", type=str, default="rectified_data", help="Point cloud folder under data_root")
    parser.add_argument("--label_dir", type=str, default="aug_label", help="Label folder under data_root")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/aug_scene_point_analysis",
        help="Directory for CSV and plots",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="Limit processed frames; 0 means all")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    bin_root = data_root / args.bin_dir
    label_root = data_root / args.label_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bin_root.exists():
        raise FileNotFoundError(f"Point cloud folder not found: {bin_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label folder not found: {label_root}")

    bin_files = sorted(bin_root.glob("*.bin"))
    if args.max_frames > 0:
        bin_files = bin_files[: args.max_frames]

    if not bin_files:
        raise RuntimeError(f"No .bin files found in {bin_root}")

    rows = []
    num_frames = 0
    num_boxes = 0

    for bin_file in bin_files:
        frame_id = bin_file.stem
        label_file = label_root / f"{frame_id}.txt"
        if not label_file.exists():
            continue

        points = np.fromfile(bin_file, dtype=np.float32)
        if points.size % 4 != 0:
            continue
        points = points.reshape(-1, 4)
        points_xyz = points[:, :3].astype(np.float64)

        objects = parse_kitti_label_file(str(label_file))
        if not objects:
            num_frames += 1
            continue

        for box_index, obj in enumerate(objects):
            x = obj["x"]
            y = obj["y"]
            z = obj["z"]
            h = max(obj["h"], 1e-6)
            w = max(obj["w"], 1e-6)
            l = max(obj["l"], 1e-6)
            ry = obj["ry"]

            mask = points_in_kitti_box_mask(points_xyz, x, y, z, h, w, l, ry)
            point_count = int(mask.sum())
            volume = float(h * w * l)
            density = float(point_count / volume)
            dist = float(np.sqrt(x * x + y * y + z * z))

            rows.append(
                {
                    "frame_id": frame_id,
                    "box_index": box_index,
                    "class_name": obj["class_name"],
                    "x": safe_float(x),
                    "y": safe_float(y),
                    "z": safe_float(z),
                    "h": safe_float(h),
                    "w": safe_float(w),
                    "l": safe_float(l),
                    "ry": safe_float(ry),
                    "distance": safe_float(dist),
                    "volume": safe_float(volume),
                    "point_count": point_count,
                    "density_pts_per_m3": safe_float(density),
                }
            )
            num_boxes += 1

        num_frames += 1

    if not rows:
        raise RuntimeError("No boxes were processed. Check paths and labels.")

    per_box_csv = out_dir / "per_box_points.csv"
    write_per_box_csv(rows, str(per_box_csv))

    # Class-wise summary
    classes = sorted({r["class_name"] for r in rows})
    class_summary = []
    for cls in classes:
        class_rows = [r for r in rows if r["class_name"] == cls]
        c = np.array([r["point_count"] for r in class_rows], dtype=np.float64)
        d = np.array([r["density_pts_per_m3"] for r in class_rows], dtype=np.float64)
        class_summary.append(
            {
                "class_name": cls,
                "num_boxes": int(c.size),
                "mean_points": safe_float(c.mean()),
                "p25_points": safe_float(np.percentile(c, 25)),
                "p50_points": safe_float(np.percentile(c, 50)),
                "p75_points": safe_float(np.percentile(c, 75)),
                "min_points": safe_float(c.min()),
                "max_points": safe_float(c.max()),
                "mean_density": safe_float(d.mean()),
                "median_density": safe_float(np.median(d)),
            }
        )

    class_csv = out_dir / "class_summary.csv"
    write_class_summary_csv(class_summary, str(class_csv))

    make_plots(rows, str(out_dir))

    # Console summary
    all_counts = np.array([r["point_count"] for r in rows], dtype=np.float64)
    print("=" * 80)
    print("Points-In-Boxes Analysis Complete")
    print("=" * 80)
    print(f"Processed frames: {num_frames}")
    print(f"Processed boxes : {num_boxes}")
    if args.max_frames > 0:
        print(f"Frame limit      : {args.max_frames}")
    print(f"Global mean     : {all_counts.mean():.2f}")
    print(f"Global p25/p50/p75 : {np.percentile(all_counts, 25):.2f} / {np.percentile(all_counts, 50):.2f} / {np.percentile(all_counts, 75):.2f}")
    print(f"Global min/max  : {all_counts.min():.0f} / {all_counts.max():.0f}")
    print("\nOutputs:")
    print(f"  - {per_box_csv}")
    print(f"  - {class_csv}")
    print(f"  - {out_dir / 'hist_points_per_box.png'}")
    print(f"  - {out_dir / 'boxplot_points_per_class.png'}")
    print(f"  - {out_dir / 'scatter_distance_vs_points.png'}")


if __name__ == "__main__":
    main()
