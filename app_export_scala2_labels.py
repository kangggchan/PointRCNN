import argparse
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import torch

from lidar_det.detector import dccla


def load_kitti_bin(bin_path: Path) -> np.ndarray:
    points = np.fromfile(str(bin_path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(
            f"Invalid KITTI .bin file '{bin_path}': expected float32 x,y,z,intensity tuples."
        )
    return points.reshape(-1, 4)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DCCLA on KITTI-format .bin files and export detections to txt labels."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Dataset/bin"),
        help="Folder containing KITTI-format .bin files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("WarehouseData_labels"),
        help="Folder to save output txt labels when not updating an existing label folder.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("Dataset/label"),
        help=(
            "Existing label folder to update in place. Matching txt files must already exist; "
            "entries with --label-name are replaced/appended without creating new txt files."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("DCCLA_JRDB2022.pth"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="Human",
        help="Class name written at the beginning of each label line.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.7,
        help="Keep detections with score > threshold (minimum enforced: 0.7).",
    )
    return parser.parse_args()


def write_label_file(txt_path: Path, label_name: str, boxes: np.ndarray):
    lines = []
    for box in boxes:
        x, y, z, l, w, h, yaw = box.tolist()
        lines.append(
            f"{label_name} "
            f"{x:.4f} {y:.4f} {z:.4f} {l:.4f} {w:.4f} {h:.4f} {yaw:.4f}\n"
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def update_label_file_in_place(txt_path: Path, label_name: str, boxes: np.ndarray):
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {txt_path}. In-place updates do not create new txt files."
        )

    prefix = f"{label_name} "
    with txt_path.open("r", encoding="utf-8") as f:
        existing_lines = [line for line in f if not line.startswith(prefix)]

    for box in boxes:
        x, y, z, l, w, h, yaw = box.tolist()
        existing_lines.append(
            f"{label_name} "
            f"{x:.4f} {y:.4f} {z:.4f} {l:.4f} {w:.4f} {h:.4f} {yaw:.4f}\n"
        )

    with txt_path.open("w", encoding="utf-8") as f:
        f.writelines(existing_lines)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-only mode is enabled, but CUDA is not available in this environment."
        )

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {args.data_dir}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if args.label_dir is not None and not args.label_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {args.label_dir}")

    bin_files = sorted(args.data_dir.glob("*.bin"))
    if len(bin_files) == 0:
        raise RuntimeError(f"No .bin files found in: {args.data_dir}")

    if args.label_dir is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    effective_thresh = max(args.score_thresh, 0.7)
    detector = dccla(str(args.ckpt), gpu=True)

    total_kept = 0
    for i, bin_path in enumerate(bin_files, start=1):
        points_xyzi = load_kitti_bin(bin_path)
        points_for_model = points_xyzi[:, :3].astype(np.float32).T

        boxes, scores = cast(Tuple[np.ndarray, np.ndarray], detector(points_for_model))

        keep_mask = scores > effective_thresh
        boxes = boxes[keep_mask]

        if args.label_dir is None:
            txt_path = args.output_dir / f"{bin_path.stem}.txt"
            write_label_file(txt_path, args.label_name, boxes)
        else:
            txt_path = args.label_dir / f"{bin_path.stem}.txt"
            update_label_file_in_place(txt_path, args.label_name, boxes)

        total_kept += len(boxes)
        if i % 50 == 0 or i == len(bin_files):
            print(f"Processed {i}/{len(bin_files)} frames")

    print("Done. Device used: gpu")
    print(f"Input folder: {args.data_dir}")
    if args.label_dir is None:
        print(f"Output folder: {args.output_dir}")
    else:
        print(f"Updated label folder in place: {args.label_dir}")
    print(f"Score threshold: > {effective_thresh:.2f}")
    print(f"Label name: {args.label_name}")
    print(f"Total detections written: {total_kept}")


if __name__ == "__main__":
    main()
