"""
Playback script: visualize pre-generated label files alongside their point clouds in Open3D.

Label format (one detection per line):
    <class_name> <x> <y> <z> <l> <w> <h> <yaw>

Controls
--------
  N  /  →   : next frame
  P  /  ←   : previous frame
  Space      : toggle auto-play
  Q  / Esc   : quit
"""
import argparse
import time
from pathlib import Path

import numpy as np

BOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # top face
    [0, 4], [1, 5], [2, 6], [3, 7],  # vertical pillars
]

# ── colour per class name (RGB 0-1, falls back to white) ─────────────────────
CLASS_COLORS = {
    "human": [0.0, 1.0, 0.0],
    "car":   [1.0, 0.5, 0.0],
    "cyclist": [0.0, 0.5, 1.0],
}


# ── data loading helpers ──────────────────────────────────────────────────────

def load_bin(bin_path: Path) -> np.ndarray:
    pts = np.fromfile(str(bin_path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Unexpected size in {bin_path}")
    return pts.reshape(-1, 4)[:, :3]  # xyz only


def load_labels(txt_path: Path):
    """Return (boxes, class_names) where boxes is (N,7) float32 [x y z l w h yaw]."""
    boxes, names = [], []
    if not txt_path.exists():
        return np.zeros((0, 7), dtype=np.float32), []
    with txt_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            names.append(parts[0])
            boxes.append([float(v) for v in parts[1:8]])
    if boxes:
        return np.array(boxes, dtype=np.float32), names
    return np.zeros((0, 7), dtype=np.float32), []


# ── Open3D geometry builders ─────────────────────────────────────────────────

def _pcd_geometry(points_xyz: np.ndarray):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    ranges = np.linalg.norm(points_xyz[:, :2], axis=1)
    t = np.clip(ranges / 40.0, 0.0, 1.0)
    colors = np.stack([0.2 + 0.8 * t, 0.8 - 0.6 * t, 1.0 - 0.8 * t], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _box_corners(box: np.ndarray) -> np.ndarray:
    """Return (8, 3) corners for a box [x, y, z, l, w, h, yaw]."""
    x, y, z, l, w, h, yaw = box
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    # local half-extents
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    # 8 corners in local frame (front-right-top ordering)
    local = np.array([
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
    ])
    rot = np.array([[cos_y, -sin_y, 0],
                    [sin_y,  cos_y, 0],
                    [0,      0,     1]])
    return (rot @ local.T).T + np.array([x, y, z])


def _lineset_geometry(corners8: np.ndarray, color):
    import open3d as o3d
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners8)
    ls.lines  = o3d.utility.Vector2iVector(BOX_EDGES)
    ls.colors = o3d.utility.Vector3dVector([color] * len(BOX_EDGES))
    return ls


def build_geometries(points_xyz: np.ndarray, boxes: np.ndarray, class_names: list):
    geoms = [_pcd_geometry(points_xyz)]
    for box, name in zip(boxes, class_names):
        color = CLASS_COLORS.get(name.lower(), [1.0, 1.0, 1.0])
        corners = _box_corners(box)
        geoms.append(_lineset_geometry(corners, color))
    return geoms


# ── playback logic ────────────────────────────────────────────────────────────

def run_playback(
    bin_files: list,
    label_files: list,
    fps: float,
    start_frame: int,
):
    import open3d as o3d

    state = {
        "idx": start_frame,
        "playing": False,
        "last_t": time.time(),
        "quit": False,
    }
    n_frames = len(bin_files)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="DCCLA Label Playback", width=1400, height=900)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size = 1.5

    # ── helpers ───────────────────────────────────────────────────────────────
    loaded_geoms = []

    def load_frame(idx):
        nonlocal loaded_geoms
        pts = load_bin(bin_files[idx])
        boxes, names = load_labels(label_files[idx])
        geoms = build_geometries(pts, boxes, names)

        for g in loaded_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        loaded_geoms = geoms
        reset = idx == start_frame
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)

        vis.get_view_control()          # keep current viewpoint after first frame
        vis.update_renderer()
        vis.poll_events()

        label_path = label_files[idx]
        _, names_loaded = load_labels(label_path)
        n_det = len(names_loaded)
        vis.get_render_option()         # touch to force title refresh
        # Open3D doesn't expose set_window_title; print to terminal instead
        print(
            f"  Frame {idx+1:4d}/{n_frames}  |  "
            f"{bin_files[idx].name}  |  detections: {n_det}"
        )

    def go_next(_vis):
        state["idx"] = (state["idx"] + 1) % n_frames
        load_frame(state["idx"])

    def go_prev(_vis):
        state["idx"] = (state["idx"] - 1) % n_frames
        load_frame(state["idx"])

    def toggle_play(_vis):
        state["playing"] = not state["playing"]
        print("  Auto-play:", "ON" if state["playing"] else "OFF")

    def quit_cb(_vis):
        state["quit"] = True
        vis.close()

    # ── key bindings ──────────────────────────────────────────────────────────
    # Open3D key codes: printable ASCII = ord(char); arrow keys via GLFW codes
    vis.register_key_callback(ord("N"), go_next)
    vis.register_key_callback(ord("P"), go_prev)
    vis.register_key_callback(262, go_next)    # GLFW_KEY_RIGHT
    vis.register_key_callback(263, go_prev)    # GLFW_KEY_LEFT
    vis.register_key_callback(32,  toggle_play)  # Space
    vis.register_key_callback(ord("Q"), quit_cb)
    vis.register_key_callback(256, quit_cb)    # Escape

    # ── initial frame ─────────────────────────────────────────────────────────
    print("\nControls:  N/→ next  |  P/← prev  |  Space toggle play  |  Q/Esc quit\n")
    load_frame(state["idx"])

    interval = 1.0 / fps

    while vis.poll_events():
        if state["quit"]:
            break
        if state["playing"]:
            now = time.time()
            if now - state["last_t"] >= interval:
                state["last_t"] = now
                state["idx"] = (state["idx"] + 1) % n_frames
                load_frame(state["idx"])
        vis.update_renderer()

    vis.destroy_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Playback DCCLA label files with their point clouds in Open3D."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Dataset/bin"),
        help="Folder with KITTI-format .bin files (default: scala2_data).",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("Dataset/label"),
        help="Folder with .txt label files (default: scala2_data_labels).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Auto-play frame rate (default: 10 fps).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Frame index to start at (0-based, default: 0).",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start in auto-play mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    bin_files  = sorted(args.data_dir.glob("*.bin"))
    label_files = sorted(args.label_dir.glob("*.txt"))

    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {args.data_dir}")
    if not label_files:
        raise FileNotFoundError(f"No .txt label files found in {args.label_dir}")

    if len(bin_files) != len(label_files):
        print(
            f"Warning: {len(bin_files)} .bin files vs {len(label_files)} label files. "
            "Using the shorter count."
        )
    n = min(len(bin_files), len(label_files))
    bin_files   = bin_files[:n]
    label_files = label_files[:n]

    start = max(0, min(args.start, n - 1))
    print(f"Found {n} frames  ({args.data_dir}  +  {args.label_dir})")

    run_playback(bin_files, label_files, fps=args.fps, start_frame=start)


if __name__ == "__main__":
    main()
