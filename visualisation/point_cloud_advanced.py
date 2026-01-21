#!/usr/bin/env python3
"""
ORB-SLAM3 Map + Trajectory viewer (Open3D)

Fixes the "tiny dot / can't zoom" issue by:
- clipping extreme outlier points (so the auto-fit view isn't ruined)
- optional uniform scaling for visualization
- setting an initial camera view based on your map extent
- keeping point size / line width configurable

Usage:
  pip install open3d numpy
  python3 point_cloud_gpt.py
"""

import numpy as np

# ======================= CONFIGURATION =======================

# Input Files
# You can edit these paths directly to point to your files
POINTS_FILE = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Experiment2/TEST2_MONO_MAP.txt"
TRAJ_FILE   = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Experiment2/TEST2_MONO_KEYFRAMES.txt"

# Cleaning / Noise Reduction
CLEAN_OUTLIERS  = True      # Set to True to remove "ghost" points (statistical outlier removal)
VOXEL_SIZE      = 0.02       # Downsample resolution (e.g. 0.02 = 2cm). 0.0 disables. Recommended ~0.02 if CLEAN_OUTLIERS is True.
CLIP_PERCENTILE = 99.5      # Clip furthest outliers (percentage to keep). Lower (e.g. 99.0) removes more distant noise.

# View Settings
INVERT_UP       = False     # Set to True if the initial view is upside down
AUTO_ALIGN_VIEW = True      # Automatically calculate the best initial view based on trajectory

# Visualization Aesthetics
VIS_SCALE       = 1.0       # Uniform scale (e.g. 50 or 100) if units are tiny. 1.0 keeps original units.
POINT_SIZE      = 3.0       # Size of points in the viewer
TRAJ_WIDTH      = 2.0       # Width of the trajectory line
BG_COLOR        = [0.05, 0.05, 0.05] # Background RGB (0.0 to 1.0)
NO_CENTER       = False     # If True, keeps original coordinates (good for multiple clouds). False centers everything.

# Camera Frustums
SHOW_CAM_FRUSTUM = True    # Show camera pyramids along path
FRUSTUM_SCALE    = 0.03     # Size of frustums
FRUSTUM_STEP     = 15       # Draw every Nth frame (to avoid clutter)
# =============================================================


# --------------------------- loaders ---------------------------

def load_xyz_txt(path: str) -> np.ndarray:
    """Loads whitespace-separated XYZ points (3 floats per line). Ignores blank/invalid lines."""
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not pts:
        raise ValueError(f"No valid XYZ points found in {path}")
    return np.asarray(pts, dtype=np.float64)


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix. Normalizes to avoid degenerate cases."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    x, y, z, w = q

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ], dtype=np.float64)


def parse_trajectory(path: str) -> np.ndarray:
    """
    Returns poses as Nx4x4 (world coordinates).
    Supports common ORB-SLAM formats:
      - t tx ty tz qx qy qz qw  (8 cols)
      - tx ty tz qx qy qz qw    (7 cols)
      - t r11..r33 tx ty tz     (13 cols)
      - r11..r33 tx ty tz       (12 cols)
    """
    poses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue

            M = np.eye(4, dtype=np.float64)

            if len(vals) == 8:
                _, tx, ty, tz, qx, qy, qz, qw = vals
                M[:3, :3] = quat_to_R(qx, qy, qz, qw)
                M[:3, 3] = [tx, ty, tz]
                poses.append(M)

            elif len(vals) == 7:
                tx, ty, tz, qx, qy, qz, qw = vals
                M[:3, :3] = quat_to_R(qx, qy, qz, qw)
                M[:3, 3] = [tx, ty, tz]
                poses.append(M)

            elif len(vals) == 13:
                _, *rest = vals
                r = rest[:9]
                t = rest[9:12]
                M[:3, :3] = np.array(r, dtype=np.float64).reshape(3, 3)
                M[:3, 3] = np.array(t, dtype=np.float64)
                poses.append(M)

            elif len(vals) == 12:
                r = vals[:9]
                t = vals[9:12]
                M[:3, :3] = np.array(r, dtype=np.float64).reshape(3, 3)
                M[:3, 3] = np.array(t, dtype=np.float64)
                poses.append(M)

    if not poses:
        raise ValueError(
            f"No valid poses parsed from {path}. "
            f"Paste 1–2 example lines and I'll adapt the parser."
        )
    return np.stack(poses, axis=0)


# --------------------------- geometry helpers ---------------------------

def clip_outliers_by_distance(pts: np.ndarray, percentile: float) -> np.ndarray:
    """Keep points whose distance-to-median is <= given percentile."""
    if percentile is None or not (0.0 < percentile < 100.0):
        return pts
    med = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - med, axis=1)
    cut = np.percentile(dist, percentile)
    return pts[dist <= cut]


def make_trajectory_lineset(points_xyz: np.ndarray):
    import open3d as o3d
    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_xyz),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return ls


def make_camera_frustum(scale=0.05):
    """Wireframe frustum in local frame, looking down +Z."""
    import open3d as o3d
    o = np.array([0, 0, 0], dtype=np.float64)
    z = 1.0 * scale
    x = 0.6 * scale
    y = 0.45 * scale
    c1 = np.array([-x, -y, z])
    c2 = np.array([ x, -y, z])
    c3 = np.array([ x,  y, z])
    c4 = np.array([-x,  y, z])

    V = np.vstack([o, c1, c2, c3, c4])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    fr = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(V),
        lines=o3d.utility.Vector2iVector(lines),
    )
    return fr


def transform_lineset(ls, T):
    import open3d as o3d
    ls2 = o3d.geometry.LineSet(ls)  # copy
    ls2.transform(T)
    return ls2


def extent_radius(pts: np.ndarray) -> float:
    """A robust 'scene radius' from points/trajectory."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ext = mx - mn
    r = float(np.linalg.norm(ext)) * 0.5
    return max(r, 1e-6)


def compute_initial_view(poses: np.ndarray, pts: np.ndarray, invert_up: bool):
    """
    Computes a smart initial camera view based on the trajectory and point cloud.
    Returns: (lookat, up, front, zoom)
    """
    # 1. LookAt: Center of the point cloud (robust median)
    if pts.shape[0] > 0:
        lookat = np.median(pts, axis=0)
    else:
        lookat = np.zeros(3)

    # 2. Up Vector: Average of camera 'up' vectors (-Y in camera frame)
    # Camera frame usually: X=Right, Y=Down, Z=Forward
    # So 'Up' in world is R * [0, -1, 0]
    cam_up_local = np.array([0, -1, 0], dtype=np.float64)
    up_accum = np.zeros(3, dtype=np.float64)
    
    # 3. Front Vector: Average of camera 'forward' vectors (+Z in camera frame)
    # So 'Forward' in world is R * [0, 0, 1]
    cam_fwd_local = np.array([0, 0, 1], dtype=np.float64)
    fwd_accum = np.zeros(3, dtype=np.float64)

    # Accumulate over all poses
    # poses shape is (N, 4, 4)
    for i in range(poses.shape[0]):
        R = poses[i, :3, :3]
        up_accum += R @ cam_up_local
        fwd_accum += R @ cam_fwd_local

    # Average and normalize
    if np.linalg.norm(up_accum) > 1e-6:
        up = up_accum / np.linalg.norm(up_accum)
    else:
        up = np.array([0, 1, 0]) # Fallback

    if np.linalg.norm(fwd_accum) > 1e-6:
        front = fwd_accum / np.linalg.norm(fwd_accum)
    else:
        front = np.array([0, 0, 1]) # Fallback

    if invert_up:
        up = -up

    vis_front = -front

    # 4. Zoom
    # Simple heuristic based on scene extent
    scene_radius = extent_radius(pts)
    zoom = 0.7 if scene_radius > 10 else 0.8
    
    return lookat, up, vis_front, zoom



# --------------------------- main app ---------------------------

def main():
    import open3d as o3d

    # Load
    print(f"Loading Points: {POINTS_FILE}")
    print(f"Loading Traj  : {TRAJ_FILE}")
    pts = load_xyz_txt(POINTS_FILE)
    poses = parse_trajectory(TRAJ_FILE)
    traj_xyz = poses[:, :3, 3]

    # Print quick scale diagnostics (helps debug mismatched units)
    p_ext = pts.max(axis=0) - pts.min(axis=0)
    t_ext = traj_xyz.max(axis=0) - traj_xyz.min(axis=0)
    print("PointCloud extent:", p_ext, "| radius~", extent_radius(pts))
    print("Trajectory extent:", t_ext, "| radius~", extent_radius(traj_xyz))

    # Clip outliers (THIS is usually the "tiny dot" fix)
    before = pts.shape[0]
    pts = clip_outliers_by_distance(pts, CLIP_PERCENTILE)
    after = pts.shape[0]
    if after != before:
        print(f"Clipped outliers: kept {after}/{before} points (percentile={CLIP_PERCENTILE})")

    # Random subsample if huge
    MAX_POINTS = 400000
    if pts.shape[0] > MAX_POINTS:
        idx = np.random.choice(pts.shape[0], size=MAX_POINTS, replace=False)
        pts = pts[idx]
        print(f"Subsampled points to {MAX_POINTS}")

    # Scale for visualization
    if VIS_SCALE != 1.0:
        pts = pts * VIS_SCALE
        traj_xyz = traj_xyz * VIS_SCALE
        poses = poses.copy()
        poses[:, :3, 3] *= VIS_SCALE
        print(f"Applied visualization scale x{VIS_SCALE}")

    # Build point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if VOXEL_SIZE and VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)
        print(f"Voxel downsampled: voxel={VOXEL_SIZE} -> points={np.asarray(pcd.points).shape[0]}")

    # Statistical Outlier Removal (Clean)
    if CLEAN_OUTLIERS:
        print("Running statistical outlier removal (this might take a moment)...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)
        # Update pts so the camera view calculation uses the cleaned cloud
        pts = np.asarray(pcd.points)
        print(f"Cleaned outliers: kept {pts.shape[0]} points")

    # Center scene (makes camera controls nicer)
    if not NO_CENTER:
        center = pcd.get_center()
        pcd.translate(-center)
        traj_xyz = traj_xyz - center
        poses = poses.copy()
        poses[:, :3, 3] = traj_xyz
    else:
        center = np.zeros(3, dtype=np.float64)

    # Trajectory lines
    traj_ls = make_trajectory_lineset(traj_xyz)

    # Colors
    pcd.paint_uniform_color([0.75, 0.75, 0.75])
    traj_ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.25, 0.25]]), (len(traj_xyz) - 1, 1))
    )

    geoms = [pcd, traj_ls]

    # Frustums
    if SHOW_CAM_FRUSTUM:
        base_fr = make_camera_frustum(scale=FRUSTUM_SCALE * VIS_SCALE)
        base_fr.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.25, 1.0, 0.25]]), (len(base_fr.lines), 1))
        )
        step = max(1, FRUSTUM_STEP)
        for i in range(0, poses.shape[0], step):
            geoms.append(transform_lineset(base_fr, poses[i]))

    # Axes sized to scene
    scene_r = max(extent_radius(np.asarray(pcd.points)), extent_radius(traj_xyz))
    axes_size = max(scene_r * 0.05, 0.05)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size)
    geoms.append(axes)

    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("ORB-SLAM3 Map + Trajectory", width=1400, height=850)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(POINT_SIZE)
    opt.background_color = np.asarray(BG_COLOR, dtype=np.float64)
    try:
        opt.line_width = float(TRAJ_WIDTH)
    except Exception:
        pass

    # Set a much better initial view (auto-aligned to trajectory)
    if AUTO_ALIGN_VIEW:
        vc = vis.get_view_control()
        
        lookat, up, front, zoom = compute_initial_view(poses, pts, INVERT_UP)
        
        vc.set_lookat(lookat)
        vc.set_up(up)
        vc.set_front(front)
        vc.set_zoom(zoom)

    print("\nControls: mouse drag=rotate, wheel=zoom, shift+drag=pan")
    print("Edit 'CONFIGURATION' at the top of this script to change settings (clean, scale, invert_up, etc).")
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
