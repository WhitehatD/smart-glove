#!/usr/bin/env python3
"""
kalibr_to_orbslam3_no_flags.py

Convert a Kalibr stereo calibration YAML to an ORB-SLAM3 stereo YAML (with online rectification)
WITHOUT any CLI flags. Edit the CONFIG section only.

Requirements:
  pip install pyyaml opencv-python numpy

What it does:
- Reads Kalibr YAML (cam0/cam1 intrinsics + distortion + cam1.T_cn_cnm1)
- Assumes Kalibr translation is in METERS (typical)
- Computes rectification (R1,R2,P1,P2,Q) via cv2.stereoRectify
- Writes an ORB-SLAM3 YAML that ORB-SLAM3 can parse + rectify online

If rectification is "wrong"/tracking is bad:
- Set INVERT_T = True (Kalibr transform direction mismatch is the most common issue)
- Or set ZERO_DISPARITY = True/False (usually True is fine)
"""

import os
import numpy as np
import cv2
import yaml

# ----------------------- CONFIG (EDIT THESE) -----------------------

KALIBR_YAML_PATH = "camchain-stereo_calib.yaml"     # <-- your Kalibr file
OUTPUT_ORBSLAM3_YAML = "STEREO_LAST.yaml"

FPS = 30
RGB = 1  # 0=BGR, 1=RGB (set to 0 if you feed BGR images)

# stereoRectify options
ALPHA = 0.0               # 0=crop to valid pixels, 1=keep all (black borders)
ZERO_DISPARITY = True     # recommended: aligns principal points

# Kalibr transform direction:
# Kalibr cam1.T_cn_cnm1 is commonly "T_cam1_cam0" (cam1 <- cam0) but this can differ.
# If your result is bad, flip this to True.
INVERT_T = False

# YAML matrix datatype: 'f' float32 recommended for ORB-SLAM3
DT = "f"  # "f" or "d"

# ORB parameters (safe defaults)
ORB_NFEATURES = 2000
ORB_INI_TH_FAST = 12
ORB_MIN_TH_FAST = 7

# Include Viewer parameters (often required by ORB-SLAM3 example binaries)
INCLUDE_VIEWER_PARAMS = True

# ----------------------- END CONFIG -----------------------


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def opencv_matrix_yaml(name, M, dt="f"):
    if dt == "f":
        M = np.asarray(M, dtype=np.float32)
    else:
        M = np.asarray(M, dtype=np.float64)
    rows, cols = M.shape
    data = ", ".join(f"{float(v):.10f}" for v in M.reshape(-1))
    return (
        f"{name}: !!opencv-matrix\n"
        f"   rows: {rows}\n"
        f"   cols: {cols}\n"
        f"   dt: {dt}\n"
        f"   data: [{data}]\n"
    )


def main():
    if not os.path.exists(KALIBR_YAML_PATH):
        raise FileNotFoundError(f"Kalibr YAML not found: {KALIBR_YAML_PATH}")

    with open(KALIBR_YAML_PATH, "r", encoding="utf-8") as f:
        k = yaml.safe_load(f)

    if "cam0" not in k or "cam1" not in k:
        raise ValueError("Kalibr YAML must contain top-level keys: cam0 and cam1")

    cam0 = k["cam0"]
    cam1 = k["cam1"]

    # -------- intrinsics --------
    fx0, fy0, cx0, cy0 = [float(x) for x in cam0["intrinsics"]]
    fx1, fy1, cx1, cy1 = [float(x) for x in cam1["intrinsics"]]

    K0 = np.array([[fx0, 0.0, cx0],
                   [0.0, fy0, cy0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    K1 = np.array([[fx1, 0.0, cx1],
                   [0.0, fy1, cy1],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    # -------- distortion (radtan -> OpenCV: k1 k2 p1 p2 k3) --------
    d0 = [float(x) for x in cam0.get("distortion_coeffs", [])]
    d1 = [float(x) for x in cam1.get("distortion_coeffs", [])]

    def pad_dist(d):
        # Kalibr radtan typically: [k1, k2, p1, p2] (k3 not provided)
        if len(d) == 4:
            return np.array([d[0], d[1], d[2], d[3], 0.0], dtype=np.float64)
        if len(d) >= 5:
            return np.array(d[:5], dtype=np.float64)
        raise ValueError(f"Unexpected distortion length: {len(d)} (need 4 or >=5)")

    D0 = pad_dist(d0).reshape(1, -1)
    D1 = pad_dist(d1).reshape(1, -1)

    # -------- resolution --------
    w0, h0 = [int(x) for x in cam0["resolution"]]
    w1, h1 = [int(x) for x in cam1["resolution"]]
    if (w0, h0) != (w1, h1):
        raise ValueError(f"cam0 resolution {w0,h0} != cam1 resolution {w1,h1}")
    img_size = (w0, h0)

    # -------- extrinsics --------
    if "T_cn_cnm1" not in cam1:
        raise ValueError("Kalibr cam1 must contain T_cn_cnm1 (4x4)")

    T = np.array(cam1["T_cn_cnm1"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cn_cnm1 must be 4x4, got {T.shape}")

    if INVERT_T:
        T = invert_T(T)

    R = T[:3, :3].copy()
    t = T[:3, 3].copy()  # assumed meters

    # -------- stereoRectify --------
    flags = cv2.CALIB_ZERO_DISPARITY if ZERO_DISPARITY else 0

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K0, D0, K1, D1,
        img_size,
        R, t,
        flags=flags,
        alpha=float(ALPHA),
    )

    # -------- baseline / bf --------
    baseline = float(abs(t[0]))  # use x-baseline for stereo bf
    bf = baseline * fx0
    th_depth = 40.0 * baseline

    # ORB-SLAM3 Stereo.T_c1_c2
    T_c1_c2 = np.eye(4, dtype=np.float64)
    T_c1_c2[:3, :3] = R
    T_c1_c2[:3, 3] = t

    # -------- write YAML --------
    out = []
    out.append("%YAML:1.0\n")
    out.append("# ORB-SLAM3 Stereo Config (converted from Kalibr)\n")
    out.append('File.version: "1.0"\n\n')

    out.append('Camera.type: "PinHole"\n\n')

    out.append(f"Camera1.fx: {fx0:.10f}\n")
    out.append(f"Camera1.fy: {fy0:.10f}\n")
    out.append(f"Camera1.cx: {cx0:.10f}\n")
    out.append(f"Camera1.cy: {cy0:.10f}\n")
    out.append(f"Camera1.k1: {float(D0[0,0]):.10f}\n")
    out.append(f"Camera1.k2: {float(D0[0,1]):.10f}\n")
    out.append(f"Camera1.p1: {float(D0[0,2]):.10f}\n")
    out.append(f"Camera1.p2: {float(D0[0,3]):.10f}\n")
    out.append(f"Camera1.k3: {float(D0[0,4]):.10f}\n\n")

    out.append(f"Camera2.fx: {fx1:.10f}\n")
    out.append(f"Camera2.fy: {fy1:.10f}\n")
    out.append(f"Camera2.cx: {cx1:.10f}\n")
    out.append(f"Camera2.cy: {cy1:.10f}\n")
    out.append(f"Camera2.k1: {float(D1[0,0]):.10f}\n")
    out.append(f"Camera2.k2: {float(D1[0,1]):.10f}\n")
    out.append(f"Camera2.p1: {float(D1[0,2]):.10f}\n")
    out.append(f"Camera2.p2: {float(D1[0,3]):.10f}\n")
    out.append(f"Camera2.k3: {float(D1[0,4]):.10f}\n\n")

    out.append(f"Camera.width: {img_size[0]}\n")
    out.append(f"Camera.height: {img_size[1]}\n")
    out.append(f"Camera.fps: {FPS:.2f}\n")
    out.append(f"Camera.RGB: {int(RGB)}\n\n")

    out.append("# Transformation from Camera 1 (cam0/left) to Camera 2 (cam1/right)\n")
    out.append(opencv_matrix_yaml("Stereo.T_c1_c2", T_c1_c2, dt=DT))
    out.append("\n")

    out.append(f"Stereo.b: {baseline:.10f}\n")
    out.append(f"Stereo.bf: {bf:.10f}\n")
    out.append(f"Stereo.ThDepth: {th_depth:.10f}\n\n")

    out.append("# Stereo Rectification (for online rectification)\n")
    out.append(opencv_matrix_yaml("LEFT.K", K0, dt=DT))
    out.append(opencv_matrix_yaml("RIGHT.K", K1, dt=DT))
    out.append(opencv_matrix_yaml("LEFT.D", D0, dt=DT))
    out.append(opencv_matrix_yaml("RIGHT.D", D1, dt=DT))
    out.append(opencv_matrix_yaml("LEFT.R", R1, dt=DT))
    out.append(opencv_matrix_yaml("RIGHT.R", R2, dt=DT))
    out.append(opencv_matrix_yaml("LEFT.P", P1, dt=DT))
    out.append(opencv_matrix_yaml("RIGHT.P", P2, dt=DT))
    out.append(opencv_matrix_yaml("Q", Q, dt=DT))
    out.append("\n")

    out.append(f"ORBextractor.nFeatures: {ORB_NFEATURES}\n")
    out.append("ORBextractor.scaleFactor: 1.2\n")
    out.append("ORBextractor.nLevels: 8\n")
    out.append(f"ORBextractor.iniThFAST: {ORB_INI_TH_FAST}\n")
    out.append(f"ORBextractor.minThFAST: {ORB_MIN_TH_FAST}\n\n")

    if INCLUDE_VIEWER_PARAMS:
        out.append("# Viewer Parameters\n")
        out.append("Viewer.KeyFrameSize: 0.05\n")
        out.append("Viewer.KeyFrameLineWidth: 1.0\n")
        out.append("Viewer.GraphLineWidth: 0.9\n")
        out.append("Viewer.PointSize: 2.0\n")
        out.append("Viewer.CameraSize: 0.08\n")
        out.append("Viewer.CameraLineWidth: 3.0\n")
        out.append("Viewer.ViewpointX: 0.0\n")
        out.append("Viewer.ViewpointY: -0.7\n")
        out.append("Viewer.ViewpointZ: -1.8\n")
        out.append("Viewer.ViewpointF: 500.0\n")

    with open(OUTPUT_ORBSLAM3_YAML, "w", encoding="utf-8") as f:
        f.write("".join(out))

    print(f"[OK] Wrote: {OUTPUT_ORBSLAM3_YAML}")
    print(f"     Resolution: {img_size[0]}x{img_size[1]}")
    print(f"     Baseline (m): {baseline:.6f}")
    print(f"     bf: {bf:.3f}")
    print(f"     ThDepth (m): {th_depth:.3f}")
    print("\nIf results look wrong, the usual fix is: set INVERT_T = True at the top and rerun.")


if __name__ == "__main__":
    main()
