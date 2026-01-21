#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# ===================== SETTINGS =====================
YAML_FILE   = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_KALIBR_RANA.yaml"
LEFT_GLOB   = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/720p_dataset/mav0/cam0/data/*.png"
RIGHT_GLOB  = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/720p_dataset/mav0/cam1/data/*.png"

NUM_LINES   = 20          # horizontal guide lines
SHOW_SCALE  = 0.5         # display scale

# --- rectification usage ---
USE_YAML_RECTIFICATION = False
# If False: compute R1,R2,P1,P2 via cv2.stereoRectify from K,D,R,T.
# If True : use LEFT.R/RIGHT.R/LEFT.P/RIGHT.P directly from YAML (if present).

# --- patch-match metric (recommended) ---
PATCH      = 11           # must be odd (e.g., 9/11/15)
Y_WINDOW   = 2            # search +- this many pixels vertically in right image
MAX_DISP   = 220          # max horizontal search (px). 200-300 is typical at 720p.
N_POINTS   = 400          # number of corners to test per frame
MIN_DISP   = 0            # ignore negative disparity matches
NCC_THRESH = 0.55         # reject weak matches (0..1, higher=more strict)

# Frame sampling
EVAL_EVERY_N = 1          # 1=all frames, 5=evaluate every 5th frame

# ====================================================

def read_orbslam3_yaml(yaml_path):
    print(f"Reading YAML: {yaml_path}")
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError("Could not open YAML file.")

    def r(name, default=None):
        node = fs.getNode(name)
        if node.empty():
            return default
        return node.real()

    def m(name):
        node = fs.getNode(name)
        if node.empty():
            return None
        return node.mat()

    params = {}

    # Intrinsics
    params["K1"] = np.eye(3, dtype=np.float64)
    params["K1"][0,0] = r("Camera1.fx")
    params["K1"][1,1] = r("Camera1.fy")
    params["K1"][0,2] = r("Camera1.cx")
    params["K1"][1,2] = r("Camera1.cy")

    params["K2"] = np.eye(3, dtype=np.float64)
    params["K2"][0,0] = r("Camera2.fx")
    params["K2"][1,1] = r("Camera2.fy")
    params["K2"][0,2] = r("Camera2.cx")
    params["K2"][1,2] = r("Camera2.cy")

    # Distortion (prefer 5 if present)
    k1 = r("Camera1.k1", 0.0); k2 = r("Camera1.k2", 0.0); p1 = r("Camera1.p1", 0.0); p2 = r("Camera1.p2", 0.0); k3 = r("Camera1.k3", 0.0)
    params["D1"] = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    k1 = r("Camera2.k1", 0.0); k2 = r("Camera2.k2", 0.0); p1 = r("Camera2.p1", 0.0); p2 = r("Camera2.p2", 0.0); k3 = r("Camera2.k3", 0.0)
    params["D2"] = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Resolution
    params["width"]  = int(r("Camera.width"))
    params["height"] = int(r("Camera.height"))

    # Extrinsics
    T_mat = m("Stereo.T_c1_c2")
    if T_mat is None:
        raise RuntimeError("Stereo.T_c1_c2 not found in YAML.")
    params["R"] = T_mat[:3,:3].astype(np.float64)
    params["T"] = T_mat[:3, 3].astype(np.float64).reshape(3,)

    # Optional rectification matrices from YAML
    params["LEFT_R"]  = m("LEFT.R")
    params["RIGHT_R"] = m("RIGHT.R")
    params["LEFT_P"]  = m("LEFT.P")
    params["RIGHT_P"] = m("RIGHT.P")

    fs.release()
    return params


def pair_by_filename(left_files, right_files):
    """Pair cam0/cam1 using basename without extension. Works for EuRoC timestamps.png naming."""
    lf = {os.path.splitext(os.path.basename(p))[0]: p for p in left_files}
    rf = {os.path.splitext(os.path.basename(p))[0]: p for p in right_files}
    keys = sorted(set(lf.keys()).intersection(set(rf.keys())))
    pairs = [(lf[k], rf[k], k) for k in keys]
    return pairs


def build_rectify_maps(params):
    img_size = (params["width"], params["height"])

    if USE_YAML_RECTIFICATION and (params["LEFT_R"] is not None) and (params["RIGHT_R"] is not None) and (params["LEFT_P"] is not None) and (params["RIGHT_P"] is not None):
        R1 = params["LEFT_R"].astype(np.float64)
        R2 = params["RIGHT_R"].astype(np.float64)
        P1 = params["LEFT_P"].astype(np.float64)
        P2 = params["RIGHT_P"].astype(np.float64)
        # initUndistortRectifyMap wants 3x3 newCameraMatrix; P is 3x4, so use left 3x3 block
        P1_3 = P1[:3,:3]
        P2_3 = P2[:3,:3]
    else:
        # Compute rectification from K,D,R,T
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            params["K1"], params["D1"],
            params["K2"], params["D2"],
            img_size,
            params["R"], params["T"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        P1_3 = P1[:3,:3]
        P2_3 = P2[:3,:3]

    map1x, map1y = cv2.initUndistortRectifyMap(params["K1"], params["D1"], R1, P1_3, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(params["K2"], params["D2"], R2, P2_3, img_size, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y


def ncc_patch(a, b):
    """Normalized cross correlation between 2 same-size patches (float32). Returns [-1..1]."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a*b).sum() / denom)


def rectification_metric_patch(rectL_gray, rectR_gray):
    """
    Row-constrained patch matching:
    - detect corners in left
    - search right along epipolar line (same row +- Y_WINDOW)
    - pick best NCC
    Returns (mean_dy, std_dy, n_matches)
    """
    h, w = rectL_gray.shape[:2]
    half = PATCH // 2

    # Corners in left
    corners = cv2.goodFeaturesToTrack(rectL_gray, maxCorners=N_POINTS, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return None

    corners = corners.reshape(-1, 2)

    dys = []
    for (x, y) in corners:
        x = int(round(x)); y = int(round(y))

        # left patch bounds
        if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
            continue

        patchL = rectL_gray[y-half:y+half+1, x-half:x+half+1].astype(np.float32)

        # search range in right: x' in [x-MAX_DISP, x-MIN_DISP]
        x_min = max(0, x - MAX_DISP)
        x_max = min(w - 1, x - MIN_DISP)

        if x_max - x_min < PATCH:
            continue

        best = -1.0
        best_xy = None

        for yy in range(max(half, y - Y_WINDOW), min(h - half, y + Y_WINDOW + 1)):
            for xx in range(x_min + half, x_max - half + 1):
                patchR = rectR_gray[yy-half:yy+half+1, xx-half:xx+half+1].astype(np.float32)
                score = ncc_patch(patchL, patchR)
                if score > best:
                    best = score
                    best_xy = (xx, yy)

        if best_xy is None or best < NCC_THRESH:
            continue

        dx = x - best_xy[0]
        if dx < MIN_DISP:
            continue

        dy = abs(y - best_xy[1])
        dys.append(dy)

    if len(dys) < 20:
        return None

    dys = np.array(dys, dtype=np.float64)
    return float(dys.mean()), float(dys.std()), int(len(dys))


def draw_lines(combined, num_lines):
    h, w = combined.shape[:2]
    gap = max(1, h // num_lines)
    for i in range(0, h, gap):
        cv2.line(combined, (0, i), (w, i), (0, 255, 0), 1)


def main():
    params = read_orbslam3_yaml(YAML_FILE)
    map1x, map1y, map2x, map2y = build_rectify_maps(params)

    left_files  = sorted(glob.glob(LEFT_GLOB))
    right_files = sorted(glob.glob(RIGHT_GLOB))
    if not left_files or not right_files:
        print("No images found! Check paths.")
        return

    pairs = pair_by_filename(left_files, right_files)
    if not pairs:
        print("Could not pair frames by filename. Make sure both folders use same timestamps.")
        return

    print(f"Paired {len(pairs)} frames by filename.")
    print("Controls: [Space/n] Next frame | [q] Quit\n")

    all_means = []
    all_stds  = []
    frames_eval = 0

    for idx, (l_path, r_path, key) in enumerate(pairs):
        if (idx % EVAL_EVERY_N) != 0:
            continue

        imgL = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue

        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

        res = rectification_metric_patch(rectL, rectR)
        if res is None:
            mean_err, std_err, nmatch = 999.0, 999.0, 0
        else:
            mean_err, std_err, nmatch = res
            all_means.append(mean_err)
            all_stds.append(std_err)
            frames_eval += 1

        combined = np.hstack((rectL, rectR))
        combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        draw_lines(combined, NUM_LINES)

        # verdict
        if mean_err < 0.5:
            verdict, col = "EXCELLENT", (0,255,0)
        elif mean_err < 1.0:
            verdict, col = "GOOD", (0,255,255)
        elif mean_err < 2.0:
            verdict, col = "OKAY", (0,165,255)
        else:
            verdict, col = "BAD/OFF", (0,0,255)

        txt = f"{os.path.basename(l_path)} | Matches={nmatch} | VertDisp: Mean={mean_err:.2f}px Std={std_err:.2f}px [{verdict}]"
        cv2.putText(combined, txt, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

        show = cv2.resize(combined, None, fx=SHOW_SCALE, fy=SHOW_SCALE)
        cv2.imshow("Rectification Check (row-constrained NCC)", show)

        k = cv2.waitKey(0)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\n=== Summary ===")
    print(f"Frames evaluated: {frames_eval}")
    if frames_eval == 0:
        print("No frames produced enough valid matches. Try lowering NCC_THRESH or increasing N_POINTS.")
        return

    mean_all = float(np.mean(all_means))
    std_all  = float(np.mean(all_stds))
    print(f"Mean vertical disparity: {mean_all:.3f} px")
    print(f"Std  vertical disparity: {std_all:.3f} px\n")

    print("Rule of thumb:")
    print("- ~0.0–0.5 px : excellent rectification")
    print("- ~0.5–1.0 px : good")
    print("- ~1–2 px     : usable but not great")
    print("- >2 px       : likely something off (pairing, wrong K/D, wrong R/T, or images already rectified twice)")


if __name__ == "__main__":
    main()
