import cv2
import numpy as np
import os
import glob

# 1. VALUES FROM YOUR NEW XML
size = (1280, 720)

# Camera Matrix L
K1 = np.array([
    [1092.5954, 0, 657.0905],
    [0, 1083.3805, 448.0608],
    [0, 0, 1]
])
# distCoeffsL: [k1, k2, p1, p2, k3]
D1 = np.array([0.051402, -0.578053, 0.006933, -0.007027, 1.032889])

# Camera Matrix R
K2 = np.array([
    [1107.0807, 0, 665.8274],
    [0, 1095.3423, 463.6532],
    [0, 0, 1]
])
# distCoeffsR: [k1, k2, p1, p2, k3]
D2 = np.array([0.050311, -0.700774, 0.009139, -0.005651, 1.483197])

# Rotation Matrix R from XML
R = np.array([
    [0.999928, -0.005648, -0.010537],
    [0.005932, 0.999614, 0.027119],
    [0.010380, -0.027180, 0.999576]
])

# Translation T from XML (Converted to mm to match OpenCV expectation if R/T is Cam2->Cam1)
# Note: stereoRectify usually expects T in the same units as your desired baseline
T = np.array([[-75.0748], [3.8073], [7.9491]])

# 2. COMPUTE RECTIFICATION
# alpha=0 means images are zoomed so that only valid pixels are visible (no black borders)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)

def process_folder(input_dir, output_dir, mx, my):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Rectifying {len(images)} images -> {output_dir}")
    for img_path in images:
        img = cv2.imread(img_path)
        rectified = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), rectified)

# 3. PATHS (Update these to your Mac paths)
left_in = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/720p_dataset/mav0/cam0/data"
right_in = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/720p_dataset/mav0/cam1/data"

left_out = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/7200/mav0/cam0/data"
right_out = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/7200/mav0/cam1/data"

process_folder(left_in, left_out, map1x, map1y)
process_folder(right_in, right_out, map2x, map2y)