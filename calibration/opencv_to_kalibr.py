#!/usr/bin/env python
"""
Convert OpenCV stereo calibration parameters to Kalibr camchain.yaml format
"""

import numpy as np
import cv2 as cv
import yaml
import os

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [x, y, z, w]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return [x, y, z, w]

def convert_opencv_to_kalibr(xml_file, output_yaml):
    """Convert OpenCV calibration XML to Kalibr YAML format"""
    
    # Read the XML file
    cv_file = cv.FileStorage(xml_file, cv.FILE_STORAGE_READ)
    
    # Read camera matrices and distortion coefficients
    cameraMatrixL = cv_file.getNode('cameraMatrixL').mat()
    distCoeffsL = cv_file.getNode('distCoeffsL').mat().flatten()
    
    cameraMatrixR = cv_file.getNode('cameraMatrixR').mat()
    distCoeffsR = cv_file.getNode('distCoeffsR').mat().flatten()
    
    # Read rotation and translation
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat().flatten()
    
    # Read image size
    imageSize = cv_file.getNode('imageSize').mat().flatten().astype(int)
    
    cv_file.release()
    
    # Extract intrinsic parameters for cam0 (left)
    fx_L = cameraMatrixL[0, 0]
    fy_L = cameraMatrixL[1, 1]
    cx_L = cameraMatrixL[0, 2]
    cy_L = cameraMatrixL[1, 2]
    
    # Extract intrinsic parameters for cam1 (right)
    fx_R = cameraMatrixR[0, 0]
    fy_R = cameraMatrixR[1, 1]
    cx_R = cameraMatrixR[0, 2]
    cy_R = cameraMatrixR[1, 2]
    
    # Convert rotation matrix to quaternion
    q = rotation_matrix_to_quaternion(R)
    
    # Create the camchain dictionary in Kalibr format
    camchain = {
        'cam0': {
            'camera_model': 'pinhole',
            'intrinsics': [float(fx_L), float(fy_L), float(cx_L), float(cy_L)],
            'distortion_model': 'radtan',
            'distortion_coeffs': distCoeffsL[:4].tolist(),  # [k1, k2, p1, p2]
            'resolution': [int(imageSize[0]), int(imageSize[1])],
            'rostopic': '/cam0/image_raw'
        },
        'cam1': {
            'T_cn_cnm1': [
                [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(T[0])],
                [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(T[1])],
                [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(T[2])],
                [0.0, 0.0, 0.0, 1.0]
            ],
            'camera_model': 'pinhole',
            'intrinsics': [float(fx_R), float(fy_R), float(cx_R), float(cy_R)],
            'distortion_model': 'radtan',
            'distortion_coeffs': distCoeffsR[:4].tolist(),  # [k1, k2, p1, p2]
            'resolution': [int(imageSize[0]), int(imageSize[1])],
            'rostopic': '/cam1/image_raw'
        }
    }
    
    # Write to YAML file
    with open(output_yaml, 'w') as f:
        yaml.dump(camchain, f, default_flow_style=None, sort_keys=False)
    
    print(f"Conversion complete! Saved to {output_yaml}")
    print("\nCamera 0 (Left) Intrinsics:")
    print(f"  fx: {fx_L:.2f}, fy: {fy_L:.2f}")
    print(f"  cx: {cx_L:.2f}, cy: {cy_L:.2f}")
    print(f"  Distortion: {distCoeffsL[:4]}")
    
    print("\nCamera 1 (Right) Intrinsics:")
    print(f"  fx: {fx_R:.2f}, fy: {fy_R:.2f}")
    print(f"  cx: {cx_R:.2f}, cy: {cy_R:.2f}")
    print(f"  Distortion: {distCoeffsR[:4]}")
    
    print("\nBaseline (translation):")
    print(f"  T: {T}")
    print(f"  Baseline distance: {np.linalg.norm(T):.4f} meters (or mm if your calibration was in mm)")
    
    return camchain

if __name__ == "__main__":
    # Usage
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.join(script_dir, 'stereo_calibration_params_test.xml')
    output_yaml = os.path.join(script_dir, 'camchain-opencv_adjusted.yaml')
    
    convert_opencv_to_kalibr(xml_file, output_yaml)
