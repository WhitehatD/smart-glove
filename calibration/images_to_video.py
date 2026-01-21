#!/usr/bin/env python3
"""
Images to Video Converter
-------------------------
Combines a sequence of images (PNG, JPG) into an MP4 video file.
"""

import cv2
import glob
import os
import sys

# ======================= CONFIGURATION =======================

# Input settings
# Folder containing the images
INPUT_FOLDER = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Experiment5/mav0/cam0/data" 
IMAGE_EXT    = "png"          # png, jpg, jpeg, etc.

# Output settings
OUTPUT_FILE  = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Experiment5/EXPERIMENT5_VIDEO.mp4"
FPS          = 30             # Frames Per Second (30 is standard real-time)

# advanced
SORT_BY_NAME = True           # True: sort by filename (timestamps), False: filesystem order
RESIZE_TO    = None           # None to keep original size, or tuple (width, height) e.g. (1280, 720)

# =============================================================

def make_video():
    # 1. Find images
    search_path = os.path.join(INPUT_FOLDER, f"*.{IMAGE_EXT}")
    print(f"Searching for images in: {search_path}")
    
    files = glob.glob(search_path)
    if not files:
        print("No images found! Check the path and extension.")
        return

    # 2. Sort
    if SORT_BY_NAME:
        files.sort()
    else:
        files.sort(key=os.path.getmtime)
        
    count = len(files)
    print(f"Found {count} images.")
    
    # 3. Read first frame to get size
    first_img = cv2.imread(files[0])
    if first_img is None:
        print(f"Error reading first image: {files[0]}")
        return
        
    height, width, layers = first_img.shape
    size = (width, height)
    print(f"Resolution: {width}x{height}")

    if RESIZE_TO:
        size = RESIZE_TO
        print(f"Resizing to: {size}")

    # 4. Initialize Video Writer
    # 'mp4v' is a solid default for .mp4 on Mac/Linux. 
    # 'avc1' (H.264) is better but sometimes requires extra codec installation.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, size)
    
    print(f"Writing video to: {OUTPUT_FILE}")
    print("Processing...")

    # 5. Write frames
    for i, filename in enumerate(files):
        img = cv2.imread(filename)
        
        if img is None:
            print(f"Skipping bad file: {filename}")
            continue
            
        if RESIZE_TO:
            img = cv2.resize(img, RESIZE_TO)
            
        out.write(img)
        
        # Progress bar
        if i % 50 == 0:
            sys.stdout.write(f"\rFrame {i+1}/{count} ({(i+1)/count*100:.1f}%)")
            sys.stdout.flush()

    out.release()
    print(f"\n\nDone! Saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    make_video()
