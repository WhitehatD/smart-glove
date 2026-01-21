#!/usr/bin/env python3
"""
generate_timestamps.py

Scans a folder of images (e.g., 1234567.png), extracts the filename (timestamp),
and saves them to a sorted text file (timestamps.txt).

Useful for ORB-SLAM3 which requires a list of timestamps for the dataset.
"""

import os
import glob

# ======================= CONFIGURATION =======================
# Folder containing the images
INPUT_FOLDER = "/Users/batigozen/Desktop/my_dataset/mav0/cam0/data"

# Output text file
OUTPUT_FILE = "timestamps.txt"

# Extension to look for
EXT = "png"

# Save only the name (without extension)?
# True:  "123456.png" -> writes "123456" (Standard for ORB-SLAM3)
# False: "123456.png" -> writes "123456.png"
STRIP_EXTENSION = True
# =============================================================

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder not found: {INPUT_FOLDER}")
        return

    search_pattern = os.path.join(INPUT_FOLDER, f"*.{EXT}")
    print(f"Scanning: {search_pattern}")
    
    files = glob.glob(search_pattern)
    files.sort() # Ensure sorted order
    
    if not files:
        print("No images found.")
        return

    print(f"Found {len(files)} images. Writing to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w") as f:
        for filepath in files:
            filename = os.path.basename(filepath)
            
            if STRIP_EXTENSION:
                # Remove the last extension (e.g. .png)
                name_val = os.path.splitext(filename)[0]
                f.write(name_val + "\n")
            else:
                f.write(filename + "\n")

    print(f"Done! Saved {len(files)} timestamps to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
