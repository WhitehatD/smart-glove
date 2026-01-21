#!/usr/bin/env python3

"""
Fix ORB-SLAM3 timestamps.txt format.

Input format (current):
  timestamp filename.png

Output format (EuRoC / ORB-SLAM3):
  timestamp
"""

import sys

INPUT_FILE = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Session_20260121_181308/timestamps.txt"
OUTPUT_FILE = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS/Session_20260121_181308/timestamps_fixed.txt"

def main():
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    timestamps = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ts = line.split()[0]   # take only first column
        timestamps.append(ts)

    with open(OUTPUT_FILE, "w") as f:
        for ts in timestamps:
            f.write(ts + "\n")

    print(f"[OK] Fixed timestamps written to: {OUTPUT_FILE}")
    print(f"Lines: {len(timestamps)}")

if __name__ == "__main__":
    main()
