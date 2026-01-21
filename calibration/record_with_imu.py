#!/usr/bin/env python3
"""
Raspberry Pi EuRoC-style recorder for ORB-SLAM3 Mono-Inertial (EuRoC runner).

Outputs:
  <DATASET_PATH>/
    mav0/
      cam0/
        data/              # <timestamp>.png (grayscale)
        data.csv           # EuRoC-style: timestamp [ns],filename
      imu0/
        data.csv           # EuRoC-style: timestamp [ns],wx,wy,wz,ax,ay,az
    MH01.txt               # ORB-SLAM3-style timestamps list (one timestamp per line)

Notes:
- Timestamps are relative to start (t0) in nanoseconds, monotonic increasing.
- IMU is requested at 200 Hz via report_interval=5 ms (Adafruit BNO08x driver).
- Camera frames are timestamped when read from OpenCV (arrival time, not exposure time).
"""

import os
import sys
import time
import threading
import queue
import signal

import cv2
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C as BNO080_I2C
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE
# ===================== CONFIGURATION =====================
DATASET_PATH = "my_pi_bno080_dataset"

CAM_INDEX = 0
RESOLUTION = (640, 480)
CAM_FPS = 30                # best-effort; some cameras ignore this
FOURCC = "MJPG"             # good for USB cams; adjust if needed

IMU_RATE_HZ = 200
IMU_REPORT_INTERVAL_MS = int(1000 / IMU_RATE_HZ)  # 5ms @ 200Hz

I2C_FREQUENCY_HZ = 400_000  # 400kHz often helps on Pi

IMG_QUEUE_MAX = 200         # prevent RAM blow-up if disk is slow

WRITE_PNG_COMPRESSION = 3   # 0..9, lower=faster/larger
# =========================================================


def ensure_dirs(base: str) -> dict:
    cam_data = os.path.join(base, "mav0", "cam0", "data")
    imu_dir = os.path.join(base, "mav0", "imu0")

    os.makedirs(cam_data, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    return {
        "cam_data": cam_data,
        "cam_csv": os.path.join(base, "mav0", "cam0", "data.csv"),
        "imu_csv": os.path.join(base, "mav0", "imu0", "data.csv"),
        "mh_txt": os.path.join(base, "MH01.txt"),
    }

class PiEuRoCRecorderBNO080:
    def __init__(self):
        # 1. First, define the paths and create the folders
        self.paths = ensure_dirs(DATASET_PATH)

        # 2. Then, do your debug prints and IMU setup
        print("DEBUG: Initializing I2C...")
        i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY_HZ)

        print("DEBUG: Finding BNO080...")
        self.bno = BNO080_I2C(i2c)

        print("DEBUG: BNO080 Found!")

        # 3. Now the rest of the variables
        self.running = True
        self.t0_ns = time.time_ns()

        # ... rest of your code (opening files, setting up camera, etc.)
        # Bounded queue so we don’t OOM if disk can’t keep up.
        self.img_queue: "queue.Queue[tuple[int, any]]" = queue.Queue(maxsize=IMG_QUEUE_MAX)

        # --- Open files (line-buffered for safety) ---
        self.f_cam_csv = open(self.paths["cam_csv"], "w", buffering=1)
        # EuRoC camera data.csv header comment is common (not required but nice)
        self.f_cam_csv.write("#timestamp [ns],filename\n")

        self.f_mh = open(self.paths["mh_txt"], "w", buffering=1)

        self.f_imu = open(self.paths["imu_csv"], "w", buffering=1)
        self.f_imu.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
        )

        # --- Setup IMU (I2C) ---
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=I2C_FREQUENCY_HZ)
            self.bno = BNO080_I2C(i2c)

            # Request 200Hz for both accel + gyro
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER, report_interval=IMU_REPORT_INTERVAL_MS)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE, report_interval=IMU_REPORT_INTERVAL_MS)

        except Exception as e:
            self.close()
            raise RuntimeError(
                f"CRITICAL: Could not init BNO080 over I2C: {e}\n"
                "Check wiring: SDA->GPIO2, SCL->GPIO3, 3.3V power, common ground."
            )

        # --- Setup camera ---
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            self.close()
            raise RuntimeError(f"CRITICAL: Camera index {CAM_INDEX} not found/openable.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))

        # --- Worker threads ---
        self.t_save = threading.Thread(target=self._saver_worker, name="img_saver", daemon=True)
        self.t_imu = threading.Thread(target=self._imu_worker, name="imu_logger", daemon=True)
        self.t_save.start()
        self.t_imu.start()

        print("===============================================")
        print("Pi EuRoC Recorder (Mono-Inertial) starting...")
        print(f"Dataset path: {os.path.abspath(DATASET_PATH)}")
        print(f"Camera: idx={CAM_INDEX}, res={RESOLUTION}, fps~{CAM_FPS}, fourcc={FOURCC}")
        print(f"IMU: BNO080 @ {IMU_RATE_HZ}Hz (report_interval={IMU_REPORT_INTERVAL_MS}ms), I2C={I2C_FREQUENCY_HZ}Hz")
        print("Stop: Ctrl+C")
        print("===============================================")

    def now_rel_ns(self) -> int:
        return time.time_ns() - self.t0_ns

    def _imu_worker(self):
        """
        Log IMU at ~200Hz.
        We try to read new values each tick. If the driver returns duplicates,
        ORB-SLAM3 typically still runs, but better wiring/driver timing helps.
        """
        tick_ns = int(1e9 / IMU_RATE_HZ)
        next_tick = time.perf_counter_ns()

        last_row = None  # used to suppress exact duplicates
        while self.running:
            next_tick += tick_ns
            try:
                gyro = self.bno.gyro          # (x, y, z) rad/s
                accel = self.bno.acceleration # (x, y, z) m/s^2

                if gyro is not None and accel is not None:
                    ts = self.now_rel_ns()
                    gx, gy, gz = gyro
                    ax, ay, az = accel

                    row = (gx, gy, gz, ax, ay, az)
                    # Suppress exact duplicates to reduce repeated samples if polling outruns updates
                    if row != last_row:
                        self.f_imu.write(f"{ts},{gx},{gy},{gz},{ax},{ay},{az}\n")
                        last_row = row

            except Exception:
                # occasional I2C read hiccups can happen; skip this sample
                pass

            # Sleep until next tick (best-effort)
            remaining = next_tick - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
            else:
                # We're late; don't sleep (catch up)
                next_tick = time.perf_counter_ns()

    def _saver_worker(self):
        cam_dir = self.paths["cam_data"]
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(WRITE_PNG_COMPRESSION)]

        while self.running or not self.img_queue.empty():
            try:
                ts, gray = self.img_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filename = f"{ts}.png"
            filepath = os.path.join(cam_dir, filename)

            # Write image
            try:
                cv2.imwrite(filepath, gray, png_params)
            except Exception:
                # If disk write fails, we still log timestamps for debugging
                pass

            # EuRoC cam0 data.csv wants: timestamp, filename
            self.f_cam_csv.write(f"{ts},{filename}\n")

            # ORB-SLAM3 EuRoC runner uses MHxx.txt timestamps list (one per line)
            self.f_mh.write(f"{ts}\n")

    def run(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                ts = self.now_rel_ns()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Show live preview (for debugging & alignment)
                try:
                    self.img_queue.put((ts, gray), timeout=0.01)
                except queue.Full:
                    continue

        finally:
            self.stop()

    def stop(self):
        if not self.running:
            return
        self.running = False

        # Give workers time to drain
        try:
            self.t_imu.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.t_save.join(timeout=5.0)
        except Exception:
            pass

        # Release camera and close files
        try:
            self.cap.release()
        except Exception:
            pass

        self.close()
        print("Stopped. Dataset written to:", os.path.abspath(DATASET_PATH))
        print("Use with ORB-SLAM3 mono_inertial_euroc:")
        print("  - Vocabulary file (ORBvoc)")
        print("  - Your settings YAML")
        print("  - Dataset path:", os.path.abspath(DATASET_PATH))
        print("  - Timestamps file:", os.path.abspath(self.paths["mh_txt"]))

    def close(self):
        for f in ["f_cam_csv", "f_mh", "f_imu"]:
            try:
                obj = getattr(self, f, None)
                if obj:
                    obj.flush()
                    obj.close()
            except Exception:
                pass


def main():
    rec = PiEuRoCRecorderBNO080()

    def handle_sigint(sig, frame):
        rec.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    rec.run()


if __name__ == "__main__":
    main()