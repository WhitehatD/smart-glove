import cv2
import os
import time

def main():
    # 1. Output Base Directory
    base_output_dir = "/Users/batigozen/Desktop/ORB_SLAM3_Dataset/ORB_SLAM3_Dataset_rect/EXPERIMENTS"
    os.makedirs(base_output_dir, exist_ok=True)

    # 2. Open Cameras
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    if not cap0.isOpened() or not cap1.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    # 3. Force 720p and 30 FPS
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # 4. Verify actual camera mode
    print("\nCamera configuration:")
    print(f"Cam0: {int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap0.get(cv2.CAP_PROP_FPS):.2f} FPS")
    print(f"Cam1: {int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap1.get(cv2.CAP_PROP_FPS):.2f} FPS")

    print("\nControls:")
    print("  's' -> Start/Stop Recording Sequence")
    print("  'q' -> Quit")

    is_recording = False
    session_dir = None
    file_cam0 = None
    file_cam1 = None
    timestamps_file = None
    frame_count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Error: Failed to grab frames.")
            break

        current_time_ns = time.time_ns()

        if is_recording:
            filename = f"{current_time_ns}.png"

            path_cam0 = os.path.join(session_dir, "mav0", "cam0", "data", filename)
            path_cam1 = os.path.join(session_dir, "mav0", "cam1", "data", filename)

            cv2.imwrite(path_cam0, frame0)
            cv2.imwrite(path_cam1, frame1)

            file_cam0.write(f"{current_time_ns}\n")
            file_cam1.write(f"{current_time_ns}\n")
            timestamps_file.write(f"{current_time_ns}\n")

            frame_count += 1

            # Visual indicator
            cv2.circle(frame0, (30, 30), 10, (0, 0, 255), -1)
            cv2.circle(frame1, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame0, f"REC {frame_count}", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera 0 (Left)", frame0)
        cv2.imshow("Camera 1 (Right)", frame1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            if not is_recording:
                # START RECORDING
                session_name = time.strftime("Session_%Y%m%d_%H%M%S")
                session_dir = os.path.join(base_output_dir, session_name)

                cam0_data_dir = os.path.join(session_dir, "mav0", "cam0", "data")
                cam1_data_dir = os.path.join(session_dir, "mav0", "cam1", "data")

                os.makedirs(cam0_data_dir, exist_ok=True)
                os.makedirs(cam1_data_dir, exist_ok=True)

                file_cam0 = open(os.path.join(session_dir, "mav0", "cam0", "data.csv"), "w")
                file_cam1 = open(os.path.join(session_dir, "mav0", "cam1", "data.csv"), "w")
                timestamps_file = open(os.path.join(session_dir, "timestamps.txt"), "w")

                header = "#timestamp [ns]\n"
                file_cam0.write(header)
                file_cam1.write(header)

                frame_count = 0
                is_recording = True
                print(f"\n[REC] Started recording to: {session_dir}")

            else:
                # STOP RECORDING
                is_recording = False
                file_cam0.close()
                file_cam1.close()
                timestamps_file.close()

                print(f"\n[STOP] Sequence saved. Total frames: {frame_count}")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
