import cv2
import os
import datetime
from pynput import keyboard

# Ask user for Android's IP address
ip_address = input("Enter your Android device's IP address (e.g., 192.168.1.100): ")
url = f"http://{ip_address}:8080/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

frame_count = 0
is_recording = False
out = None
folder = "slam_test/data"

if not os.path.exists(folder):
    os.makedirs(folder)

print("Press 'P' to start/stop recording. Press 'Q' to quit.")

def on_press(key):
    global is_recording, out, cap
    try:
        if key.char.lower() == "p":
            if not is_recording:
                print("Recording started...")
                is_recording = True
                now = datetime.datetime.now()
                filename = os.path.join(folder, now.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
                height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height), isColor=False)

            else:
                print("Recording stopped.")
                is_recording = False
                if out is not None:
                    out.release()
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or failed — trying again...")
        break

    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Processed Feed", processed_frame)
    frame_count += 1

    if is_recording and out is not None:
        out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
listener.stop()
