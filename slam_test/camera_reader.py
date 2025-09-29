import cv2

# Change this to your Android's IP address
url = "http://145.127.70.75:8080/video"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Example processing: convert to grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Processed Feed", processed_frame)

    # Save frame every 30 frames
    if cv2.waitKey(1) & 0xFF == ord("s"):
        cv2.imwrite("processed_frame.png", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
