# Reading Phone Camera on a Laptop

This project streams the camera feed from your phone to your laptop so you can process it with Python and OpenCV.

**How it works:**  
- Install a camera streaming app on your phone (Android or iOS).  
- The app gives you a URL for the camera feed.  
- Your laptop connects to that URL using OpenCV.  
- You can then process the frames in Python.

**Important:**  
- Your phone and laptop must be on the same Wi‑Fi network.  
- The app must provide a direct video stream URL (often ending in `/video`).  

**Android Support:**  
Apps like *IP Webcam* or *DroidCam* can stream your Android phone camera.  

**iOS Support:**  
Apps like *EpocCam* or *NDI HX Camera* can stream your iPhone camera.
