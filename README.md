🧠 Face Detection and Blurring using YOLOv8 (Google Colab Compatible)
This project uses Ultralytics YOLOv8 to detect human faces in video frames and apply a Gaussian blur for privacy. It's designed to work seamlessly in Google Colab, where traditional OpenCV display functions are disabled.

📂 Files
face_blur_yolov8_colab.py – Python script (see code in example below)

yolov8n-face.pt – YOLOv8 nano model trained for face detection

README.md – Project documentation

🔧 Requirements
Make sure your Colab environment has the following installed:

bash
Copy
Edit
pip install ultralytics opencv-python-headless
📥 Getting Started in Colab
Upload Files

Upload yolov8n-face.pt and your video (e.g., sample_video.mp4) using Colab’s file upload interface.

Run the Script

python
Copy
Edit
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2
import time

# Load YOLOv8 face detection model
model = YOLO("yolov8n-face.pt")

# Load video file (or use 0 for webcam if running locally)
cap = cv2.VideoCapture("sample_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Process detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_region = frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred_face

    # Display in Colab
    cv2_imshow(frame)
    time.sleep(0.05)

cap.release()
🧪 Example Output
Each frame will show blurred faces in real-time. The output is rendered in the Colab notebook using cv2_imshow().

💡 Notes
Do not use cv2.imshow() in Colab — it will throw a DisabledFunctionError.

For large videos, consider sampling fewer frames to reduce processing time.

📦 Credits
Model: Ultralytics YOLOv8

Face detection weights: YOLOv8n-Face

