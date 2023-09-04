# 📹 Real-Time People Detection with YOLOv3 and OpenCV 🕵️‍♀️

## 🌟 Overview

This project leverages the YOLOv3 (You Only Look Once, Version 3) object detection model coupled with OpenCV to perform real-time people detection via a webcam. The script is implemented in Python and uses OpenCV for video capture and manipulation, while YOLOv3 is responsible for the object detection tasks.

## 🛠 Requirements

- Python 3.x
- OpenCV (cv2) library
- YOLOv3 weights and configuration files
- `coco.names` file for class labels

> 🚨 **Note**: Don't forget to download the YOLOv3 weights as they are too large to be included in a Git repository.

## 🎯 How it Works

### 🔍 Load YOLOv3 Model

The YOLOv3 model is initialized using its weights and configuration files. Layer names are extracted to ascertain the output layers used for detection.

```python
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indexes.flatten()]

📸 Video Capture
The webcam is accessed using OpenCV's VideoCapture class. The coco.names file is read to obtain class names.
cap = cv2.VideoCapture(0)
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

👀 Object Detection
Each frame captured from the webcam is processed for object detection. The processed frame is then displayed in real-time.

while True:
    # Capture frame
    ret, frame = cap.read()
    # Object detection logic here
    # ...

#📦 Non-Max Suppression (NMS)
To remove redundant bounding boxes around a single object, NMS is applied.

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

🖼 Display
Bounding boxes and class labels are overlaid on the frame, which is subsequently displayed using OpenCV.

cv2.imshow("Image", frame)

🛑 Termination
Press q to terminate the application.

if cv2.waitKey(1) & 0xFF == ord('q'):
    break

🚀 How to Run
Ensure all required files and libraries are installed.
Run the script: python ambiente3.py
Press q to exit the application.

📜 License
This project is open-source and available under the MIT License.

