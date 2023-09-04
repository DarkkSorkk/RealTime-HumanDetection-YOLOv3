Real-Time People Detection with YOLOv3 and OpenCV
Overview
This project utilizes the YOLOv3 (You Only Look Once, Version 3) object detection model and OpenCV to perform real-time people detection using a webcam. The script is written in Python and uses OpenCV for video capture and manipulation, along with YOLOv3 for object detection tasks.

Requirements
Python 3.x
OpenCV (cv2) library
YOLOv3 weights and configuration files
coco.names file (class labels)
How it Works
Load YOLOv3 Model
The YOLOv3 model is loaded using its weights and configuration files. Layer names are extracted to determine the output layers used for detection.

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indexes.flatten()]
Video Capture
The webcam is accessed using OpenCV's VideoCapture class. The coco.names file is read to get the class names.
cap = cv2.VideoCapture(0)
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
Object Detection
Each frame captured from the webcam is processed to detect people. The processed frame is displayed in real-time.
while True:
    # Capture frame
    ret, frame = cap.read()
    # Object detection logic here
    # ...
Non-Max Suppression (NMS)
To eliminate multiple bounding boxes around a single object, NMS is applied.
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
Display
The bounding boxes and class labels are drawn on the frame, which is then displayed using OpenCV.
cv2.imshow("Imagem", frame)
Termination
Press q to close the application.
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
How to Run
Make sure you have all the required files and libraries.
Run the script: python ambiente3.py
Press q to quit the application.
License
This project is open-source and available under the MIT License.
