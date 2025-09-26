import cv2
import numpy as np
import time
import os

# Set base path where config, weights, and names files are stored
base_path = "/Users/mac/VS code/Project-1"
weights_path = os.path.join(base_path, "yolov3.weights")
cfg_path = os.path.join(base_path, "yolov3.cfg")
names_path = os.path.join(base_path, "coco.names")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Load COCO class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load webcam (macOS users: try CAP_AVFOUNDATION if needed)
cap = cv2.VideoCapture(0, cv2.CAP_ANY)

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create a blob and set it as input to the network
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show FPS
    fps_label = f"FPS: {1 / (end - start):.2f}"
    cv2.putText(frame, fps_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
