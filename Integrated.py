import cv2
import torch
import numpy as np
import time
import threading
from ultralytics import YOLO

# Load YOLOv10 model
model = YOLO('yolov5l.pt').to("cuda")

# Define class names for the model
class_names = model.names

# Define the indices for 'person' and 'vehicle' classes
person_index = None
vehicle_indices = []

# Class names for vehicles (add more as needed)
vehicle_classes = ["car", "bus", "truck", "motorbike", "bicycle"]

# Prepare class indices for detection
for class_id, class_name in class_names.items():
    if class_name == 'person':
        person_index = class_id

if person_index is None:
    print("Error: 'person' class not found in the model.")
    exit()

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform = midas_transforms.small_transform
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Initialize the webcam
cap = cv2.VideoCapture("rtsp://192.168.1.25:554/ch01.264")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the depth threshold
depth_brightness_threshold = 240.0

# Create a named window (regular size)
cv2.namedWindow('YOLOv10 and MiDaS Object Detection')

# For FPS calculation
fps = 0
prev_time = time.time()

# Global variable for sharing frames between threads
frame_lock = threading.Lock()
current_frame = None
frame_available = False

# Function for reading frames in a separate thread
def capture_frames():
    global current_frame, frame_available
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = cv2.resize(frame, (640, 480))
                frame_available = True
        else:
            print("Error: Failed to capture image.")
            break

# Start the thread for capturing frames
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True  # Daemonize thread
capture_thread.start()

while True:
    with frame_lock:
        if not frame_available:
            continue
        frame = current_frame.copy()
        frame_available = False

    prev_time = time.time()

    # Perform object detection
    results = model(frame)[0]

    # Track if any relevant object (person/vehicle) is detected
    person_or_vehicle_detected = False
    brightest_value = float('-inf')

    # Only compute depth if relevant objects are detected
    for result in results.boxes.data:
        x1, y1, x2, y2, confidence, class_id = result
        class_id = int(class_id)

        if class_id == person_index:
            person_or_vehicle_detected = True
            break  # Exit loop early since a relevant object is detected

    if person_or_vehicle_detected:
        # Apply depth estimation to the resized frame
        input_batch = transform(frame).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        # Normalize depth map for visualization and calculations
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_BONE)

        # Process each result and find the closest object
        for result in results.boxes.data:
            x1, y1, x2, y2, confidence, class_id = result
            class_id = int(class_id)

            if class_id == person_index:
                # Get the depth values within the bounding box
                box_depth = depth_map_normalized[int(y1):int(y2), int(x1):int(x2)]

                # Find the brightest (maximum) value in the bounding box
                box_brightest_value = box_depth.max()

                # Update the brightest value if this is the closest one found
                brightest_value = max(brightest_value, box_brightest_value)

                label = f'{class_names[class_id]} {confidence:.2f}'
                cv2.rectangle(depth_map_colored, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(depth_map_colored, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Determine the final decision based on the closest object
        final_decision = "STOP" if brightest_value > depth_brightness_threshold else "GO"
        color = (0, 0, 255) if final_decision == "STOP" else (0, 255, 0)

        # Display the decision on the depth map
        cv2.putText(depth_map_colored, final_decision, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)

        # Display FPS on the depth map
        cv2.putText(depth_map_colored, f'FPS: {fps:.2f}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Display the depth map with the decision and FPS
        cv2.imshow('YOLOv10 and MiDaS Object Detection', depth_map_colored)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

