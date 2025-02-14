import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO('yolov8n.pt')

# Vehicle Classes (COCO IDs)
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

# Time weights (adjustable based on experiments)
k_motorcycle = 1.0  # Seconds per motorcycle
k_car = 2.0         # Seconds per car
k_truck_bus = 3.0   # Seconds per truck/bus

# Function to calculate green light duration for a video frame
def calculate_green_time(frame):
    if frame is None:
        return 10, 0, 0, 0  # Return default values if frame is not valid

    # Perform object detection
    results = model(frame)

    # Ensure valid results
    if len(results) == 0 or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 10, 0, 0, 0  # Return minimum green time if no detections

    detected_objects = results[0].boxes.data

    # Initialize vehicle counts
    motorcycle_count = 0
    car_count = 0
    truck_bus_count = 0

    # Assign weights based on vehicle type
    for obj in detected_objects:
        class_id = int(obj[5])  # Extract class ID

        if class_id == motorcycle_id:
            motorcycle_count += 1
        elif class_id == car_id:
            car_count += 1
        elif class_id in [bus_id, truck_id]:
            truck_bus_count += 1

    # Compute Green Light Duration
    green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)

    # Apply constraints (min 10 sec, max 90 sec)
    green_time = max(10, min(90, green_time))

    return green_time, motorcycle_count, car_count, truck_bus_count

# Open video file
video_path = "demo_video.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1
    
    # Process every 10th frame to reduce computation load
    if frame_count % 10 == 0:
        green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame)
        print(f"Frame {frame_count}: Motorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}")
        print(f"Recommended Green Light Duration: {green_time:.2f} seconds\n")
    
    # Display the frame
    cv2.imshow("Traffic Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
