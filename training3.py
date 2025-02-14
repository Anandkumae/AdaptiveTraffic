import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Read image
img_path = "ait1.jpg"  # Ensure this image exists
img = cv2.imread(img_path)

# Perform object detection
results = model(img)

# Extract detected objects
detected_objects = results[0].boxes.data

# Vehicle Classes (COCO IDs)
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

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

# Time weights (adjustable based on experiments)
k_motorcycle = 1.5  # Seconds per motorcycle
k_car = 2.5         # Seconds per car
k_truck_bus = 4.5   # Seconds per truck/bus

# Compute Green Light Duration
green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)

# Display results
print(f"Motorcycles: {motorcycle_count}, Cars: {car_count}, Buses/Trucks: {truck_bus_count}")
print(f"Recommended Green Light Duration: {green_time:.2f} seconds")

# Convert image to RGB and display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Motorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}\n"
          f"Green Light Time: {green_time:.2f} sec")
plt.show()
