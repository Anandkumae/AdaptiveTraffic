# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO('yolov8n.pt')

# # Read image (path should be valid for each lane image)
# img_paths = ["lane1.jpg", "lane2.jpg", "lane3.jpg", "lane4.jpg"]

# # Vehicle Classes (COCO IDs)
# motorcycle_id = 3
# car_id = 2
# bus_id = 5
# truck_id = 7

# # Time weights (adjustable based on experiments)
# k_motorcycle = 1.0 # Seconds per motorcycle
# k_car = 2.0        # Seconds per car
# k_truck_bus = 3.0   # Seconds per truck/bus

# # Function to calculate green light duration for each lane
# def calculate_green_time(img):
#     # Perform object detection
#     results = model(img)
    
#     # Extract detected objects
#     detected_objects = results[0].boxes.data
    
#     # Initialize vehicle counts
#     motorcycle_count = 0
#     car_count = 0
#     truck_bus_count = 0

#     # Assign weights based on vehicle type
#     for obj in detected_objects:
#         class_id = int(obj[5])  # Extract class ID

#         if class_id == motorcycle_id:
#             motorcycle_count += 1
#         elif class_id == car_id:
#             car_count += 1
#         elif class_id in [bus_id, truck_id]:
#             truck_bus_count += 1

#     # Compute Green Light Duration
#     green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)
    
#     # Apply constraints (min 10 sec, max 90 sec)
#     green_time = max(10, min(90, green_time))

#     return green_time, motorcycle_count, car_count, truck_bus_count

# # Create a figure with subplots for 4 lanes (2 rows and 2 columns)
# fig, axs = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)  # Use constrained layout

# # Loop over each lane and calculate green light time
# for lane_idx, (img_path, ax) in enumerate(zip(img_paths, axs.flatten()), start=1):
#     img = cv2.imread(img_path)

#     # Calculate green light time for current lane
#     green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(img)

#     # Display results for each lane
#     print(f"Lane {lane_idx}:")
#     print(f"  Motorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}")
#     print(f"  Recommended Green Light Duration: {green_time:.2f} seconds\n")

#     # Convert image to RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Display image in the subplot
#     ax.imshow(img_rgb)
#     ax.axis('off')

#     # Adjust title padding separately for top and bottom rows
#     title_pad = 30 if lane_idx <= 2 else 40  # Extra padding for bottom row
#     ax.set_title(f"Lane {lane_idx}\nMotorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}\n"
#                  f"Green Light Time: {green_time:.2f} sec", fontsize=10, pad=title_pad)

# # Add a main title
# fig.suptitle("Traffic Analysis for All Lanes", fontsize=16, fontweight='bold', y=1.02)

# plt.show()

pip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO('yolov8n.pt')

# Define image paths
img_paths = ["lane1.jpg", "lane2.jpg", "lane3.jpg", "lane4.jpg"]

# Vehicle Classes (COCO IDs)
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

# Time weights (adjustable based on experiments)
k_motorcycle = 1.0  # Seconds per motorcycle
k_car = 2.0         # Seconds per car
k_truck_bus = 3.0   # Seconds per truck/bus

# Function to calculate green light duration for each lane
def calculate_green_time(img):
    if img is None:
        return 10, 0, 0, 0  # Return default values if image is not loaded

    # Perform object detection
    results = model(img)

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

# Create a figure with subplots for 4 lanes (2 rows and 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

# Loop over each lane and calculate green light time
for lane_idx, (img_path, ax) in enumerate(zip(img_paths, axs.flatten()), start=1):
    if not os.path.exists(img_path):
        print(f"Warning: '{img_path}' not found. Skipping this lane.")
        ax.set_title(f"Lane {lane_idx}\nImage Not Found", fontsize=12, color="red")
        ax.axis('off')
        continue  # Skip this lane if the image is missing

    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: '{img_path}' could not be loaded. Check the file format.")
        ax.set_title(f"Lane {lane_idx}\nImage Load Failed", fontsize=12, color="red")
        ax.axis('off')
        continue  # Skip this lane if the image is unreadable

    # Calculate green light time for current lane
    green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(img)

    # Display results for each lane
    print(f"Lane {lane_idx}:")
    print(f"  Motorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}")
    print(f"  Recommended Green Light Duration: {green_time:.2f} seconds\n")

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display image in the subplot
    ax.imshow(img_rgb)
    ax.axis('off')

    # Adjust title padding separately for top and bottom rows
    title_pad = 30 if lane_idx <= 2 else 40  # Extra padding for bottom row
    ax.set_title(f"Lane {lane_idx}\nMotorcycles: {motorcycle_count}, Cars: {car_count}, Trucks/Buses: {truck_bus_count}\n"
                 f"Green Light Time: {green_time:.2f} sec", fontsize=10, pad=title_pad)

# Add a main title
fig.suptitle("Traffic Analysis for All Lanes", fontsize=16, fontweight='bold', y=1.02)

plt.show()
