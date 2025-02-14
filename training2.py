# Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor

# Load YOLOv8 Model
model = YOLO('yolov8n.pt')  # Using YOLOv8 Nano version (smallest model)

# Load Image
img_path = "lane1.jpg"  # Replace with your image path
img = cv2.imread(img_path)

# Perform Object Detection
results = model(img)

# Extract detected objects
detected_objects = results[0].boxes.data  # YOLO detection results

# Count Vehicles
vehicle_classes = [2, 3, 5, 7]  # COCO class indices for car, motorcycle, bus, truck
vehicle_count = sum(1 for obj in detected_objects if int(obj[5]) in vehicle_classes)

print(f"Detected Vehicles: {vehicle_count}")

# ---- Random Forest Regression for Traffic Signal Timing ----

# Example Dataset: [Vehicle Count, Green Light Duration]
# This dataset should ideally come from real traffic data.
train_data = np.array([
    [5, 10],   # 5 vehicles -> 10 sec green time
    [10, 15],  # 10 vehicles -> 15 sec green time
    [15, 20],  # 15 vehicles -> 20 sec green time
    [20, 30],  # 20 vehicles -> 30 sec green time
    [25, 40],  # 25 vehicles -> 40 sec green time
    [30, 50],  # 30 vehicles -> 50 sec green time
])

# Split dataset into input (X) and output (Y)
X_train = train_data[:, 0].reshape(-1, 1)  # Vehicle count
Y_train = train_data[:, 1]  # Green time

# Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Predict Green Light Duration
predicted_green_time = rf_model.predict([[vehicle_count]])
print(f"Recommended Green Light Duration: {predicted_green_time[0]:.2f} seconds")

# ---- Display Image with Detected Vehicles ----

# Convert image to RGB for proper visualization
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot image with bounding boxes
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Detected Vehicles: {vehicle_count}")
plt.show()
