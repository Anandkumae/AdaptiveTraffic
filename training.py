# Import necessary libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # or other YOLOv8 models like yolov8s.pt, yolov8m.pt, etc.

# Correct the image path (absolute or relative path)
img_path = 'lane2.jpg'  # Ensure the correct path to your image
img = cv2.imread(img_path)

# Run the model on the image
results = model(img)

# Iterate through each result and display it
for result in results:
    result.show()  # This automatically shows the image with bounding boxes

# If you want to display using Matplotlib instead of show()
# Convert the image to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes using Matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes for better visual
plt.show()
