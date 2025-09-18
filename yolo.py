from ultralytics import YOLO
from PIL import Image
import torch

# Load YOLO model
model = YOLO("models/deepfashion2_yolov8s-seg.pt")

# Load image
image_path = "uploads/image.png"   # change to your image path
image = Image.open(image_path).convert("RGB")

# Run inference
results = model(image, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]

# Extract polygons
polygons = []
if hasattr(results, "masks") and results.masks is not None and hasattr(results.masks, "xy"):
    for mask in results.masks.xy:
        polygons.append(mask.tolist())

print("Polygons:", polygons)