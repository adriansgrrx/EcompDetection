from ultralytics import YOLO

# Load a model
model = YOLO("path/to/model.pt")  # Load an official model or custom model

# Export the model
model.export(format="edgetpu")