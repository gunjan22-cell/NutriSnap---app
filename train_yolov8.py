from ultralytics import YOLO
import os

# ✅ Define the correct dataset path
DATASET_PATH = "C:/Users/ABHISHEK BUDHWAT/OneDrive/Desktop/NEW_NUTRISNAP/Yolo_Dataset/data.yaml"

# ✅ Check if the dataset file exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or 'yolov8m.pt' if needed

# ✅ Train the model
model.train(
    data=DATASET_PATH,
    epochs=50,
    batch=16,
    imgsz=640,
    device="cpu",  # Change to 'cuda' if using GPU
    workers=4,
    project="trained_models",
    name="yolov8_indian_food"
)

print("🎉 Training Completed!")


