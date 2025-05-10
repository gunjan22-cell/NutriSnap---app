from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model (use 'best.pt' if you have a custom-trained model)
YOLO_MODEL_PATH = 'yolov8n.pt'  # Change to 'best.pt' if using your custom-trained model
model = YOLO(YOLO_MODEL_PATH)

# Define confidence threshold
CONFIDENCE_THRESHOLD = 30  # Adjust as needed

def detect_objects(image):
    """Detects food items using YOLOv8 model."""
    
    # Ensure YOLO model is correctly loaded
    if model is None:
        print("Error: YOLO model is not loaded.")
        return []

    # Convert image to RGB format (YOLO expects RGB)
    if image is None or not isinstance(image, np.ndarray):
        print("Error: Invalid image input.")
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference on the image
    results = model(image_rgb)

    detected_foods = []
    
    for result in results:
        # Check if detection boxes exist
        if not hasattr(result.boxes, 'data'):
            continue
        
        # Get model class names
        class_names = model.names  # Extract class names from the model
        
        for box in result.boxes.data:
            try:
                # Extract detection box info
                x1, y1, x2, y2, confidence, class_id = box.tolist()
                confidence = round(confidence * 100, 2)
                
                if confidence > CONFIDENCE_THRESHOLD:
                    class_name = class_names.get(int(class_id), "Unknown")
                    detected_foods.append({'name': class_name, 'confidence': confidence})
            except Exception as e:
                print(f"Error processing detection: {e}")

    return detected_foods
