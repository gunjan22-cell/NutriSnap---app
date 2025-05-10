

import requests

url = "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt"
response = requests.get(url, stream=True)

with open("C:/Users/ABHISHEK BUDHWAT/OneDrive/Desktop/NEW_NUTRISNAP/yolov8n.pt", "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print("✅ Download complete: yolov8n.pt")
