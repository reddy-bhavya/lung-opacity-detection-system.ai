from ultralytics import YOLO

print("=" * 50)
print("TESTING YOLO INSTALLATION")
print("=" * 50)

# Download tiny model for testing
print("\nDownloading YOLOv8 model...")
model = YOLO('yolov8n-cls.pt')

print("\n✅ SUCCESS! YOLOv8 is working!")
print(f"Model type: {type(model)}")
print("=" * 50)