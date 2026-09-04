from ultralytics import YOLO
import os

os.makedirs("results", exist_ok=True)

# load the trained model (not the pretrained yolov8n checkpoint)
model = YOLO("runs/detect/train/weights/best.pt")

# run on all test images, save the ones with confidence above 0.5
results = model("dataset/test/images", conf=0.5, iou=0.5)

# save annotated results with bounding boxes
for i, result in enumerate(results):
    result.save(filename=f"results/detection_{i}.jpg")

print("Done! Check the results/ folder")
