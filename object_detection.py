# YOLOv5 PyTorch HUB Inference (DetectionModels only)
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)  # or yolov5n - yolov5x6 or custom
im = 'image.jpeg'  # file, Path, PIL.Image, OpenCV, nparray, list
results = model(im)  # inference
print(results.xyxy[0])