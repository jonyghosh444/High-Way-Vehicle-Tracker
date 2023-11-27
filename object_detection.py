# YOLOv5 PyTorch HUB Inference (DetectionModels only)
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)  # or yolov5n - yolov5x6 or custom
im = 'Data/image.jpeg'  # file, Path, PIL.Image, OpenCV, nparray, list
results = model(im,augment=False)  # inference
boxes = results.xyxy[0].tolist()

boxno = 1
print(boxes)
for box in boxes:
    x,y,w,h,_,_ = box
    cx = int((x + x + w)/2)
    cy = int((y+y+h)/2)
    print("Box No:", boxno, " ", x, y, w, h)
    boxno+=1
    
results.show()