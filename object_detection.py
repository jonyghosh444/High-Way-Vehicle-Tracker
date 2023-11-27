# YOLOv5 PyTorch HUB Inference (DetectionModels only)
import torch

model = torch.hub.load('/home/hb/Desktop/dynamic/yolov5', 'custom', path='/home/hb/Desktop/dynamic/yolov5/yolov5s.pt', source='local')  # PyTorch
im = 'Data/image.jpeg'  # file, Path, PIL.Image, OpenCV, nparray, list
results = model(im,augment=False)  # inference
boxes = results.xyxy[0].tolist()

boxno = 1
print(boxes)
for box in boxes:
    x,y,w,h,_,clss= box
    cx = int((x + x + w)/2)
    cy = int((y+y+h)/2)
    print("Box No:", boxno, " ", x, y, w, h)
    boxno+=1
    
results.show()