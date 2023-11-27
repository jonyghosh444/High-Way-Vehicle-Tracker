import cv2
import numpy as np
import torch

model = torch.hub.load('/home/hb/Desktop/dynamic/yolov5', 'custom', path='/home/hb/Desktop/dynamic/yolov5/yolov5s.pt', source='local')  # PyTorch
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Global variables to store the line coordinates
line_start = (-1, -1)
line_end = (-1, -1)
draw_line = False

# Scale factor for resizing bounding boxes
scale_factor_h = 0.3  # You can adjust this value as needed
scale_factor_w = 0.15  # You can adjust this value as needed

# Counting variables
vehicle_count = 0
prev_vehicle_count = 0

# Variable to keep track of vehicles that have crossed the line
vehicles_crossed = set()

# Function to check if a bounding box intersects with a line
def is_intersecting_line(x1, y1, x2, y2, line_start, line_end):
    # Use a simple check for intersection between bounding box and line
    return cv2.pointPolygonTest(np.array([line_start, line_end], np.int32), (x1, y1), False) >= 0 or \
           cv2.pointPolygonTest(np.array([line_start, line_end], np.int32), (x2, y2), False) >= 0

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global line_start, line_end, draw_line

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button is clicked
        if not draw_line:
            line_start = (x, y)
            draw_line = True
        else:
            line_end = (x, y)
            draw_line = False

# Create a window and set the callback function
cv2.namedWindow('VideoFrame')
cv2.setMouseCallback('VideoFrame', mouse_callback)

# Create a video capture object (replace 'your_video.mp4' with your video file)
cap = cv2.VideoCapture('Data/video.mp4')

while True:
    print(f"vehicle no: {vehicle_count}")
    # Capture a frame from the video
    ret, frame = cap.read()

    # Check if the frame is successfully captured
    if not ret:
        print("Failed to capture frame. End of video?")
        break

    # Detect objects on frame
    res = model(frame)

    boxes = res.xyxy[0].tolist()
    boxno = 1
    for box in boxes:
        (x, y, w, h, _, _) = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure coordinates are non-negative
        x, y = max(0, x), max(0, y)

        # Scale down the width and height of the bounding box
        w, h = int(w * scale_factor_w), int(h * scale_factor_h)

        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        # print("Box No:", boxno, " ", x, y, w, h)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the bounding box intersects with the line
        if draw_line and is_intersecting_line(x, y, x + w, y + h, line_start, line_end):
            # Check if the vehicle is not already marked as crossed
            if boxno not in vehicles_crossed:
                vehicles_crossed.add(boxno)
                vehicle_count += 1

        boxno += 1

    # Draw the dynamic line if both start and end points are set
    if draw_line:
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
        
    

    # Display the vehicle count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('VideoFrame', frame)

    # Check if the vehicle count has changed
    if vehicle_count != prev_vehicle_count:
        print("Vehicle Count:", vehicle_count)
        prev_vehicle_count = vehicle_count

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
