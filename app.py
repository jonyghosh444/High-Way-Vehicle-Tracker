import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)  # or yolov5n - yolov5x6 or custom

# Global variables to store the line coordinates
line_start = (-1, -1)
line_end = (-1, -1)
draw_line = False

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
cap = cv2.VideoCapture('video.mp4')

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
        # Detect objects on frame
    res = model.detect(frame)
    
    # (class_ids, scores, boxes) = model(frame)
    # boxno = 1
    # for box in boxes:
    #     (x, y, w, h) = box
    #     cx = int((x + x + w)/2)
    #     cy = int((y+y+h)/2)
    #     print("Box No:", boxno, " ", x, y, w, h)

    #     # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    #     cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 2)
    #     center_points_crnt_frame.append((cx, cy))
    #     boxno += 1

    # Draw the dynamic line if both start and end points are set
    if line_start != (-1, -1) and line_end != (-1, -1):
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('VideoFrame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
