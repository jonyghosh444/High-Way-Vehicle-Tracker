import cv2
import numpy as np
import torch

model = torch.hub.load('/home/hb/Desktop/dynamic/yolov5', 'custom', path='/home/hb/Desktop/dynamic/yolov5/yolov5s.pt', source='local')  # PyTorch

# Global variables to store the line coordinates
line_start = (-1, -1)
line_end = (-1, -1)
draw_line = False

# Scale factor for resizing bounding boxes
scale_factor_h = 0.35  # You can adjust this value as needed
scale_factor_w = 0.25  # You can adjust this value as needed

# Counting variables
vehicle_count = 0
prev_vehicle_count = 0

# Variable to keep track of vehicles that have crossed the line
vehicles_crossed = set()


# Function to check if two line segments intersect
def do_line_segments_intersect(p1, p2, p3, p4):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False

# Function to check if a bounding box intersects with a line segment
def is_bbox_intersecting_line(x, y, w, h, line_start, line_end):
    # Line segment represented by two points
    p1 = np.array(line_start, dtype=np.float32)
    p2 = np.array(line_end, dtype=np.float32)

    # Bounding box represented by four corners
    bbox_corners = np.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)], dtype=np.float32)

    # Check for intersection
    for i in range(4):
        j = (i + 1) % 4
        p3, p4 = bbox_corners[i], bbox_corners[j]
        # Check if the line segments intersect
        if do_line_segments_intersect(p1, p2, p3, p4):
            return True

    return False

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

count = 0

while True:
    # print(f"Vehicle count: {vehicle_count}")
    # Capture a frame from the video
    ret, frame = cap.read()
    count+=1
    # Check if the frame is successfully captured
    if not ret:
        print("Failed to capture frame. End of video?")
        break
    
    # 10 frame (skiping frame)
    if count % 3 !=0:
        continue

    # Detect objects on frame
    res = model(frame)

    boxes = res.xyxy[0].tolist()
    
    boxno = 1
    for box in boxes:
        (x, y, w, h, _, clss) = box
        x, y, w, h, clss= int(x), int(y), int(w), int(h), int(clss)
        
        if clss == 1 or clss == 2 or clss == 3 or clss == 5 or clss == 7:
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
            
            if draw_line and is_bbox_intersecting_line(x, y, w, h, line_start, line_end):
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
