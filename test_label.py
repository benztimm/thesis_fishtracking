import cv2
import os
import numpy as np

video_name = 'video9862.mp4'
label_folder = 'labels_9862'
# Initialize video capture
video = cv2.VideoCapture(video_name)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame width, height, and FPS
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))


# Function to convert normalized YOLO coordinates to absolute coordinates
def convert_to_absolute(coords, frame_shape):
    _, x_center, y_center, width, height = map(float, coords)
    x_center_abs = int(x_center * frame_shape[1])
    y_center_abs = int(y_center * frame_shape[0])
    width_abs = int(width * frame_shape[1])
    height_abs = int(height * frame_shape[0])
    x1 = int(x_center_abs - (width_abs / 2))
    y1 = int(y_center_abs - (height_abs / 2))
    return x1, y1, width_abs, height_abs

# Get label folder
absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = label_folder
label_folder = os.path.join(absolute_path, relative_path)

# Get and sort label files
label_files = sorted(os.listdir(label_folder))

frame_index = 0
while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit when video ends
    
    # Read and convert labels if available
    if frame_index < len(label_files):
        label_file = os.path.join(label_folder, label_files[frame_index])
        with open(label_file, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                coords = list(map(float, line.split()))
                x1, y1, w, h = convert_to_absolute(coords, frame.shape)
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Draw green box
                cv2.putText(frame, 'GT', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    

    
    # Display frame
    cv2.imshow('Ground Truth Overlay', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_index += 1

# Release resources
video.release()
cv2.destroyAllWindows()
