import cv2
import os
import numpy as np
import csv
from scipy.optimize import minimize
from scipy.spatial.distance import mahalanobis
import tkinter as tk
from collections import defaultdict
import pandas as pd
import copy


TRACKER_NUM = 6

'''
#7482_F3
output_write_video = False
output_filename = 'output7482_F3_draw_box.mp4'

collect_data = False
switch = 'tracker'
version ='_1'
data_folder = 'data_7482_F3'

video_name = 'video7482_F3.mp4'

label_folder = 'labels_7482_F3'
reinitialization_data_filename = 'reinitialization_data_1_7482_F3.csv'
#
'''
#9862
output_write_video = False
output_filename = 'output9862_update-kalman.mp4'

collect_data = False
switch = 'kalman'
version ='_5'
data_folder = 'data_9862'

video_name = 'video9862.mp4'

label_folder = 'labels_9862'
reinitialization_data_filename = 'reinitialization_data_1_9862.csv'
#

current_dir = os.getcwd()
reinitialization_data_folder = 'reinitialization_data'
reinitialization_data_folder_path = os.path.join(current_dir, reinitialization_data_folder)


reinitialization_data_filename = os.path.join(reinitialization_data_folder_path, reinitialization_data_filename)


if switch == 'kalman':
    kal_or_tracker = f'kalman{version}'
else:
    kal_or_tracker = f'tracker{version}'

def compute_frame_rmse(estimated_data, ground_truth_data):
    """
    Compute the RMSE between estimated centroids and ground truth centroids 
    based on common frame IDs.

    Parameters:
        estimated_data (list of dicts): Each dict should contain:
            - 'Frame': an integer representing the frame number.
            - 'Centroid': a list or array of coordinates (e.g., [x, y]).
        ground_truth_data (list of dicts): Similar structure as estimated_data.

    Returns:
        float: The RMSE computed over frames that appear in both collections.
               Returns None if there are no common frames.
    """
    # Build dictionaries mapping frame id to centroid (as NumPy arrays)
    est_dict = {entry['Frame']: np.array(entry['Centroid']) for entry in estimated_data}
    gt_dict  = {entry['Frame']: np.array(entry['Centroid']) for entry in ground_truth_data}
    
    # Find the set of common frame IDs
    common_frames = sorted(set(est_dict.keys()).intersection(gt_dict.keys()))
    if not common_frames:
        print("No common frames found; cannot compute RMSE.")
        return None

    # Collect the differences for each common frame
    differences = []
    for frame in common_frames:
        diff = est_dict[frame] - gt_dict[frame]
        differences.append(diff)
    
    differences = np.array(differences)  # Shape: (n_common_frames, n_dims)
    
    # Compute squared Euclidean distance for each frame
    squared_distances = np.sum(differences**2, axis=1)
    
    # Compute the mean squared error (MSE) and then the RMSE
    mse = np.mean(squared_distances)
    rmse = np.sqrt(mse)
    return rmse
        
def filter_outliers(position_estimates, position_covariances):
    """
    Filters out outliers from position estimates using Mahalanobis distance.
    Returns filtered estimates and covariances.
    """
    if len(position_estimates) < 2:
        return position_estimates, position_covariances

    # Convert to numpy arrays
    estimates_array = np.array([est.flatten() for est in position_estimates])
    median_position = np.median(estimates_array, axis=0)
    cov_matrix = np.cov(estimates_array, rowvar=False)

    # Regularize the covariance matrix
    regularized_cov_matrix = regularize_covariance_matrix(cov_matrix)
    inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)

    # Calculate Mahalanobis distances
    mahalanobis_distances = [
        mahalanobis(pos.flatten(), median_position, inv_cov_matrix)
        for pos in estimates_array
    ]

    # Apply the 2-sigma rule
    distances_array = np.array(mahalanobis_distances)
    mean_distance = np.mean(distances_array)
    std_dev_distance = np.std(distances_array)
    lower_bound = mean_distance - 2 * std_dev_distance
    upper_bound = mean_distance + 2 * std_dev_distance

    # Filter indices within bounds
    valid_indices = [
        idx for idx, dist in enumerate(mahalanobis_distances)
        if lower_bound <= dist <= upper_bound
    ]

    # Return filtered estimates and covariances
    filtered_estimates = [position_estimates[idx] for idx in valid_indices]
    filtered_covariances = [position_covariances[idx] for idx in valid_indices]

    return filtered_estimates, filtered_covariances

def compute_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Each box is represented as (x, y, width, height).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute intersection and union areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


def save_distance_data_to_csv(csv_filename, distance_data):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Distance'])
        writer.writeheader()
        writer.writerows(distance_data)
        
def load_reinitialization_data(csv_filename):
    reinitialization_data = []
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            reinitialization_data.append({
                'Frame': int(row['Frame']),
                'Tracker': row['Tracker'],
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['w']),
                'h': int(row['h'])
            })
    return reinitialization_data

def regularize_covariance_matrix(cov_matrix, epsilon=1e-6):
    # Add a small value to the diagonal to make the matrix invertible
    return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

def select_trackers_to_reinitialize():
    """Creates a simple GUI for selecting which trackers to reinitialize."""
    selected_trackers = []

    # Create a new Tkinter window
    root = tk.Tk()
    root.title("Select Trackers")

    # List to hold the IntVar for each checkbox
    vars = []

    def on_select():
        """Callback for the selection button."""
        for i, var in enumerate(vars):
            if var.get() == 1:
                selected_trackers.append(i)
        root.destroy()  # Close the window after selection

    # Create a label
    tk.Label(root, text="Select the trackers to reinitialize:").pack()

    # Create checkboxes for each tracker
    for i, name in enumerate(tracker_type_name):
        var = tk.IntVar()
        chk = tk.Checkbutton(root, text=name, variable=var)
        chk.pack(anchor='w')
        vars.append(var)

    # Add a button to submit the selection
    tk.Button(root, text="Reinitialize", command=on_select).pack()

    # Start the Tkinter event loop
    root.mainloop()

    return selected_trackers

def multi_covariance_intersection(estimates, covariances):
    num_estimates = len(estimates)
    
    def objective(weights):
        # Normalize weights
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        
        # Compute weighted inverse covariance sum
        weighted_inv_cov_sum = sum(w * np.linalg.inv(P) for w, P in zip(weights, covariances))
        
        # Compute fused covariance
        fused_cov = np.linalg.inv(weighted_inv_cov_sum)
        
        # Minimize trace of fused covariance
        return np.trace(fused_cov)
    
    # Initialize equal weights
    initial_weights = np.ones(num_estimates) / num_estimates
    bounds = [(0, 1)] * num_estimates
    
    # Optimize weights
    result = minimize(objective, initial_weights, bounds=bounds)
    optimal_weights = result.x / np.sum(result.x)  # Normalize optimized weights
    
    # Compute fused covariance
    fused_cov = np.linalg.inv(sum(optimal_weights[j] * np.linalg.inv(covariances[j]) for j in range(num_estimates)))
    
    # Compute fused estimate
    fused_estimate = fused_cov.dot(
        sum(optimal_weights[j] * np.linalg.inv(covariances[j]).dot(estimates[j]) for j in range(num_estimates))
    )
    
    return fused_estimate, fused_cov


tracker_type_name = ['CSRT', 'KCF', 'MIL', 'BOOSTING', 'MEDIANFLOW',  'MOSSE', 'TLD']
tracker_type_name = tracker_type_name[:TRACKER_NUM]
def create_tracker(tracker_type):
    if tracker_type == 0:
        return cv2.TrackerCSRT.create()
    if tracker_type == 1:
        return cv2.legacy.TrackerKCF.create()
    if tracker_type == 2:
        return cv2.legacy.TrackerMIL.create()
    if tracker_type == 3:
        return cv2.legacy.TrackerBoosting.create()
    if tracker_type == 4:
        return cv2.legacy.TrackerMedianFlow.create()
    if tracker_type == 5:
        return cv2.legacy.TrackerMOSSE.create()
    if tracker_type == 6:
        return cv2.legacy.TrackerTLD.create()
    """
    if tracker_type == 7:
        return cv2.TrackerDaSiamRPN.create()
    if tracker_type == 8:
        return cv2.TrackerGOTURN.create()
    if tracker_type == 9:
        return cv2.TrackerCSRT.create()
    """
def is_centroid_overlapping(rect1, centroid_x2,centroid_y2):
    x1, y1, w1, h1 = rect1
    # Check if the centroid of rect2 lies within the boundaries of rect1
    return (x1 <= centroid_x2 <= (x1 + w1)) and (y1 <= centroid_y2 <= (y1 + h1))
  
    
def calculate_distance(centroid1, centroid2):
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)


def convert_to_absolute(coords, frame_shape, factor=1):
    _, x_center, y_center, width, height = map(float, coords)

    width *= factor
    height *= factor

    x_center_abs = int(x_center * frame_shape[1])
    y_center_abs = int(y_center * frame_shape[0])
    width_abs = int(width * frame_shape[1])
    height_abs = int(height * frame_shape[0])

    x1 = int(x_center_abs - (width_abs / 2))
    y1 = int(y_center_abs - (height_abs / 2))

    return x1, y1, width_abs, height_abs

def save_reinitialization_data_to_csv(csv_filename, reinitialization_data):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        writer.writerows(reinitialization_data)
        
def auto_reinitialize_trackers(current_frame, reinitialization_data, tracker):
    # Filter the reinitialization data for the current frame
    reinit_for_frame = [data for data in reinitialization_data if data['Frame'] == current_frame]
    if reinit_for_frame:
        print(f"Auto-reinitializing trackers at frame {current_frame}")
        #print(reinit_for_frame)
        
    for data in reinit_for_frame:
        try:
            tracker_idx = tracker_type_name.index(data['Tracker'])  # Get the tracker index from the tracker name
            roi = (data['x'], data['y'], data['w'], data['h'])
            #print(f"Auto-reinitializing {data['Tracker']} at frame {current_frame} with ROI: {roi}")
    
            # Reinitialize the tracker with the new ROI
    
            tracker[tracker_idx] = create_tracker(tracker_idx)
            tracker[tracker_idx].init(frame, roi)
        except Exception as e:
            print(e)
            continue

# Load the reinitialization data from the CSV file
csv_filename = reinitialization_data_filename
reinitialization_data = load_reinitialization_data(csv_filename)
#reinitialization_data = None
if reinitialization_data:
    print(f"Loaded {len(reinitialization_data)} reinitialization data entries")
    #print(reinitialization_data)
# get label folder
absolute_path = os.path.dirname(os.path.abspath(__file__))
relative_path = label_folder
label_folder = os.path.join(absolute_path, relative_path)

# get label file
label = [label for label in os.listdir(label_folder)]
label.sort()

# read label file and put it into a list
label_container = []    
for la in label:
    with open(os.path.join(label_folder, la), 'r') as f:
        line = f.read().splitlines()
        line_ac = [float(val) for sublist in line for val in sublist.split()]
        label_container.append(line_ac)
        
# read video file
video = cv2.VideoCapture(video_name)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()
    
# Read first frame
ret, frame = video.read()
# convert label to absolute coordinates
label_container_abs = [convert_to_absolute(coords, frame.shape) for coords in label_container]
# get first label and remove it from the container
first_location = label_container_abs.pop(0)

tracker = []
kalman = []
position_estimates = []
position_covariances = []
data_distance_collection = []
rmse_collection = defaultdict(list)
ground_truth_centroids_collection = []
rmse_over_time = []

for i in range(TRACKER_NUM):
    tracker.append(create_tracker(i))
    tracker[i].init(frame, first_location)
    kalman.append(cv2.KalmanFilter(4, 2))  # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
    kalman[i].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman[i].transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman[i].processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32) * 0.03
    kalman[i].measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
    x_init, y_init, w_init, h_init = first_location
    kalman[i].statePre = np.array([[x_init + w_init / 2], [y_init + h_init / 2], [0], [0]], np.float32)
    kalman[i].statePost = kalman[i].statePre

color = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),(128, 0, 128),(255, 165, 0)]

tracker_positions = []
failed_trackers = []
tracking_box_size = []
centroid = []
is_inside_collection = []
reinitialization_data_new = []
iou_collection = []
user_break = False

if output_write_video:
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 7
    output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for i in range(len(label_container_abs)):
    ret, frame = video.read()
    frame_copy = copy.deepcopy(frame)

    # Check if the user pressed 'x' to select a new ROI
    key = cv2.waitKey(30) & 0xFF
    if key == ord('x') or key == ord('X'):
        print("User pressed 'x'. Select a new ROI.")

        # Let the user select a new ROI
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        if roi is not None:
            print(f"Selected ROI: {roi}")

            # Call the tracker selection GUI
            selected_trackers = select_trackers_to_reinitialize()

            if selected_trackers:
                # Reinitialize only the selected trackers
                for j in selected_trackers:
                    print(f"Reinitializing {tracker_type_name[j]}")
                    tracker[j] = create_tracker(j)
                    tracker[j].init(frame, roi)
                    reinitialization_data_new.append({
                        'Frame': i,
                        'Tracker': tracker_type_name[j],
                        'x': int(roi[0]),
                        'y': int(roi[1]),
                        'w': int(roi[2]),
                        'h': int(roi[3])
                    })
                
                # If needed, reset Kalman filter states for the selected trackers
                """
                for j in selected_trackers:
                    kalman[j].statePre = np.array([[roi[0] + roi[2] / 2], [roi[1] + roi[3] / 2], [0], [0]], np.float32)
                    kalman[j].statePost = np.array([[roi[0] + roi[2] / 2], [roi[1] + roi[3] / 2], [0], [0]], np.float32)
                """
            else:
                print("No trackers selected for reinitialization.")


    if key == 27:  # Press 'Esc' to exit
        user_break = True
        break
    if reinitialization_data:
        auto_reinitialize_trackers(i, reinitialization_data, tracker)
    if not ret:
        break
    
    # Initialize the data collection for the current frame
    tracker_positions.append([])
    failed_trackers.append([])
    position_covariances.append([])
    position_estimates.append([])
    tracking_box_size.append([])
    centroid.append([])
    
    for j in range(TRACKER_NUM):
        success, box = tracker[j].update(frame_copy)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            centroid_x_tracking = int(x + w/2)
            centroid_y_tracking = int(y + h/2)
              
            tracker_positions[i].append([centroid_x_tracking, centroid_y_tracking])
            tracking_box_size[i].append([w, h])
            
             # Update Kalman Filter with the new measurement
            measurement = np.array([[np.float32(centroid_x_tracking)], [np.float32(centroid_y_tracking)]])
            kalman[j].correct(measurement)
            # Get the new prediction
            prediction = kalman[j].predict()
            # Draw the predicted position
            pt = (int(prediction[0]), int(prediction[1]))
            
                
            
            if switch == 'kalman':
                centroid[i].append({
                    'Tracker': j,
                    'x': int(prediction[0]),
                    'y': int(prediction[1])
                })
                cv2.circle(frame, pt, radius=5, color=color[j], thickness=-1)
                cv2.putText(frame, f'kalman_{tracker_type_name[j]}', (int(prediction[0]), int(prediction[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color[j], thickness=2)
            elif switch == 'tracker':
                centroid[i].append({
                'Tracker': j,
                'x': centroid_x_tracking,
                'y': centroid_y_tracking
            })
                iou = compute_iou([x, y, w, h], label_container_abs[i])
                iou_collection.append({
                    'Frame': i,
                    'Tracker': tracker_type_name[j],
                    'IoU': iou
                })
                cv2.circle(frame, (centroid_x_tracking, centroid_y_tracking), radius=5, color=color[j], thickness=-1)  # The -1 thickness fills the circle
                cv2.putText(frame, f'{tracker_type_name[j]}', (centroid_x_tracking, centroid_y_tracking-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color[j], thickness=2)
            
            # If needed, print or store these values
            position_estimate = kalman[j].statePost[:2]
            position_covariance = kalman[j].errorCovPost[:2, :2]
            position_estimates[i].append(position_estimate)
            position_covariances[i].append(position_covariance)

        else:
            failed_trackers[i].append(j)
            """
            tracker[j] = create_tracker(j)
            x_new, y_new, w_new, h_new = label_container_abs[i]
            tracker[j].init(frame, (x_new, y_new, w_new, h_new))
            """

    
    # plot ground truth
    ground_truth = label_container_abs[i]
    centroid_x = int(label_container_abs[i][0] + label_container_abs[i][2]/2)
    centroid_y = int(label_container_abs[i][1] + label_container_abs[i][3]/2)
    cv2.putText(frame, f'Real Fish Location', (label_container_abs[i][0], label_container_abs[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness=2)
    cv2.rectangle(frame, (label_container_abs[i][0],label_container_abs[i][1]), (label_container_abs[i][0]+label_container_abs[i][2], label_container_abs[i][1]+label_container_abs[i][3]), color=(0,0,0), thickness=2)
    cv2.circle(frame, (centroid_x, centroid_y), radius=7, color=(0,0,0), thickness=-1)  # The -1 thickness fills the circle
    #cv2.putText(frame, f'Real Fish Centroid', (centroid_x, centroid_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,0), thickness=2)
    
    if position_estimates[i]:
        # Filter outliers before performing CI
        filtered_position_estimates, filtered_position_covariances = filter_outliers(position_estimates[i], position_covariances[i])
        if filtered_position_estimates:
            # Perform Covariance Intersection with filtered data
            fused_position, fused_covariance = multi_covariance_intersection(
                filtered_position_estimates, filtered_position_covariances
            )
            # Draw the fused position and uncertainty
            x_fuse, y_fuse = int(fused_position[0]), int(fused_position[1])
            cv2.circle(frame, (x_fuse, y_fuse), 7, (255, 255, 255), -1)
            cv2.putText(frame, f'Fused location', (x_fuse, y_fuse - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            fuse_distance = calculate_distance([x_fuse, y_fuse], [centroid_x, centroid_y])
            data_distance_collection.append({
                'Frame': i,
                'Tracker': 'Fused',
                'Distance': fuse_distance
            })
            rmse_collection['Fuse'].append({'Frame': i, 'Centroid': [x_fuse, y_fuse]})
            #fuse_centroid_box = adjust_centroid_to_ground_truth_size(x, y, ground_truth_width, ground_truth_height)
            #iou = calculate_iou(fuse_centroid_box, ground_truth)
            #print(f"IoU for fused tracker: {iou}")
            #cv2.rectangle(frame, (fuse_centroid_box[0], fuse_centroid_box[1]), (fuse_centroid_box[0] + fuse_centroid_box[2], fuse_centroid_box[1] + fuse_centroid_box[3]), color=(255,255,255), thickness=2)
            is_inside_fuse = is_centroid_overlapping(label_container_abs[i], x_fuse, y_fuse)
            is_inside_collection.append({
                'Frame': i,
                'Tracker': 'Fused',
                'Is_inside': is_inside_fuse
            })
        else:
            print("No valid trackers within bounds for CI.")
    else:
        print("No position estimates available for CI.")
        
        
    # draw ellipse
    # Calculate the median position of all trackers
    tracker_positions_array = np.array(tracker_positions[i])
    median_position = np.median(tracker_positions_array, axis=0)

    # Calculate the mean box size of all trackers
    tracker_box_size_array = np.array(tracking_box_size[i])
    mean_box_size = np.mean(tracker_box_size_array, axis=0)

    # Calculate the covariance matrix of the positions
    position_estimates_array = np.array([pe.flatten() for pe in position_estimates[i]])
    cov_matrix = np.cov(position_estimates_array, rowvar=False) 

    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    # Calculate the angle of rotation for the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Lengths of the ellipse axes (scaled by the Mahalanobis distance level)
    # Here, we use a scaling factor of 2, which corresponds to approximately the 95% confidence interval in 2D
    axis_length1 = 2 * np.sqrt(eigenvalues[0]) * 2  # Major axis
    axis_length2 = 2 * np.sqrt(eigenvalues[1]) * 2  # Minor axis

    # Draw the overall Mahalanobis ellipse on the frame
    center = (int(median_position[0]), int(median_position[1]))
    cv2.ellipse(
        frame, center,
        axes=(int(axis_length1), int(axis_length2)),
        angle=angle,
        startAngle=0, endAngle=360,
        color=(255, 255, 255), thickness=2
    )

    
    ground_truth_centroids_collection.append({'Frame': i, 'Centroid': [centroid_x, centroid_y]})
    """
        centroid data collection
    """
    for c in centroid[i]:
        distance = calculate_distance([c['x'], c['y']], [centroid_x, centroid_y])
        data_distance_collection.append({
            'Frame': i,
            'Tracker': tracker_type_name[c['Tracker']],
            'Distance': distance
        })
        rmse_collection[tracker_type_name[c['Tracker']]].append({'Frame': i, 'Centroid': [c['x'], c['y']]})
    
    w,h = int(mean_box_size[0]), int(mean_box_size[1])
    x_min = int(x_fuse - w // 2)
    y_min = int(y_fuse - h // 2)
    cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), color=(255,255,255), thickness=2)
    cv2.putText(frame, f'fused_box_from_mean_size', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if switch == 'tracker':
        iou = compute_iou([x_min, y_min, w, h], ground_truth)
        iou_collection.append({
            'Frame': i,
            'Tracker': 'Fused',
            'IoU': iou
        })
    
    for j in failed_trackers[i]:
        print(f"Tracker {tracker_type_name[j]} failed, reinitializing tracker...")
        tracker[j] = create_tracker(j)
        tracker[j].init(frame_copy, (x_min, y_min, w, h))
        kalman[j].statePre = np.array([[x_min + w / 2], [y_min + h / 2], [0], [0]], np.float32)
        kalman[j].statePost = kalman[j].statePre
            
    for c in centroid[i]:
        is_inside = is_centroid_overlapping(label_container_abs[i], c['x'], c['y'])
        is_inside_collection.append({
            'Frame': i,
            'Tracker': tracker_type_name[c['Tracker']],
            'Is_inside': is_inside
        })
    #=======================================================================================================
    # RMSE over time
    # Compute RMSE for each tracker
    for tracker_name, rmse_data in rmse_collection.items():
        rmse = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
        rmse_over_time.append({
            'Frame': i,
            'Tracker': tracker_name,
            'Rmse': rmse
        })     
    if output_write_video:
        output_video.write(frame)

    frame_resized = cv2.resize(frame, (1920,1080))
    cv2.imshow('Frame', frame)
    
cv2.destroyAllWindows()
if output_write_video:
    output_video.release()
video.release()


if not user_break and collect_data:
    folder = data_folder
    csv_filename = f'distance_data_{kal_or_tracker}.csv'
    save_distance_data_to_csv(os.path.join(folder,csv_filename), data_distance_collection)

    csv_filename = f'is_inside_{kal_or_tracker}.csv'
    with open(os.path.join(folder,csv_filename), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Is_inside'])
        writer.writeheader()
        writer.writerows(is_inside_collection)
        
    rmse_calculated_collection = []
    for tracker_name, rmse_data in rmse_collection.items():
        rmse = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
        rmse_calculated_collection.append({
                'Tracker': tracker_name,
                'RMSE': rmse
            })
    csv_filename = f'rmse_{kal_or_tracker}.csv'
    with open(os.path.join(folder,csv_filename), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Tracker', 'RMSE'])
        writer.writeheader()
        writer.writerows(rmse_calculated_collection)
        
    csv_filename = f'rmse_over_time_{kal_or_tracker}.csv'
    with open(os.path.join(folder,csv_filename), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Rmse'])
        writer.writeheader()
        writer.writerows(rmse_over_time)
        
    if switch == 'tracker':
        iou_csv_filename = f'iou_{kal_or_tracker}.csv'
        with open(os.path.join(os.path.join(folder,iou_csv_filename)), mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'IoU'])
            writer.writeheader()
            writer.writerows(iou_collection)
    print("Data saved to CSV files.")
else:
    print("Data not saved to CSV files. User break the program.")
    df_data_distance_collection = pd.DataFrame(data_distance_collection)
    df_is_inside_collection = pd.DataFrame(is_inside_collection)
    rmse_calculated_collection = []
    for tracker_name, rmse_data in rmse_collection.items():
        rmse = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
        rmse_calculated_collection.append({
                'Tracker': tracker_name,
                'RMSE': rmse
            })
    df_rmse_calculated_collection = pd.DataFrame(rmse_calculated_collection)
    df_iou_collection = pd.DataFrame(iou_collection)
    df_reinit = pd.DataFrame(reinitialization_data)

    
    tracker_counts = df_is_inside_collection.groupby('Tracker')['Is_inside'].value_counts().sort_values(ascending=False)
    max_distance = df_data_distance_collection.groupby('Tracker')['Distance'].max().sort_values(ascending=False)
    min_distance = df_data_distance_collection.groupby('Tracker')['Distance'].min().sort_values()
    average_distance = df_data_distance_collection.groupby('Tracker')['Distance'].mean().sort_values()
    rmse = df_rmse_calculated_collection.sort_values(by='RMSE')
    average_iou = df_iou_collection.groupby('Tracker')['IoU'].mean().sort_values(ascending=False) if switch == 'tracker' else None
    
    total_tracker_counts = df_is_inside_collection.groupby('Tracker')['Is_inside'].count()
    failed_counts = len(label_container_abs) - total_tracker_counts
    failed_counts.sort_values(ascending=False)
    
    print("================================================================================================")
    print(f"Max Distance:\n{max_distance}")
    print("-" * 50)
    print(f"Min Distance:\n{min_distance}")
    print("-" * 50)
    print(f"Average Distance:\n{average_distance}")
    print("-" * 50)
    print(f"RMSE:\n{rmse}")
    print("-" * 50)
    print(f"is_inside:\n{tracker_counts}")
    print("-" * 50)
    print(f"Average IoU:\n{average_iou}")
    print("-" * 50)
    print(f"Reinitialization Count:\n{df_reinit['Tracker'].value_counts(ascending=False)}")
    print("-" * 50)
    print(f"Failed Counts:\n{failed_counts.sort_values(ascending=False)}")


"""
csv_filename = 'reinitialization_data_3_v2.csv'
if not csv_filename:
    print("No reinitialization data file name.")
    file_name = input("Please enter the file name for save_reinitialization_data: ")
    save_reinitialization_data_to_csv(file_name, reinitialization_data_new)
else:
    save_reinitialization_data_to_csv(csv_filename, reinitialization_data_new)
"""






