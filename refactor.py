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

# =============================================================================
# Configuration & Global Constants
# =============================================================================

TRACKER_NUM = 6
OUTPUT_WRITE_VIDEO = False
OUTPUT_FILENAME = 'output9862_update-kalman.mp4'
COLLECT_DATA = False
SWITCH = 'tracker'  # 'kalman' or 'tracker'
VERSION = '_5'
DATA_FOLDER = 'data_9862'
VIDEO_NAME = 'video9862.mp4'
LABEL_FOLDER = 'labels_9862'
REINIT_FILENAME = 'reinitialization_data_1_9862.csv'

CURRENT_DIR = os.getcwd()
REINIT_DATA_FOLDER = os.path.join(CURRENT_DIR, 'reinitialization_data')
REINIT_DATA_FILENAME = os.path.join(REINIT_DATA_FOLDER, REINIT_FILENAME)

if SWITCH == 'kalman':
    KAL_OR_TRACKER = f'kalman{VERSION}'
else:
    KAL_OR_TRACKER = f'tracker{VERSION}'
    
    
"""  
TRACKER_NUM = 6
OUTPUT_WRITE_VIDEO = False
OUTPUT_FILENAME = 'output7482_F3_draw_box.mp4'
COLLECT_DATA = False
SWITCH = 'kalman'  # 'kalman' or 'tracker'
VERSION = '_1'
DATA_FOLDER = 'data_7482_F3'
VIDEO_NAME = 'video7482_F3.mp4'
LABEL_FOLDER = 'labels_7482_F3'
REINIT_FILENAME = 'reinitialization_data_1_7482_F3.csv'

CURRENT_DIR = os.getcwd()
REINIT_DATA_FOLDER = os.path.join(CURRENT_DIR, 'reinitialization_data')
REINIT_DATA_FILENAME = os.path.join(REINIT_DATA_FOLDER, REINIT_FILENAME)

if SWITCH == 'kalman':
    KAL_OR_TRACKER = f'kalman{VERSION}'
else:
    KAL_OR_TRACKER = f'tracker{VERSION}'
"""

# =============================================================================
# Utility Functions
# =============================================================================
def compute_frame_rmse(estimated_data, ground_truth_data):
    """Compute RMSE between estimated and ground truth centroids based on common frames."""
    est_dict = {entry['Frame']: np.array(entry['Centroid']) for entry in estimated_data}
    gt_dict  = {entry['Frame']: np.array(entry['Centroid']) for entry in ground_truth_data}
    common_frames = sorted(set(est_dict.keys()).intersection(gt_dict.keys()))
    if not common_frames:
        print("No common frames found; cannot compute RMSE.")
        return None
    differences = [est_dict[frame] - gt_dict[frame] for frame in common_frames]
    differences = np.array(differences)
    squared_distances = np.sum(differences**2, axis=1)
    mse = np.mean(squared_distances)
    return np.sqrt(mse)

def filter_outliers(position_estimates, position_covariances):
    """
    Filters out outliers using Mahalanobis distance.
    Returns filtered estimates and covariances.
    """
    if len(position_estimates) < 2:
        return position_estimates, position_covariances

    estimates_array = np.array([est.flatten() for est in position_estimates])
    median_position = np.median(estimates_array, axis=0)
    cov_matrix = np.cov(estimates_array, rowvar=False)
    regularized_cov_matrix = regularize_covariance_matrix(cov_matrix)
    inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)

    mahalanobis_distances = [
        mahalanobis(pos.flatten(), median_position, inv_cov_matrix)
        for pos in estimates_array
    ]
    distances_array = np.array(mahalanobis_distances)
    mean_distance = np.mean(distances_array)
    std_dev_distance = np.std(distances_array)
    lower_bound = mean_distance - 2 * std_dev_distance
    upper_bound = mean_distance + 2 * std_dev_distance

    valid_indices = [idx for idx, dist in enumerate(mahalanobis_distances)
                     if lower_bound <= dist <= upper_bound]
    filtered_estimates = [position_estimates[idx] for idx in valid_indices]
    filtered_covariances = [position_covariances[idx] for idx in valid_indices]
    return filtered_estimates, filtered_covariances

def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    return intersection_area / union_area

def save_distance_data_to_csv(csv_filename, distance_data):
    """Save distance data to CSV."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Distance'])
        writer.writeheader()
        writer.writerows(distance_data)

def load_reinitialization_data(csv_filename):
    """Load reinitialization data from CSV."""
    reinit_data = []
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            reinit_data.append({
                'Frame': int(row['Frame']),
                'Tracker': row['Tracker'],
                'x': int(row['x']),
                'y': int(row['y']),
                'w': int(row['w']),
                'h': int(row['h'])
            })
    return reinit_data

def regularize_covariance_matrix(cov_matrix, epsilon=1e-6):
    """Regularize a covariance matrix by adding a small value on its diagonal."""
    return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

def select_trackers_to_reinitialize(tracker_type_name):
    """Create a simple GUI to select trackers for reinitialization."""
    selected_trackers = []
    root = tk.Tk()
    root.title("Select Trackers")
    vars = []
    
    def on_select():
        for i, var in enumerate(vars):
            if var.get() == 1:
                selected_trackers.append(i)
        root.destroy()

    tk.Label(root, text="Select the trackers to reinitialize:").pack()
    for i, name in enumerate(tracker_type_name):
        var = tk.IntVar()
        tk.Checkbutton(root, text=name, variable=var).pack(anchor='w')
        vars.append(var)
    tk.Button(root, text="Reinitialize", command=on_select).pack()
    root.mainloop()
    return selected_trackers

def multi_covariance_intersection(estimates, covariances):
    """Fuse multiple estimates using the covariance intersection method."""
    num_estimates = len(estimates)
    
    def objective(weights):
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)
        weighted_inv_cov_sum = sum(w * np.linalg.inv(P) for w, P in zip(weights, covariances))
        fused_cov = np.linalg.inv(weighted_inv_cov_sum)
        return np.trace(fused_cov)
    
    initial_weights = np.ones(num_estimates) / num_estimates
    bounds = [(0, 1)] * num_estimates
    result = minimize(objective, initial_weights, bounds=bounds)
    optimal_weights = result.x / np.sum(result.x)
    
    fused_cov = np.linalg.inv(
        sum(optimal_weights[j] * np.linalg.inv(covariances[j]) for j in range(num_estimates))
    )
    fused_estimate = fused_cov.dot(
        sum(optimal_weights[j] * np.linalg.inv(covariances[j]).dot(estimates[j])
            for j in range(num_estimates))
    )
    return fused_estimate, fused_cov

def create_tracker(tracker_type):
    """Factory method to create a tracker given its type index."""
    if tracker_type == 0:
        return cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.legacy.TrackerCSRT_create()
    elif tracker_type == 1:
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 2:
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 3:
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 4:
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 5:
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 6:
        return cv2.legacy.TrackerTLD_create()
    else:
        raise ValueError("Invalid tracker type.")

def is_centroid_overlapping(rect, centroid_x, centroid_y):
    """Check if a given centroid lies within a rectangle."""
    x, y, w, h = rect
    return (x <= centroid_x <= (x + w)) and (y <= centroid_y <= (y + h))

def calculate_distance(centroid1, centroid2):
    """Compute the Euclidean distance between two centroids."""
    return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)

def convert_to_absolute(coords, frame_shape, factor=1):
    """Convert relative label coordinates to absolute frame coordinates."""
    _, x_center, y_center, width, height = map(float, coords)
    width *= factor
    height *= factor
    x_center_abs = int(x_center * frame_shape[1])
    y_center_abs = int(y_center * frame_shape[0])
    width_abs = int(width * frame_shape[1])
    height_abs = int(height * frame_shape[0])
    x1 = int(x_center_abs - width_abs / 2)
    y1 = int(y_center_abs - height_abs / 2)
    return x1, y1, width_abs, height_abs

def save_reinitialization_data_to_csv(csv_filename, reinit_data):
    """Save reinitialization data into a CSV file."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'x', 'y', 'w', 'h'])
        writer.writeheader()
        writer.writerows(reinit_data)

def auto_reinitialize_trackers(current_frame, reinit_data, trackers, tracker_type_name, frame):
    """Automatically reinitialize trackers if reinitialization data exists for the current frame."""
    reinit_for_frame = [data for data in reinit_data if data['Frame'] == current_frame]
    if reinit_for_frame:
        print(f"Auto-reinitializing trackers at frame {current_frame}")
    for data in reinit_for_frame:
        try:
            tracker_idx = tracker_type_name.index(data['Tracker'])
            roi = (data['x'], data['y'], data['w'], data['h'])
            trackers[tracker_idx] = create_tracker(tracker_idx)
            trackers[tracker_idx].init(frame, roi)
        except Exception as e:
            print(e)
            continue

def initialize_trackers(frame, first_location):
    """Initialize trackers and corresponding Kalman filters."""
    trackers = []
    kalman_filters = []
    for i in range(TRACKER_NUM):
        tracker = create_tracker(i)
        tracker.init(frame, first_location)
        trackers.append(tracker)

        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 5, 0],
                                       [0, 0, 0, 5]], np.float32) * 0.03
        kf.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32) * 1
        x_init, y_init, w_init, h_init = first_location
        state_init = np.array([[x_init + w_init / 2],
                               [y_init + h_init / 2],
                               [0],
                               [0]], np.float32)
        kf.statePre = state_init
        kf.statePost = state_init
        kalman_filters.append(kf)
    return trackers, kalman_filters

# =============================================================================
# Main Processing Function
# =============================================================================
def main():
    tracker_type_name = ['CSRT', 'KCF', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'MOSSE', 'TLD'][:TRACKER_NUM]
    
    # Load reinitialization data if available
    reinit_data = load_reinitialization_data(REINIT_DATA_FILENAME) if os.path.exists(REINIT_DATA_FILENAME) else None
    print(f"Reinitialization data loaded: {reinit_data}")
    
    # Load label files from the label folder
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    label_folder_path = os.path.join(absolute_path, LABEL_FOLDER)
    label_files = sorted(os.listdir(label_folder_path))
    
    label_container = []
    for file_name in label_files:
        with open(os.path.join(label_folder_path, file_name), 'r') as f:
            lines = f.read().splitlines()
            # Flatten and convert each value to float
            label_container.append([float(val) for sub in lines for val in sub.split()])
    
    # Open video file
    video = cv2.VideoCapture(VIDEO_NAME)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read first frame.")
        return
    
    # Convert labels to absolute coordinates
    label_container_abs = [convert_to_absolute(coords, frame.shape) for coords in label_container]
    first_location = label_container_abs.pop(0)
    
    # Initialize trackers and Kalman filters
    trackers, kalman_filters = initialize_trackers(frame, first_location)
    
    # Data collections for later analysis
    tracker_positions = []
    failed_trackers = []
    tracking_box_size = []
    centroid_collection = []
    position_estimates_collection = []
    position_covariances_collection = []
    data_distance_collection = []
    rmse_collection = defaultdict(list)
    ground_truth_centroids_collection = []
    rmse_over_time = []
    iou_collection = []
    is_inside_collection = []
    reinit_data_new = []
    
    # Video writer if needed
    if OUTPUT_WRITE_VIDEO:
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 7
        output_video = cv2.VideoWriter(OUTPUT_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 0, 128), (255, 165, 0)]
    
    frame_index = 0
    user_break = False
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_copy = copy.deepcopy(frame)
        
        # Check for ROI reinitialization request
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('x'), ord('X')]:
            print("User pressed 'x'. Select a new ROI.")
            roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            if roi:
                print(f"Selected ROI: {roi}")
                selected = select_trackers_to_reinitialize(tracker_type_name)
                if selected:
                    for idx in selected:
                        print(f"Reinitializing {tracker_type_name[idx]}")
                        trackers[idx] = create_tracker(idx)
                        trackers[idx].init(frame, roi)
                        reinit_data_new.append({
                            'Frame': frame_index,
                            'Tracker': tracker_type_name[idx],
                            'x': int(roi[0]),
                            'y': int(roi[1]),
                            'w': int(roi[2]),
                            'h': int(roi[3])
                        })
                else:
                    print("No trackers selected for reinitialization.")
        if key == 27:  # Esc key
            user_break = True
            break
        
        # Auto reinitialize based on loaded reinit data
        if reinit_data:
            auto_reinitialize_trackers(frame_index, reinit_data, trackers, tracker_type_name, frame)
        
        # Initialize per-frame collections
        tracker_positions.append([])
        failed_trackers.append([])
        tracking_box_size.append([])
        centroid_collection.append([])
        position_estimates_collection.append([])
        position_covariances_collection.append([])
        
        # Process each tracker
        for j in range(TRACKER_NUM):
            success, box = trackers[j].update(frame_copy)
            if success:
                x, y, w, h = [int(v) for v in box]
                centroid_x = int(x + w / 2)
                centroid_y = int(y + h / 2)
                tracker_positions[frame_index].append([centroid_x, centroid_y])
                tracking_box_size[frame_index].append([w, h])
                
                measurement = np.array([[np.float32(centroid_x)],
                                        [np.float32(centroid_y)]])
                kalman_filters[j].correct(measurement)
                prediction = kalman_filters[j].predict()
                pt = (int(prediction[0]), int(prediction[1]))
                
                if SWITCH == 'kalman':
                    centroid_collection[frame_index].append({
                        'Tracker': j,
                        'x': int(prediction[0]),
                        'y': int(prediction[1])
                    })
                    cv2.circle(frame, pt, 5, colors[j], -1)
                    cv2.putText(frame, f'kalman_{tracker_type_name[j]}', (pt[0], pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 2)
                else:
                    centroid_collection[frame_index].append({
                        'Tracker': j,
                        'x': centroid_x,
                        'y': centroid_y
                    })
                    iou_val = compute_iou([x, y, w, h], label_container_abs[frame_index])
                    iou_collection.append({'Frame': frame_index, 'Tracker': tracker_type_name[j], 'IoU': iou_val})
                    cv2.circle(frame, (centroid_x, centroid_y), 5, colors[j], -1)
                    cv2.putText(frame, f'{tracker_type_name[j]}', (centroid_x, centroid_y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 2)
                
                # Store Kalman filter outputs
                pos_est = kalman_filters[j].statePost[:2]
                pos_cov = kalman_filters[j].errorCovPost[:2, :2]
                position_estimates_collection[frame_index].append(pos_est)
                position_covariances_collection[frame_index].append(pos_cov)
            else:
                failed_trackers[frame_index].append(j)
        
        # Process ground truth label for this frame
        ground_truth = label_container_abs[frame_index]
        gt_centroid = (int(ground_truth[0] + ground_truth[2] / 2),
                       int(ground_truth[1] + ground_truth[3] / 2))
        cv2.putText(frame, 'Real Fish Location', (ground_truth[0], ground_truth[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.rectangle(frame, (ground_truth[0], ground_truth[1]),
                      (ground_truth[0]+ground_truth[2], ground_truth[1]+ground_truth[3]),
                      (0, 0, 0), 2)
        cv2.circle(frame, gt_centroid, 7, (0, 0, 0), -1)
        ground_truth_centroids_collection.append({'Frame': frame_index, 'Centroid': list(gt_centroid)})
        
        # Fusion using covariance intersection (if any estimates exist)
        if position_estimates_collection[frame_index]:
            filt_est, filt_cov = filter_outliers(
                position_estimates_collection[frame_index],
                position_covariances_collection[frame_index]
            )
            if filt_est:
                fused_pos, fused_cov = multi_covariance_intersection(filt_est, filt_cov)
                x_fuse, y_fuse = int(fused_pos[0]), int(fused_pos[1])
                cv2.circle(frame, (x_fuse, y_fuse), 7, (255, 255, 255), -1)
                cv2.putText(frame, 'Fused location', (x_fuse, y_fuse-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                fuse_distance = calculate_distance([x_fuse, y_fuse], list(gt_centroid))
                data_distance_collection.append({
                    'Frame': frame_index,
                    'Tracker': 'Fused',
                    'Distance': fuse_distance
                })
                rmse_collection['Fuse'].append({'Frame': frame_index, 'Centroid': [x_fuse, y_fuse]})
                inside = is_centroid_overlapping(ground_truth, x_fuse, y_fuse)
                is_inside_collection.append({'Frame': frame_index, 'Tracker': 'Fused', 'Is_inside': inside})
            else:
                print("No valid trackers for covariance intersection.")
        else:
            print("No position estimates available for fusion.")
        
        # Draw ellipse based on covariance of position estimates
        if position_estimates_collection[frame_index]:
            estimates_arr = np.array([pe.flatten() for pe in position_estimates_collection[frame_index]])
            median_position = np.median(estimates_arr, axis=0)
            cov_matrix = np.cov(estimates_arr, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            axis_length1 = 2 * np.sqrt(eigenvalues[0]) * 2
            axis_length2 = 2 * np.sqrt(eigenvalues[1]) * 2
            center = (int(median_position[0]), int(median_position[1]))
            cv2.ellipse(frame, center, (int(axis_length1), int(axis_length2)),
                        angle, 0, 360, (255, 255, 255), 2)
        
        # Draw bounding box from mean tracker box size (if available)
        tracker_box_arr = np.array(tracking_box_size[frame_index])
        if tracker_box_arr.size:
            mean_box = np.mean(tracker_box_arr, axis=0).astype(int)
            x_min = int(x_fuse - mean_box[0] // 2)
            y_min = int(y_fuse - mean_box[1] // 2)
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + mean_box[0], y_min + mean_box[1]),
                          (255, 255, 255), 2)
            cv2.putText(frame, 'fused_box_from_mean_size', (x_min, y_min-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            iou_val = compute_iou([x_min, y_min, mean_box[0], mean_box[1]], ground_truth)
            iou_collection.append({'Frame': frame_index, 'Tracker': 'Fused', 'IoU': iou_val})
        
        # Process per-tracker distance and inside status
        for tracker_data in centroid_collection[frame_index]:
            dist = calculate_distance([tracker_data['x'], tracker_data['y']], list(gt_centroid))
            data_distance_collection.append({
                'Frame': frame_index,
                'Tracker': tracker_type_name[tracker_data['Tracker']],
                'Distance': dist
            })
            rmse_collection[tracker_type_name[tracker_data['Tracker']]].append({
                'Frame': frame_index,
                'Centroid': [tracker_data['x'], tracker_data['y']]
            })
            inside = is_centroid_overlapping(ground_truth, tracker_data['x'], tracker_data['y'])
            is_inside_collection.append({
                'Frame': frame_index,
                'Tracker': tracker_type_name[tracker_data['Tracker']],
                'Is_inside': inside
            })
        
        # Reinitialize any failed trackers
        for j in failed_trackers[frame_index]:
            print(f"Tracker {tracker_type_name[j]} failed, reinitializing...")
            trackers[j] = create_tracker(j)
            trackers[j].init(frame_copy, (x_min, y_min, mean_box[0], mean_box[1]))
            kalman_filters[j].statePre = np.array([[x_min + mean_box[0] / 2],
                                                   [y_min + mean_box[1] / 2],
                                                   [0],
                                                   [0]], np.float32)
            kalman_filters[j].statePost = kalman_filters[j].statePre
        
        # Update RMSE over time per tracker
        for tracker_name, rmse_data in rmse_collection.items():
            rmse_val = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
            rmse_over_time.append({'Frame': frame_index, 'Tracker': tracker_name, 'Rmse': rmse_val})
        
        if OUTPUT_WRITE_VIDEO:
            output_video.write(frame)
        
        cv2.imshow('Frame', frame)
        frame_index += 1

    cv2.destroyAllWindows()
    if OUTPUT_WRITE_VIDEO:
        output_video.release()
    video.release()
    
    # Save or display the collected data
    if not user_break and COLLECT_DATA:
        save_distance_data_to_csv(os.path.join(DATA_FOLDER, f'distance_data_{KAL_OR_TRACKER}.csv'),
                                  data_distance_collection)
        with open(os.path.join(DATA_FOLDER, f'is_inside_{KAL_OR_TRACKER}.csv'),
                  mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Is_inside'])
            writer.writeheader()
            writer.writerows(is_inside_collection)
        
        rmse_calculated_collection = []
        for tracker_name, rmse_data in rmse_collection.items():
            rmse_val = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
            rmse_calculated_collection.append({'Tracker': tracker_name, 'RMSE': rmse_val})
        
        with open(os.path.join(DATA_FOLDER, f'rmse_{KAL_OR_TRACKER}.csv'),
                  mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Tracker', 'RMSE'])
            writer.writeheader()
            writer.writerows(rmse_calculated_collection)
        
        with open(os.path.join(DATA_FOLDER, f'rmse_over_time_{KAL_OR_TRACKER}.csv'),
                  mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'Rmse'])
            writer.writeheader()
            writer.writerows(rmse_over_time)
        
        if SWITCH == 'tracker':
            with open(os.path.join(DATA_FOLDER, f'iou_{KAL_OR_TRACKER}.csv'),
                      mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Frame', 'Tracker', 'IoU'])
                writer.writeheader()
                writer.writerows(iou_collection)
        print("Data saved to CSV files.")
    else:
        print("Data not saved to CSV files.")
        # Optionally, display the results using pandas DataFrames
        df_distance = pd.DataFrame(data_distance_collection)
        df_inside = pd.DataFrame(is_inside_collection)
        rmse_calculated_collection = []
        for tracker_name, rmse_data in rmse_collection.items():
            rmse_val = compute_frame_rmse(rmse_data, ground_truth_centroids_collection)
            rmse_calculated_collection.append({'Tracker': tracker_name, 'RMSE': rmse_val})
        df_rmse = pd.DataFrame(rmse_calculated_collection)
        df_iou = pd.DataFrame(iou_collection)
        df_reinit = pd.DataFrame(reinit_data)
        
        tracker_counts = df_inside.groupby('Tracker')['Is_inside'].value_counts().sort_values(ascending=False)
        max_distance = df_distance.groupby('Tracker')['Distance'].max().sort_values(ascending=False)
        min_distance = df_distance.groupby('Tracker')['Distance'].min().sort_values()
        average_distance = df_distance.groupby('Tracker')['Distance'].mean().sort_values()
        rmse_df = df_rmse.sort_values(by='RMSE')
        average_iou = df_iou.groupby('Tracker')['IoU'].mean().sort_values(ascending=False)
        
        total_tracker_counts = df_inside.groupby('Tracker')['Is_inside'].count()
        failed_counts = len(label_container_abs) - total_tracker_counts
        print("================================================================================================")
        print(f"Max Distance:\n{max_distance}")
        print("-" * 50)
        print(f"Min Distance:\n{min_distance}")
        print("-" * 50)
        print(f"Average Distance:\n{average_distance}")
        print("-" * 50)
        print(f"RMSE:\n{rmse_df}")
        print("-" * 50)
        print(f"is_inside:\n{tracker_counts}")
        print("-" * 50)
        print(f"Average IoU:\n{average_iou}")
        print("-" * 50)
        print(f"Reinitialization Count:\n{df_reinit['Tracker'].value_counts(ascending=False)}")
        print("-" * 50)
        print(f"Failed Counts:\n{failed_counts.sort_values(ascending=False)}")

if __name__ == '__main__':
    main()
