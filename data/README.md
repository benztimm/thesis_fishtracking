# 📊 Tracker Evaluation and Visualization

This directory contains evaluation logs and visualization tools for comparing tracking performance between raw outputs and Kalman-enhanced fusion on a specific underwater video dataset.

---

## 🧩 Contents

| File                           | Description                                                             |
|--------------------------------|-------------------------------------------------------------------------|
| `evaluate.py`                 | Prints summary statistics for all trackers and fused output             |
| `plot_distance_overtime.py`   | Line plot of Euclidean distance over time for each tracker              |
| `plot_rmse_overtime.py`       | Line plot of RMSE per frame per tracker                                 |
| `plot_iou_overtime.py`        | Line plot of IoU per frame for fused tracker (or optionally all)        |
| `*.csv`                       | Evaluation results generated from the main tracking pipeline            |

---

## 📌 Evaluate Summary Statistics

```
python evaluate.py
```

This script will:
- Load metrics from both `tracker` and `kalman` versions
- Print:
  - RMSE
  - Max/Min/Avg distance
  - Inside ratio
  - Reinitialization count
  - Failure count
  - Mean IoU per tracker

---

### 📈 Plot Distance Over Time
```
python plot_distance_overtime.py
```
Generates `distance_over_time_<option>_<version>_<title>.png`, a graph showing per-frame Euclidean distance.

---

## 📉 Plot RMSE Over Time
```
python plot_rmse_overtime.py
```
Generates `rmse_over_time_<option>_<version>_<title>.png`, a graph showing root mean square error over frames for each tracker.

---

### 🧠 Plot IoU Over Time
```
python plot_iou_overtime.py
```
Generates `iou_over_time_<option>_<version>_<title>.png`, a graph showing intersection over union per frames for each tracker.