import pandas as pd

# Filenames for the datasets
file_groups = [
    {
        "reinit": "../reinitialization_data/reinitialization_data_4.csv",
        "is_inside": "is_inside_kalman_4.csv",
        "distance": "distance_data_kalman_4.csv",
        "rmse": "rmse_kalman_4.csv",
    },
    {
        "reinit": "../reinitialization_data/reinitialization_data_4.csv",
        "is_inside": "is_inside_kalman_4_2.csv",
        "distance": "distance_data_kalman_4_2.csv",
        "rmse": "rmse_kalman_4_2.csv",
    },
    {
        "reinit": "../reinitialization_data/reinitialization_data_4.csv",
        "is_inside": "is_inside_kalman_4_3.csv",
        "distance": "distance_data_kalman_4_3.csv",
        "rmse": "rmse_kalman_4_3.csv",
    }
    
]

results = []

total_frames = 446  # Update this value if the total number of frames changes

for i, files in enumerate(file_groups, start=1):
    # Load data
    
    df_reinit = pd.read_csv(files["reinit"])
    df_is_inside = pd.read_csv(files["is_inside"])
    print(files["is_inside"])
    df_data = pd.read_csv(files["distance"])
    df_rmse = pd.read_csv(files["rmse"])

    # Extract name from the filename
    if 'kalman' in files["is_inside"]:
        name = files["is_inside"][files["is_inside"].index('kalman'):].split('.')[0]
    elif 'tracker' in files["is_inside"]:
        name = files["is_inside"][files["is_inside"].index('tracker'):].split('.')[0]
    else:
        name = files["is_inside"].split('.')[0]
    # Perform calculations
    tracker_counts = df_is_inside.groupby('Tracker')['Is_inside'].value_counts().sort_values(ascending=False)
    reinit_count = df_reinit['Tracker'].value_counts()
    max_distance = df_data.groupby('Tracker')['Distance'].max().sort_values()
    min_distance = df_data.groupby('Tracker')['Distance'].min().sort_values()
    average_distance = df_data.groupby('Tracker')['Distance'].mean().sort_values()

    # Calculate failed counts
    total_tracker_counts = df_is_inside.groupby('Tracker')['Is_inside'].count()
    failed_counts = total_frames - total_tracker_counts

    # Store results in a dictionary
    results.append({
        "max_distance": max_distance,
        "min_distance": min_distance,
        "average_distance": average_distance,
        "tracker_counts": tracker_counts,
        "reinit_count": reinit_count,
        "failed_counts": failed_counts.sort_values(ascending=False),
        'rmse': df_rmse.sort_values(by='RMSE'),
        'name': name
    })

# Display results
for i, result in enumerate(results, start=1):
    print(f"==== {result['name']} ====")
    print(f"Max Distance:\n{result['max_distance']}")
    print("-" * 50)
    print(f"Min Distance:\n{result['min_distance']}")
    print("-" * 50)
    print(f"Average Distance:\n{result['average_distance']}")
    print("-" * 50)
    print(f"RMSE:\n{result['rmse']}")
    print("-" * 50)
    print(f"Tracker Counts:\n{result['tracker_counts']}")
    print("-" * 50)
    print(f"Failed Counts:\n{result['failed_counts']}")
    print("-" * 50)
    print(f"Reinitialization Count:\n{result['reinit_count']}")
    print("=" * 50)
    print("")
    print("")
    
