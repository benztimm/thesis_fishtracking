import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Define options
OPTION = 'kalman'
VERSION = 'thesis'
TITLE = '7482_F3'
# Load the distance CSV file
df = pd.read_csv(f'distance_data_{OPTION}_{VERSION}.csv')  # Replace with your actual filename

# Pivot the DataFrame so that each tracker becomes a separate column
pivot_df = df.pivot(index='Frame', columns='Tracker', values='Distance')

# Create the plot
plt.figure(figsize=(12, 7))
for tracker in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[tracker], label=tracker, linewidth=2)

# Customize the plot
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.title(f'Distance Over Frames for Each {OPTION.capitalize()} {TITLE}')
plt.legend(title=OPTION.capitalize(), loc='upper right')
plt.grid(True)
plt.tight_layout()

# Save or show the plot
plt.savefig(f'distance_over_time_{OPTION}_{VERSION}_{TITLE}.png')
plt.show()
