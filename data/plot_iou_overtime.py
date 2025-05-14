import pandas as pd
import matplotlib.pyplot as plt

# Define options
OPTION = 'kalman'
VERSION = 'thesis'
TITLE = '9866_acanthopagrus_palmaris'

# Read the IoU CSV file
df = pd.read_csv(f'iou_{OPTION}_{VERSION}.csv')  # Replace with your actual filename

# Pivot the DataFrame so that each tracker becomes a separate column
pivot_df = df.pivot(index='Frame', columns='Tracker', values='IoU')

# Create a line plot for each tracker
plt.figure(figsize=(12, 7))
for tracker in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[tracker], label=tracker, linewidth=2)

# Labeling the plot
plt.xlabel('Frame')
plt.ylabel('IoU')
plt.title(f'IoU Over Frames for Each {OPTION.capitalize()} {TITLE}')
plt.legend(title=OPTION.capitalize(), loc='lower right')
plt.grid(True)
plt.tight_layout()

# Show or save plot
plt.savefig(f'iou_over_time_{OPTION}_{VERSION}_{TITLE}.png')
plt.show()
