import pandas as pd
import matplotlib.pyplot as plt

# Define options
OPTION = 'tracker'
VERSION = 'thesis'
TITLE = '7482_F3'
# Read the IoU CSV file
df = pd.read_csv(f'iou_tracker_{VERSION}.csv')  # Replace with your actual filename

# Pivot the DataFrame so that each tracker becomes a separate column
pivot_df = df.pivot(index='Frame', columns='Tracker', values='IoU')

# Create a line plot for each tracker
plt.figure(figsize=(12, 7))
for tracker in pivot_df.columns:
    if tracker != 'Fused' and OPTION == 'kalman' :
        continue
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
