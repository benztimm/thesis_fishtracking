import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iou_1.csv')

# Pivot the DataFrame so that each tracker becomes a separate column with rows indexed by Frame.
pivot_df = df.pivot(index='Frame', columns='Tracker', values='IoU')

# Create a line plot for each tracker.
plt.figure(figsize=(10, 6))
for tracker in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[tracker], label=tracker)
    
    

plt.xlabel('Frame')
plt.ylabel('IOU')
plt.title(f'IoU Over Frames for Each Tracker Data_7482_F3')
plt.legend(title='tracker')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'iou_overtime.png')
plt.show()
