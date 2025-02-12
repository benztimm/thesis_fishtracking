import pandas as pd
import matplotlib.pyplot as plt

option = 'tracker'
# Read the CSV file; replace 'rmse_data.csv' with your actual filename.
df = pd.read_csv(f'rmse_over_time_{option}_4.csv')

# Pivot the DataFrame so that each tracker becomes a separate column with rows indexed by Frame.
pivot_df = df.pivot(index='Frame', columns='Tracker', values='Rmse')

# Create a line plot for each tracker.
plt.figure(figsize=(10, 6))
for tracker in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[tracker], label=tracker)
    
    

plt.xlabel('Frame')
plt.ylabel('RMSE')
plt.title(f'RMSE Over Frames for Each {option}')
plt.legend(title=option)
plt.grid(True)
plt.tight_layout()
plt.savefig(f'rmse_overtime_{option}.png')
plt.show()
