import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# 1. Load and Preprocess Data
current_folder = os.getcwd()
excel_list = os.path.join(os.getcwd(),"data_operasi_reaktor")
file_path = os.path.join(excel_list,"data_download_practice1_14_01_2022.xlsx")

df = pd.read_excel(file_path, sheet_name="Download Transaksi", header=1)
df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
df['Seconds'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()

# 2. Segment Detection (Automatic)
df['diff_pos'] = df['Regulator Rod [%]'].diff()
df_moving = df[df['diff_pos'] != 0].copy()
df_moving['segment_id'] = (df_moving.index.to_series().diff() > 1).cumsum()

# 3. Analyze segments and collect points for visualization
segments_info = []
df['Segment_Velocity'] = 0.0

for seg_id, group in df_moving.groupby('segment_id'):
    # We filter segments that have a significant duration to avoid jitter
    if len(group) >= 5: 
        t_start, t_end = group['Seconds'].iloc[0], group['Seconds'].iloc[-1]
        y_start, y_end = group['Regulator Rod [%]'].iloc[0], group['Regulator Rod [%]'].iloc[-1]
        
        duration = t_end - t_start
        if duration > 1.0: # Minimum 1 second movement
            velocity = (y_end - y_start) / duration
            df.loc[(df['Seconds'] >= t_start) & (df['Seconds'] <= t_end), 'Segment_Velocity'] = velocity
            
            segments_info.append({
                't_start': t_start, 't_end': t_end,
                'y_start': y_start, 'y_end': y_end,
                'velocity': velocity, 'id': seg_id
            })

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

# TOP PLOT: Position with Slope Indicators
ax1.plot(df['Seconds'], df['Regulator Rod [%]'], color='lightgray', alpha=0.5, label='Data Mentah (Posisi)')
ax1.set_ylabel('Posisi (%)', fontsize=12)
ax1.set_title('Titik-Titik Acuan Perhitungan Slope (Kecepatan)', fontsize=14)

# Plotting the points and trendlines for each detected segment
for seg in segments_info:
    # Highlight start and end points
    ax1.scatter([seg['t_start'], seg['t_end']], [seg['y_start'], seg['y_end']], 
                color='red', zorder=5, s=40, edgecolors='black')
    # Draw a line connecting the points to show the slope
    ax1.plot([seg['t_start'], seg['t_end']], [seg['y_start'], seg['y_end']], 
             color='red', linestyle='--', linewidth=2)
    # Label the velocity on the graph
    ax1.text(seg['t_start'], max(seg['y_start'], seg['y_end']) + 2, 
             f"v={seg['velocity']:.2f}", color='red', fontsize=9, fontweight='bold')

ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# BOTTOM PLOT: Velocity (Step Graph)
ax2.plot(df['Seconds'], df['Segment_Velocity'], color='green', label='Kecepatan Berdasarkan Slope (%/s)')
ax2.fill_between(df['Seconds'], df['Segment_Velocity'], color='green', alpha=0.1)
ax2.set_xlabel('Waktu (detik)', fontsize=12)
ax2.set_ylabel('Kecepatan (%/detik)', fontsize=12)
ax2.set_title('Distribusi Kecepatan per Segmen', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('grafik_titik_slope.png')

# Output a sample of points for the user to see in text
print("Sampel Titik Perhitungan:")
for i, seg in enumerate(segments_info[:5]):
    print(f"Segmen {i+1}: Start({seg['t_start']:.2f}s, {seg['y_start']}%) -> End({seg['t_end']:.2f}s, {seg['y_end']}%) | Slope: {seg['velocity']:.4f}")