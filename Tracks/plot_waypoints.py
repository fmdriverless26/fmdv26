import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cone_file = 'small_track.csv'
waypoints_file = 'small_track_waypoints.csv'

cones_df = pd.read_csv(cone_file)
waypoints_df = pd.read_csv(waypoints_file, header=None, names=['x', 'y'])

yellow_cones = cones_df[cones_df['tag'] == 'yellow']
blue_cones = cones_df[cones_df['tag'] == 'blue']
big_orange_cones = cones_df[cones_df['tag'] == 'big_orange']
car_start_cones = cones_df[cones_df['tag'] == 'car_start']

plt.figure(figsize=(12, 10))

plt.scatter(yellow_cones['x'], yellow_cones['y'], 
            color='yellow', s=50, edgecolor='black', 
            label='Yellow Cones', alpha=0.8)

plt.scatter(blue_cones['x'], blue_cones['y'], 
            color='blue', s=50, edgecolor='black', 
            label='Blue Cones', alpha=0.8)

plt.scatter(big_orange_cones['x'], big_orange_cones['y'], 
            color='orange', s=60, edgecolor='black', 
            label='Big Orange Cones', alpha=0.9, marker='s')

plt.scatter(car_start_cones['x'], car_start_cones['y'], 
            color='red', s=100, edgecolor='black', 
            label='Car Start', alpha=0.9, marker='*')

plt.scatter(waypoints_df['x'], waypoints_df['y'], 
            color='green', s=40, edgecolor='black', 
            label='Waypoints', alpha=0.7, marker='o')

plt.title('Cones and Waypoints Visualization', fontsize=16, fontweight='bold')
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

plt.legend(loc='best', fontsize=10)

plt.grid(True, alpha=0.3, linestyle='--')

plt.axis('equal')

plt.figtext(0.02, 0.02, 
            f"Total Cones: {len(cones_df)}\n"
            f"Yellow: {len(yellow_cones)}, Blue: {len(blue_cones)}\n"
            f"Big Orange: {len(big_orange_cones)}, Waypoints: {len(waypoints_df)}",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

if len(waypoints_df) > 1:
    plt.plot(waypoints_df['x'], waypoints_df['y'], 
             color='green', linestyle='-', linewidth=1, alpha=0.4, label='Waypoint Path')

plt.tight_layout()
plt.show()