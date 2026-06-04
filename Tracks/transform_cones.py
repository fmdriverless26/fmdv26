#!/usr/bin/env python3
import pandas as pd
import numpy as np

INPUT_FILE = "/home/aarav/WAYPOINTS/small_track.csv"
OUTPUT_FILE = "/home/aarav/modified_eufs_sim_v2/eufs-master/install/map_lib/share/map_lib/maps/competitions/FSUK/2023/trackdrive.csv"

track = pd.read_csv(INPUT_FILE)

# Fix tags: big_orange -> orange
track["tag"] = track["tag"].replace({"big_orange": "orange"})

# Fix direction: if all zeros, make sequential
if (track["direction"] == 0).all():
    track["direction"] = range(len(track))

# --- ROTATE MAP SO CAR FACES FRONT ORANGE PAIR ---

orange_cones = track[track["tag"] == "orange"]
car_start = track[track["tag"] == "car_start"]

if len(orange_cones) == 4 and len(car_start) == 1:
    car_x = car_start.iloc[0]["x"]
    car_y = car_start.iloc[0]["y"]
    
    coords = orange_cones[["x", "y"]].values
    
    # Find 2 cones closest to car (rear pair)
    dists = []
    for i, (x, y) in enumerate(coords):
        dist = np.hypot(x - car_x, y - car_y)
        dists.append((dist, i))
    dists.sort()
    
    rear_indices = {dists[0][1], dists[1][1]}
    front_indices = [i for i in range(4) if i not in rear_indices]
    
    # Midpoint of front pair
    f1, f2 = front_indices
    mid_x = (coords[f1][0] + coords[f2][0]) / 2
    mid_y = (coords[f1][1] + coords[f2][1]) / 2
    
    # Angle from car to front midpoint
    dx = mid_x - car_x
    dy = mid_y - car_y
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    print(f"Front midpoint: ({mid_x:.2f}, {mid_y:.2f})")
    print(f"Angle to front pair: {angle_deg:.1f}°")
    
    # Rotate ALL points around car_start by -angle_rad
    # This makes the front pair directly ahead (+X)
    for idx in track.index:
        px = track.at[idx, "x"] - car_x
        py = track.at[idx, "y"] - car_y
        
        # Rotate by -angle_rad
        new_x = px * np.cos(-angle_rad) - py * np.sin(-angle_rad)
        new_y = px * np.sin(-angle_rad) + py * np.cos(-angle_rad)
        
        track.at[idx, "x"] = new_x
        track.at[idx, "y"] = new_y
    
    # car_start is now at (0,0)
    track.loc[track["tag"] == "car_start", "direction"] = 0
    
    print(f"Map rotated by {-angle_deg:.1f}°")
    print(f"car_start now at (0,0) facing 0° (+X)")
    print(f"Front orange pair should be directly ahead")

# Save
track.to_csv(OUTPUT_FILE, index=False)
print(f"\nFile saved to {OUTPUT_FILE}")
print("EUFS Sim 2 ready to launch!")
