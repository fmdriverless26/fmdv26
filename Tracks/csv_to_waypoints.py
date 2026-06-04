import pandas as pd
import numpy as np

def generate_waypoints(input_file, output_file):
    try:
        df = pd.read_csv(input_file)

        yellows = df[df['tag'] == 'yellow'][['x', 'y']].values
        blues = df[df['tag'] == 'blue'][['x', 'y']].values
        
        if len(yellows) == 0 or len(blues) == 0:
            print(f"Skipping {input_file}: Missing yellow/blue cones.")
            return

        waypoints = []

        for y in yellows:
            distances = np.linalg.norm(blues - y, axis=1)
            closest_blue = blues[np.argmin(distances)]
            waypoints.append((y + closest_blue) / 2)
            
        for b in blues:
            distances = np.linalg.norm(yellows - b, axis=1)
            closest_yellow = yellows[np.argmin(distances)]
            waypoints.append((b + closest_yellow) / 2)
            
        wp_df = pd.DataFrame(waypoints, columns=['x', 'y']).round(4).drop_duplicates()
        
        sorted_waypoints = []
        remaining = wp_df.values.tolist()
        
        if remaining:
            current = remaining.pop(0)
            sorted_waypoints.append(current)
            while remaining:
                distances = np.linalg.norm(np.array(remaining) - current, axis=1)
                closest_idx = np.argmin(distances)
                current = remaining.pop(closest_idx)
                sorted_waypoints.append(current)

        pd.DataFrame(sorted_waypoints).to_csv(output_file, index=False, header=False)
        print(f"Successfully created {output_file}")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure the file is in the directory.")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

files_to_process = [
    ('rand.csv', 'rand_waypoints.csv'),
    ('peanut.csv', 'peanut_waypoints.csv'),
    ('its_a_mess.csv', 'its_a_mess_waypoints.csv'),
    ('boa_constrictor.csv', 'boa_constrictor_waypoints.csv')
]

for input_f, output_f in files_to_process:
    generate_waypoints(input_f, output_f)