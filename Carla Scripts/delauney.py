import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev


# -------------------------------
# Circumcircle radius (filter bad triangles)
# -------------------------------
def circumcircle_radius(a, b, c):
    A = np.linalg.norm(b - c)
    B = np.linalg.norm(a - c)
    C = np.linalg.norm(a - b)

    s = (A + B + C) / 2.0
    area = max(s * (s - A) * (s - B) * (s - C), 0.0)

    if area == 0:
        return np.inf

    area = np.sqrt(area)
    return (A * B * C) / (4.0 * area)


# -------------------------------
# Order points (nearest neighbor)
# -------------------------------
def order_points(points):
    if len(points) == 0:
        return points

    ordered = [points[0]]
    remaining = list(points[1:])

    while remaining:
        last = ordered[-1]
        distances = [np.linalg.norm(p - last) for p in remaining]
        idx = np.argmin(distances)
        ordered.append(remaining.pop(idx))

    return np.array(ordered)


# -------------------------------
# Create spline curve
# -------------------------------
def create_spline(points, smoothness=0.5):
    x = points[:, 0]
    y = points[:, 1]

    tck, u = splprep([x, y], s=smoothness)

    u_new = np.linspace(0, 1, 500)
    x_new, y_new = splev(u_new, tck)

    return x_new, y_new

def order_points_graph(points):
    if len(points) < 3:
        return points

    # Distance matrix
    dist = np.linalg.norm(points[:, None] - points[None, :], axis=2)

    # For each point, get 2 nearest neighbors
    neighbors = {}
    for i in range(len(points)):
        idx = np.argsort(dist[i])[1:3]  # skip self
        neighbors[i] = list(idx)

    # Start from point 0
    ordered = [0]
    visited = set(ordered)

    while len(ordered) < len(points):
        current = ordered[-1]

        # choose next unvisited neighbor
        next_candidates = [n for n in neighbors[current] if n not in visited]

        if next_candidates:
            nxt = next_candidates[0]
        else:
            # fallback: nearest unvisited point
            remaining = list(set(range(len(points))) - visited)
            dists = [dist[current][r] for r in remaining]
            nxt = remaining[np.argmin(dists)]

        ordered.append(nxt)
        visited.add(nxt)

    return points[ordered]

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def process_cones(csv_path):
    df = pd.read_csv(csv_path)

    # ❌ Remove unwanted cones
    df = df[~df['tag'].isin(['car_start', 'big_orange'])]

    points = df[['x', 'y']].values
    colors = df['tag'].values

    tri = Delaunay(points)

    all_edges = set()
    opposite_edges = set()

    # -------------------------------
    # TRIANGLES → EDGES
    # -------------------------------
    for simplex in tri.simplices:
        pts = points[simplex]
        cols = colors[simplex]

        # ❌ Remove large outer triangles
        R = circumcircle_radius(pts[0], pts[1], pts[2])
        if R > 4.0:   # 🔧 tune this
            continue

        # Extract edges
        for i in range(3):
            a = simplex[i]
            b = simplex[(i + 1) % 3]

            edge = tuple(sorted((a, b)))
            all_edges.add(edge)

            # 🌟 Opposite color edge
            if colors[a] != colors[b]:
                opposite_edges.add(edge)

    # -------------------------------
    # MIDPOINTS
    # -------------------------------
    midpoints = []
    for a, b in opposite_edges:
        mid = (points[a] + points[b]) / 2.0
        midpoints.append(mid)

    midpoints = np.array(midpoints)

    # -------------------------------
    # ORDER + SPLINE
    # -------------------------------
    ordered_midpoints = order_points_graph(midpoints)

    spline_x, spline_y = None, None
    if len(ordered_midpoints) > 3:
        spline_x, spline_y = create_spline(ordered_midpoints)

    # -------------------------------
    # PLOTTING
    # -------------------------------
    plt.figure(figsize=(10, 8))

    # cones
    for tag, group in df.groupby('tag'):
        color = 'yellow' if tag == 'yellow' else 'blue'
        plt.scatter(group['x'], group['y'], c=color, label=tag, s=50)

    # all edges (light)
    for a, b in all_edges:
        p1, p2 = points[a], points[b]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.3)

    # opposite edges (highlight)
    for a, b in opposite_edges:
        p1, p2 = points[a], points[b]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)

    # midpoints
    if len(midpoints) > 0:
        plt.scatter(midpoints[:, 0], midpoints[:, 1],
                    c='green', s=25, label='midpoints')

    # spline curve
    if spline_x is not None:
        plt.plot(spline_x, spline_y, 'g-', linewidth=3, label='spline path')

    plt.title("Delaunay → Midpoints → Smooth Spline Track")
    plt.legend()
    plt.axis('equal')
    plt.show()


# -------------------------------
# RUN
# -------------------------------
process_cones('/home/aarav/WAYPOINTS/small_track.csv')