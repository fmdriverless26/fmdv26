"""
Passive Delaunay Mapper (Terminal 2) — Full Path & Section Exporter
===================================================================
Listens to Odometry and Cone topics to passively build a 
Delaunay triangulation map.

FIXES APPLIED:
1. Dual-Spline Generation: Uses s=1.0 for a smooth circle, and s=0.0 for rigid gate arcs.
2. Trimmed End Path: The exit path now stops ~4 orange cones deep instead of the end of the map.
3. Distinct Color Coding: Every section of the path has a unique plot style.
4. 8-File CSV Exporter: Saves start_path, end_path, circles, and 4 arcs to separate files.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import csv

# ROS 2 Messages
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance
import message_filters

# Delaunay & Splines
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev

# Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
CIRCLE_RADIUS = 9.125
CIRCLE_BAND = 1.0

# ---------------------------------------------------------------------------
# Math Helpers
# ---------------------------------------------------------------------------
def circumcircle_radius(a, b, c):
    A = np.linalg.norm(b - c)
    B = np.linalg.norm(a - c)
    C = np.linalg.norm(a - b)

    s = (A + B + C) / 2.0
    area = max(s * (s - A) * (s - B) * (s - C), 0.0)
    if area == 0: return np.inf
    return (A * B * C) / (4.0 * np.sqrt(area))

def create_straight_line(x1, y1, x2, y2, num_pts=50):
    """Generates a straight array of waypoints between two coordinates."""
    return np.linspace(x1, x2, num_pts), np.linspace(y1, y2, num_pts)

# ---------------------------------------------------------------------------
# Passive Delaunay Node
# ---------------------------------------------------------------------------
class PassiveDelaunayMapper(Node):
    def __init__(self):
        super().__init__('passive_delaunay_mapper')
        
        # ── Time-Synchronized Subscriptions ──────────────────────────────────
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/ground_truth/odom')
        self.cone_sub = message_filters.Subscriber(self, ConeArrayWithCovariance, '/ground_truth/cones')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.cone_sub], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

        # Vehicle State & Extracted Points
        self.x = self.y = self.yaw = 0.0
        self._last_x = 0.0
        self._last_cross_time = time.monotonic()
        
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        
        # ── Unified Rigid Cone Map ───────────────────────────────────────────
        self.mapped_cones: list = []
        
        # ── Path Data Storage (For Plotting & Saving) ────────────────────────
        self.right_circle_wpts: list = []  
        self.left_circle_wpts: list = []   
        self._visual_edges = []
        
        self.path_start_x, self.path_start_y = [], []
        self.path_end_x, self.path_end_y = [], []
        self.path_cright_x, self.path_cright_y = [], []
        self.path_cleft_x, self.path_cleft_y = [], []
        self.path_arcs = [([], []) for _ in range(4)]
        
        # Gate & State Tracking
        self.gate_center_x = None
        self.gate_center_y = None
        self.gate_established = False
        self.laps_completed = 0
        self.state = 'WAITING'

        self._setup_debug_plot()
        self.timer = self.create_timer(0.05, self.map_loop)
        self.get_logger().info("Passive Delaunay Mapper Started. Waiting for Cones...")

    def sync_callback(self, odom_msg, cone_msg):
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        # Lock in the very first odometry position as the Start Line
        if self.start_x is None:
            self.start_x = self.x
            self.start_y = self.y
            self.get_logger().info(f"Locked Start Position: X={self.start_x:.2f}, Y={self.start_y:.2f}")
            
        self.latest_cones = cone_msg
        self._process_new_cones()

    def _local_to_global(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        global_x = c * lx - s * ly + self.x
        global_y = s * lx + c * ly + self.y
        return global_x, global_y

    def _add_cone_rigidly(self, wx: float, wy: float, color: str):
        if self.gate_established:
            if math.hypot(wx - self.gate_center_x, wy - self.gate_center_y) > 25.0: return

        for ec in self.mapped_cones:
            if math.hypot(wx - ec[0], wy - ec[1]) < 1.0:
                if color == "Big Orange": ec[2] = "Big Orange"
                elif color == "Small Orange" and ec[2] != "Big Orange": ec[2] = "Small Orange"
                elif color == "Blue" and ec[2] not in ["Big Orange", "Small Orange", "Blue"]: ec[2] = "Blue"
                elif color == "Yellow" and ec[2] not in ["Big Orange", "Small Orange", "Yellow"]: ec[2] = "Yellow"
                return
        self.mapped_cones.append([wx, wy, color])

    def _process_new_cones(self):
        if not hasattr(self, 'latest_cones'): return
        msg = self.latest_cones
        for c in msg.blue_cones: self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Blue")
        for c in msg.yellow_cones: self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Yellow")
        for c in msg.big_orange_cones: self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Big Orange")
        try:
            for c in msg.orange_cones: self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Small Orange")
        except AttributeError: pass

    def _check_gate_crossing(self) -> bool:
        if not self.gate_established or self.gate_center_x is None:
            self._last_x = self.x
            return False
            
        crossed = (self._last_x > self.gate_center_x) and (self.x <= self.gate_center_x)
        self._last_x = self.x
        
        if crossed:
            now = time.monotonic()
            if now - self._last_cross_time > 2.0:
                self._last_cross_time = now
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Triangulation & Smart Geometric Knife
    # ──────────────────────────────────────────────────────────────────────

    def _run_delaunay(self):
        track_cones = [c for c in self.mapped_cones if c[2] in ["Blue", "Yellow", "Small Orange"]]
        if len(track_cones) < 3 or not self.gate_established: return
            
        try:
            pts = np.array([[c[0], c[1]] for c in track_cones])
            tri = Delaunay(pts)
            valid_edges = set()
            
            for simplex in tri.simplices:
                p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                if circumcircle_radius(p1, p2, p3) > 4.5 or max(np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p1-p3)) > 6.0:
                    continue
                
                for i in range(3):
                    a, b = simplex[i], simplex[(i+1)%3]
                    c1, c2 = track_cones[a], track_cones[b]
                    
                    dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
                    
                    if 2.0 < dist < 5.5:
                        if c1[2] != c2[2] or (c1[2] == "Small Orange" and c2[2] == "Small Orange"):
                            edge = tuple(sorted((a, b)))
                            valid_edges.add(edge)
                            
                            mid_x = (c1[0] + c2[0]) / 2.0
                            mid_y = (c1[1] + c2[1]) / 2.0
                            
                            if mid_x > self.gate_center_x + 11.0: continue
                                
                            y_diff = mid_y - self.gate_center_y
                            if y_diff > 0.5:
                                circle_cy = self.gate_center_y + CIRCLE_RADIUS
                                target    = self.right_circle_wpts
                            elif y_diff < -0.5:
                                circle_cy = self.gate_center_y - CIRCLE_RADIUS
                                target    = self.left_circle_wpts
                            else:
                                continue

                            if abs(math.hypot(mid_x - self.gate_center_x, mid_y - circle_cy) - CIRCLE_RADIUS) > CIRCLE_BAND:
                                continue

                            if not any(math.hypot(mid_x-ex, mid_y-ey) < 0.8 for ex, ey in target):
                                target.append([mid_x, mid_y])

            self._visual_edges = [((pts[e[0]][0], pts[e[0]][1]), (pts[e[1]][0], pts[e[1]][1])) for e in valid_edges]
                                   
        except Exception: pass

    # ──────────────────────────────────────────────────────────────────────
    # Spline Generation (Dual Smoothing Factors)
    # ──────────────────────────────────────────────────────────────────────

    def _get_spline_and_arcs(self, wpts_list, center_x, center_y, gate_pt):
        if len(wpts_list) < 4: return [], [], [], []
            
        clean_pts = [gate_pt]
        for p in wpts_list:
            if not np.allclose(p, gate_pt) and not any(math.hypot(p[0]-cp[0], p[1]-cp[1]) < 0.3 for cp in clean_pts):
                clean_pts.append(p)
        pts = np.array(clean_pts)
        
        angles = np.arctan2(pts[:, 1] - center_y, pts[:, 0] - center_x)
        sort_idx = np.argsort(angles)
        pts = pts[sort_idx]
        
        gate_idx = -1
        for i, p in enumerate(pts):
            if np.allclose(p, gate_pt):
                gate_idx = i
                break
                
        if gate_idx == -1: return [], [], [], []
        
        try:
            # 1. Smooth Spline for the main circular path (s=1.0)
            tck_smooth, u_smooth = splprep([pts[:, 0], pts[:, 1]], s=1.0, per=1)
            u_full = np.linspace(0, 1, 200)
            full_x, full_y = splev(u_full, tck_smooth)
            
            # 2. Rigid Spline specifically for the purple arcs (s=0.0)
            tck_rigid, u_rigid = splprep([pts[:, 0], pts[:, 1]], s=0.0, per=1)
            
            n = len(pts)
            idx_prev = (gate_idx - 1) % n
            idx_next = (gate_idx + 1) % n
            
            def get_arc(idxA, idxB):
                uA, uB = u_rigid[idxA], u_rigid[idxB]
                if uA > uB: u_vals = np.concatenate((np.linspace(uA, 1.0, 25), np.linspace(0.0, uB, 25)))
                else: u_vals = np.linspace(uA, uB, 50)
                # Ensure we evaluate the arc using the rigid tck
                return splev(u_vals, tck_rigid)
                
            arc1_x, arc1_y = get_arc(idx_prev, gate_idx)
            arc2_x, arc2_y = get_arc(gate_idx, idx_next)
            
            # Return smooth circle, and the rigid connecting arcs
            return full_x, full_y, (arc1_x, arc1_y), (arc2_x, arc2_y)
        except Exception:
            return [], [], [], []

    # ──────────────────────────────────────────────────────────────────────
    # Path Updating & End Coordinate Logic
    # ──────────────────────────────────────────────────────────────────────

    def _update_all_paths(self):
        """Generates all splines and straight lines and saves them to class variables."""
        if not self.gate_established: return
        gate_pt = [self.gate_center_x, self.gate_center_y]

        # 1. Start Path (Car Odom to Gate)
        if self.start_x is not None:
            self.path_start_x, self.path_start_y = create_straight_line(self.start_x, self.start_y, self.gate_center_x, self.gate_center_y)

        # 2. Trimmed End Path (Gate to ~4 orange cones deep)
        exit_cones = [c for c in self.mapped_cones if "Orange" in c[2] and c[0] > self.gate_center_x + 2.0]
        if len(exit_cones) >= 2:
            exit_cones.sort(key=lambda c: c[0])
            
            target_idx = min(7, len(exit_cones) - 1)
            
            self.end_x = exit_cones[target_idx][0]
            self.end_y = self.gate_center_y 
            
            self.path_end_x, self.path_end_y = create_straight_line(self.gate_center_x, self.gate_center_y, self.end_x, self.end_y)

        # 3. Circles and Arcs
        rx, ry, r_a1, r_a2 = self._get_spline_and_arcs(self.right_circle_wpts, self.gate_center_x, self.gate_center_y + CIRCLE_RADIUS, gate_pt)
        if len(rx) > 0: self.path_cright_x, self.path_cright_y = rx, ry
        if r_a1: self.path_arcs[0] = r_a1
        if r_a2: self.path_arcs[1] = r_a2

        lx, ly, l_a1, l_a2 = self._get_spline_and_arcs(self.left_circle_wpts, self.gate_center_x, self.gate_center_y - CIRCLE_RADIUS, gate_pt)
        if len(lx) > 0: self.path_cleft_x, self.path_cleft_y = lx, ly
        if l_a1: self.path_arcs[2] = l_a1
        if l_a2: self.path_arcs[3] = l_a2


    # ──────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────

    def map_loop(self):
        if not self.gate_established:
            big_oranges = [c for c in self.mapped_cones if c[2] == "Big Orange"]
            if len(big_oranges) >= 4:
                self.gate_center_x = sum(c[0] for c in big_oranges[:4]) / 4.0
                self.gate_center_y = sum(c[1] for c in big_oranges[:4]) / 4.0
                self.gate_established = True
                self.get_logger().info(f"Gate Locked at X: {self.gate_center_x:.2f}, Y: {self.gate_center_y:.2f}")

        if self._check_gate_crossing():
            self.laps_completed += 1
            self.get_logger().info(f"Gate Crossed! Lap Tracker: {self.laps_completed}")
            if self.laps_completed == 1: self.state = 'RIGHT_CIRCLES'
            elif self.laps_completed == 3: self.state = 'LEFT_CIRCLES'
            elif self.laps_completed == 5: 
                self.state = 'FINISHED'
                self._update_all_paths()
                self._save_waypoints()

        self._run_delaunay()
        self._update_all_paths()
        self._update_debug_plot()

    def _save_csv(self, filename, x_arr, y_arr):
        """Helper function to dump arrays into a CSV"""
        if len(x_arr) == 0: return
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in zip(x_arr, y_arr):
                writer.writerow([x, y])

    def _save_waypoints(self):
        try:
            self._save_csv('start_path.csv', self.path_start_x, self.path_start_y)
            self._save_csv('end_path.csv', self.path_end_x, self.path_end_y)
            self._save_csv('circle_right.csv', self.path_cright_x, self.path_cright_y)
            self._save_csv('circle_left.csv', self.path_cleft_x, self.path_cleft_y)
            self._save_csv('arc_1.csv', self.path_arcs[0][0], self.path_arcs[0][1])
            self._save_csv('arc_2.csv', self.path_arcs[1][0], self.path_arcs[1][1])
            self._save_csv('arc_3.csv', self.path_arcs[2][0], self.path_arcs[2][1])
            self._save_csv('arc_4.csv', self.path_arcs[3][0], self.path_arcs[3][1])
            self.get_logger().info("SUCCESS! All 8 sections saved correctly into separate CSV files.")
        except Exception as e:
            self.get_logger().error(f"Failed to save waypoints: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Debug Plotting
    # ──────────────────────────────────────────────────────────────────────
    def _setup_debug_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_title('Passive Delaunay Mapper (Full Path Exporter)')

        self._plot_triangles = LineCollection([], colors='cyan', linewidths=1.0, alpha=0.3, zorder=1)
        self.ax.add_collection(self._plot_triangles)
        
        # Paths
        self._plot_start_line, = self.ax.plot([], [], color='dodgerblue', linestyle='--', linewidth=3, zorder=5, label='Start Path')
        self._plot_end_line, = self.ax.plot([], [], color='crimson', linestyle='-', linewidth=3, zorder=5, label='End Path')
        self._plot_right_spline, = self.ax.plot([], [], 'c-', linewidth=2, zorder=2)
        self._plot_left_spline, = self.ax.plot([], [], 'c-', linewidth=2, zorder=2)
        
        self._plot_purple_arcs = []
        for _ in range(4):
            line, = self.ax.plot([], [], 'm-', linewidth=4, zorder=6, label='Gate Arc' if _==0 else "")
            self._plot_purple_arcs.append(line)
        
        self._plot_right_wpts = self.ax.scatter([], [], c='lime', s=35, label='Right Circle (Lime)', zorder=5)
        self._plot_left_wpts = self.ax.scatter([], [], c='green', s=35, label='Left Circle (Dark Green)', zorder=5)
        self._plot_gate_pt = self.ax.scatter([], [], c='red', s=100, marker='P', label='Gate Center', zorder=7)
        
        # Cones & Car
        self._plot_blue_cones = self.ax.scatter([], [], c='blue', s=20, marker='^', label='Blue cones', zorder=3)
        self._plot_yellow_cones = self.ax.scatter([], [], c='gold', s=20, marker='^', label='Yellow cones', zorder=3)
        self._plot_big_orange = self.ax.scatter([], [], c='darkorange', s=40, marker='s', label='Big Orange', zorder=4)
        self._plot_small_orange = self.ax.scatter([], [], c='orange', s=20, marker='^', label='Small Orange', zorder=4)
        self._plot_car, = self.ax.plot([], [], 'ro', markersize=6, label='Car', zorder=6)

        self.ax.legend(loc='upper right', fontsize=7)
        self.fig.canvas.draw()
        plt.show(block=False)
        self._plot_xlim = self._plot_ylim = None

    def _grow_view_bounds(self, xs, ys, margin=3.0):
        if not xs: return
        x0, x1 = float(np.min(xs)) - margin, float(np.max(xs)) + margin
        y0, y1 = float(np.min(ys)) - margin, float(np.max(ys)) + margin
        if self._plot_xlim is None: self._plot_xlim, self._plot_ylim = [x0, x1], [y0, y1]
        else:
            self._plot_xlim = [min(self._plot_xlim[0], x0), max(self._plot_xlim[1], x1)]
            self._plot_ylim = [min(self._plot_ylim[0], y0), max(self._plot_ylim[1], y1)]
        self.ax.set_xlim(self._plot_xlim); self.ax.set_ylim(self._plot_ylim)

    def _update_debug_plot(self):
        all_x, all_y = [self.x], [self.y]

        if self._visual_edges:
            self._plot_triangles.set_segments(self._visual_edges)
            
        if self.gate_established:
            self._plot_gate_pt.set_offsets(np.array([[self.gate_center_x, self.gate_center_y]]))

        # Draw lines/splines using the stored generated arrays
        self._plot_start_line.set_data(self.path_start_x, self.path_start_y)
        self._plot_end_line.set_data(self.path_end_x, self.path_end_y)
        self._plot_right_spline.set_data(self.path_cright_x, self.path_cright_y)
        self._plot_left_spline.set_data(self.path_cleft_x, self.path_cleft_y)

        for i in range(4):
            if len(self.path_arcs[i][0]) > 0:
                self._plot_purple_arcs[i].set_data(self.path_arcs[i][0], self.path_arcs[i][1])

        if self.right_circle_wpts:
            pts = np.array(self.right_circle_wpts)
            self._plot_right_wpts.set_offsets(pts)
            all_x += list(pts[:, 0]); all_y += list(pts[:, 1])
            
        if self.left_circle_wpts:
            pts = np.array(self.left_circle_wpts)
            self._plot_left_wpts.set_offsets(pts)
            all_x += list(pts[:, 0]); all_y += list(pts[:, 1])

        bx, by, yx, yy, box, boy, sox, soy = [], [], [], [], [], [], [], []
        for c in self.mapped_cones:
            if c[2] == "Blue": bx.append(c[0]); by.append(c[1])
            elif c[2] == "Yellow": yx.append(c[0]); yy.append(c[1])
            elif c[2] == "Big Orange": box.append(c[0]); boy.append(c[1])
            elif c[2] == "Small Orange": sox.append(c[0]); soy.append(c[1])

        if bx: self._plot_blue_cones.set_offsets(np.column_stack([bx, by]))
        if yx: self._plot_yellow_cones.set_offsets(np.column_stack([yx, yy]))
        if box: self._plot_big_orange.set_offsets(np.column_stack([box, boy]))
        if sox: self._plot_small_orange.set_offsets(np.column_stack([sox, soy]))

        all_x.extend(bx + yx + box + sox)
        all_y.extend(by + yy + boy + soy)

        self._plot_car.set_data([self.x], [self.y])
        self._grow_view_bounds(all_x, all_y)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def main():
    rclpy.init()
    node = PassiveDelaunayMapper()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
