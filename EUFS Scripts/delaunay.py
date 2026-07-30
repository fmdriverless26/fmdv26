"""
Passive Delaunay Mapper (Terminal 2) — Smart Geometric Knife
===================================================================
Listens to Odometry and Cone topics to passively build a 
Delaunay triangulation map.

FIXES APPLIED:
1. Smart Geometric Knife: Drops the entry straight without slicing the circles.
2. Delaunay Exclusion: Orange cones are completely ignored for triangulation.
3. Split Waypoints: Separated into right/left circle lists and deleted crossover.
4. Time-Sync: Perfect timestamp alignment prevents odometry latency smearing.
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

# Delaunay
from scipy.spatial import Delaunay

# Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Math Helpers
# ---------------------------------------------------------------------------
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

        # Vehicle State
        self.x = self.y = self.yaw = 0.0
        self._last_x = 0.0
        self._last_cross_time = time.monotonic()
        
        # ── Unified Rigid Cone Map ───────────────────────────────────────────
        self.mapped_cones: list = []
        
        # ── Two Independent Waypoint Sets ────────────────────────────────────
        self.right_circle_wpts: list = []  # Top circle
        self.left_circle_wpts: list = []   # Bottom circle
        
        self._visual_edges = []
        
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
        # Update perfect vehicle state
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y
        q = odom_msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        # Process cones immediately using frozen state
        self.latest_cones = cone_msg
        self._process_new_cones()

    def _local_to_global(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        global_x = c * lx - s * ly + self.x
        global_y = s * lx + c * ly + self.y
        return global_x, global_y

    def _add_cone_rigidly(self, wx: float, wy: float, color: str):
        if self.gate_established:
            if math.hypot(wx - self.gate_center_x, wy - self.gate_center_y) > 25.0:
                return

        for ec in self.mapped_cones:
            if math.hypot(wx - ec[0], wy - ec[1]) < 1.0:
                if color == "Big Orange":
                    ec[2] = "Big Orange"
                elif color == "Small Orange" and ec[2] != "Big Orange":
                    ec[2] = "Small Orange"
                elif color == "Blue" and ec[2] not in ["Big Orange", "Small Orange", "Blue"]:
                    ec[2] = "Blue"
                elif color == "Yellow" and ec[2] not in ["Big Orange", "Small Orange", "Yellow"]:
                    ec[2] = "Yellow"
                return
                
        self.mapped_cones.append([wx, wy, color])

    def _process_new_cones(self):
        if not hasattr(self, 'latest_cones'):
            return
            
        msg = self.latest_cones
        for c in msg.blue_cones: 
            self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Blue")
        for c in msg.yellow_cones: 
            self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Yellow")
        for c in msg.big_orange_cones:
            self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Big Orange")
        try:
            for c in msg.orange_cones:
                self._add_cone_rigidly(*self._local_to_global(c.point.x, c.point.y), "Small Orange")
        except AttributeError:
            pass

    def _check_gate_crossing(self) -> bool:
        if not self.gate_established or self.gate_center_x is None:
            self._last_x = self.x
            return False
            
        # Car drives in -X direction, so crossing is from > to <
        crossed = (self._last_x > self.gate_center_x) and (self.x <= self.gate_center_x)
        self._last_x = self.x
        
        if crossed:
            now = time.monotonic()
            if now - self._last_cross_time > 2.0:
                self._last_cross_time = now
                return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Clean Triangulation & Split Sets
    # ──────────────────────────────────────────────────────────────────────

    def _run_delaunay(self):
        # 1. Strip out orange cones entirely for triangulation
        track_cones = [c for c in self.mapped_cones if c[2] in ["Blue", "Yellow"]]
        
        if len(track_cones) < 3 or not self.gate_established:
            return
            
        try:
            pts = np.array([[c[0], c[1]] for c in track_cones])
            tri = Delaunay(pts)
            
            valid_edges = set()
            
            for simplex in tri.simplices:
                p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                # FSD FILTER 1 & 2
                R = circumcircle_radius(p1, p2, p3)
                L_max = max(np.linalg.norm(p1-p2), np.linalg.norm(p2-p3), np.linalg.norm(p1-p3))
                
                if R > 4.5 or L_max > 6.0:
                    continue
                
                for i in range(3):
                    a = simplex[i]
                    b = simplex[(i+1)%3]
                    
                    c1 = track_cones[a]
                    c2 = track_cones[b]
                    
                    # Track Bounds Only (Blue to Yellow)
                    if c1[2] != c2[2]:
                        edge = tuple(sorted((a, b)))
                        valid_edges.add(edge)
                        
                        mid_x = (c1[0] + c2[0]) / 2.0
                        mid_y = (c1[1] + c2[1]) / 2.0
                        
                        # ─── THE SMART GEOMETRIC KNIFE ────────────────────────────
                        
                        # 1. Kills the Entry Straight ONLY (Not the right side of the circles)
                        # The entry straight is > 1.0m to the right of the gate AND vertically aligned with it.
                        if mid_x > self.gate_center_x + 1.0 and abs(mid_y - self.gate_center_y) < 4.5:
                            continue
                            
                        # 2. Splits Circles & Kills Crossover (+/- 1.0m from Gate_Y)
                        y_diff = mid_y - self.gate_center_y
                        
                        if y_diff > 1.0:
                            # Belongs to Right (Top) Circle
                            if not any(math.hypot(mid_x-ex, mid_y-ey) < 0.5 for ex, ey in self.right_circle_wpts):
                                self.right_circle_wpts.append([mid_x, mid_y])
                                
                        elif y_diff < -1.0:
                            # Belongs to Left (Bottom) Circle
                            if not any(math.hypot(mid_x-ex, mid_y-ey) < 0.5 for ex, ey in self.left_circle_wpts):
                                self.left_circle_wpts.append([mid_x, mid_y])

            # Save valid edges for the visualizer
            self._visual_edges = [((pts[e[0]][0], pts[e[0]][1]), (pts[e[1]][0], pts[e[1]][1])) for e in valid_edges]
                                   
        except Exception as e: 
            pass

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
            
            if self.laps_completed == 1:
                self.state = 'RIGHT_CIRCLES'
            elif self.laps_completed == 3:
                self.state = 'LEFT_CIRCLES'
            elif self.laps_completed == 5:
                self.state = 'FINISHED'
                self._save_waypoints()

        self._run_delaunay()
        self._update_debug_plot()

    def _save_waypoints(self):
        try:
            # Save Right Circle
            with open('delaunay_right_circle.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                for wp in self.right_circle_wpts:
                    writer.writerow([wp[0], wp[1]])
            
            # Save Left Circle
            with open('delaunay_left_circle.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                for wp in self.left_circle_wpts:
                    writer.writerow([wp[0], wp[1]])
                    
            self.get_logger().info(f"SUCCESS! Waypoints saved into left & right CSV files.")
        except Exception as e:
            self.get_logger().error(f"Failed to save waypoints: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Debug Plotting
    # ──────────────────────────────────────────────────────────────────────
    def _setup_debug_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_title('Passive Delaunay Mapper (Split Circles)')

        self._plot_triangles = LineCollection([], colors='cyan', linewidths=1.0, alpha=0.5, zorder=1)
        self.ax.add_collection(self._plot_triangles)
        
        # Color coding the separated lists!
        self._plot_right_wpts = self.ax.scatter([], [], c='lime', s=35, label='Right Circle (Lime)', zorder=5)
        self._plot_left_wpts = self.ax.scatter([], [], c='green', s=35, label='Left Circle (Dark Green)', zorder=5)
        
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

        # Plot Right Waypoints
        if self.right_circle_wpts:
            pts = np.array(self.right_circle_wpts)
            self._plot_right_wpts.set_offsets(pts)
            all_x += list(pts[:, 0]); all_y += list(pts[:, 1])
            
        # Plot Left Waypoints
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
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
