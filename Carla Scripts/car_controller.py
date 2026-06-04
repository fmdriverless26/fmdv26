#!/usr/bin/env python3
"""
fm_dv_controller.py  —  Formula Manipal Driverless
====================================================
Camera-based cone detection + Delaunay planning + MPC with lap learning
"""

import carla
import csv
import math
import time
import sys
import os
import numpy as np
import queue
import threading
import cv2
from scipy.spatial import Delaunay as ScipyDelaunay
from scipy.interpolate import splprep, splev

# YOLO — loaded lazily so missing ultralytics doesn't crash startup
try:
    from ultralytics import YOLO as _YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[WARN] ultralytics not installed — YOLO overlay disabled")
    print("       pip install ultralytics")

try:
    import casadi as ca
    CASADI_OK = True
except ImportError:
    CASADI_OK = False
    print("[WARN] CasADi not found — MPC unavailable")


# ═══════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════
CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEFAULTS = {
    'use_mpc': True,
    'camera_fov': 90.0,
    'camera_range': 12.0,
    'max_edge_len': 8.0,
    'spline_res': 40,
    'hz': 50,
    'max_steer': 0.5,
    'max_speed': 14.0,
    'min_speed': 2.0,
    'max_lat_accel': 17.0,
    'horizon': 8,
    'dt': 0.1,
    'wheelbase': 2.495,
    'k_radius': 0.35,
    'w_cte': 50.0,
    'w_yaw': 20.0,
    'w_speed': 0.1,
    'w_steer': 10.0,
    'w_accel': 0.0,
    'w_steer_rate': 1500.0,
    'max_accel': 4.0,
    'min_accel': -5.0,
    'lookahead': 6,
    'draw_life': 0.12,
    'correction_rate': 0.3,
    'max_circumcircle_radius': 4.0,
    'lap_complete_radius': 3.0,
    'lap1_speed_factor': 0.4,  # lap 1 speed cap as fraction of max_speed
}


# ═══════════════════════════════════════════════════════════════
# ARG PARSING
# ═══════════════════════════════════════════════════════════════
def parse_args(args):
    params = {}
    for arg in args:
        if '=' not in arg:
            continue
        k, v = arg.split('=', 1)
        k, v = k.strip(), v.strip()
        if v.lower() == 'true':
            params[k] = True
        elif v.lower() == 'false':
            params[k] = False
        else:
            try:
                params[k] = int(v) if '.' not in v else float(v)
            except ValueError:
                params[k] = v
    return params


# ═══════════════════════════════════════════════════════════════
# CLASS 1 — ConeColourMap
# ═══════════════════════════════════════════════════════════════
class ConeColourMap:
    MATCH_THRESH = 0.8

    def __init__(self, world, cone_csv_path, center_x, center_y):
        self._map = {}
        self._build(world, cone_csv_path, center_x, center_y)

    def _build(self, world, cone_csv_path, cx, cy):
        csv_cones = []
        with open(cone_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row['tag'].strip().lower()
                if tag in ('blue', 'yellow', 'big_orange'):
                    csv_cones.append({
                        'x': float(row['x']) + cx,
                        'y': float(row['y']) + cy,
                        'colour': tag,
                    })

        actors = list(world.get_actors().filter('static.prop.*cone*'))
        matched = 0
        for actor in actors:
            loc = actor.get_location()
            best_d = float('inf')
            best_c = 'unknown'
            for cc in csv_cones:
                d = math.hypot(loc.x - cc['x'], loc.y - cc['y'])
                if d < best_d:
                    best_d = d
                    best_c = cc['colour']
            if best_d < self.MATCH_THRESH:
                self._map[actor.id] = best_c
                matched += 1

        print(f"[ConeColourMap] {matched}/{len(actors)} cones matched.")

    def get(self, actor_id):
        return self._map.get(actor_id, 'unknown')


# ═══════════════════════════════════════════════════════════════
# CLASS 2 — CameraConeDetector
# ═══════════════════════════════════════════════════════════════
class CameraConeDetector:
    def __init__(self, world, fov_deg, max_range):
        self.world = world
        self.fov_deg = fov_deg
        self.fov_half = math.radians(fov_deg / 2.0)
        self.max_range = max_range
        
    def detect(self, all_cones, car_x, car_y, car_yaw_rad):
        detected = []
        fwd_x = math.cos(car_yaw_rad)
        fwd_y = math.sin(car_yaw_rad)
        
        for cone in all_cones:
            dx = cone['x'] - car_x
            dy = cone['y'] - car_y
            dist = math.hypot(dx, dy)
            
            if dist < 0.5 or dist > self.max_range:
                continue
            
            dot = (dx * fwd_x + dy * fwd_y) / dist
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            
            if angle <= self.fov_half:
                detected.append(cone)
                
        return detected
    
    def draw_fov_lines(self, car_x, car_y, car_yaw_rad, car_z):
        w = self.world
        fwd_x = math.cos(car_yaw_rad)
        fwd_y = math.sin(car_yaw_rad)
        
        left_angle = car_yaw_rad + self.fov_half
        right_angle = car_yaw_rad - self.fov_half
        
        left_x = car_x + self.max_range * math.cos(left_angle)
        left_y = car_y + self.max_range * math.sin(left_angle)
        right_x = car_x + self.max_range * math.cos(right_angle)
        right_y = car_y + self.max_range * math.sin(right_angle)
        center_x = car_x + self.max_range * math.cos(car_yaw_rad)
        center_y = car_y + self.max_range * math.sin(car_yaw_rad)
        
        w.debug.draw_line(
            carla.Location(x=car_x, y=car_y, z=car_z + 1.5),
            carla.Location(x=center_x, y=center_y, z=car_z + 0.5),
            thickness=0.02, color=carla.Color(100, 100, 100), life_time=0.1
        )
        
        w.debug.draw_line(
            carla.Location(x=car_x, y=car_y, z=car_z + 1.5),
            carla.Location(x=left_x, y=left_y, z=car_z + 0.5),
            thickness=0.02, color=carla.Color(80, 80, 0), life_time=0.1
        )
        
        w.debug.draw_line(
            carla.Location(x=car_x, y=car_y, z=car_z + 1.5),
            carla.Location(x=right_x, y=right_y, z=car_z + 0.5),
            thickness=0.02, color=carla.Color(80, 80, 0), life_time=0.1
        )


# ═══════════════════════════════════════════════════════════════
# CLASS 3 — RGB Camera Viewer (FIXED)
# ═══════════════════════════════════════════════════════════════
class RGBCameraViewer:
    """
    RGB camera window with optional YOLO bounding-box overlay.

    Pass model_path='.../best.pt' to enable cone detection overlay.
    The YOLO model is loaded in a background thread so it never blocks
    the controller startup.  Path planning is completely unaffected —
    YOLO is display-only.

    Expected class indices (from your Roboflow data.yaml):
        0 -> blue_cone   (drawn in blue)
        1 -> yellow_cone (drawn in yellow)
        2 -> orange_cone (drawn in orange)
    Edit CLASS_COLORS / CLASS_NAMES below if your indices differ.
    """

    # ── Customise these to match your data.yaml ──────────────────────
    CLASS_NAMES  = {0: 'blue', 1: 'yellow', 2: 'orange'}
    CLASS_COLORS = {
        0: (255,  60,  30),   # BGR blue
        1: (  0, 210, 255),   # BGR yellow
        2: (  0, 100, 255),   # BGR orange
    }
    CONF_THRESH = 0.40        # minimum confidence to draw a box
    # ─────────────────────────────────────────────────────────────────

    def __init__(self, world, vehicle, width=800, height=600, fov=90,
                 model_path=None):
        self.world       = world
        self.vehicle     = vehicle
        self.width       = width
        self.height      = height
        self.fov         = fov
        self.camera      = None
        self.image_queue = queue.Queue(maxsize=2)  # drop old frames, never lag
        self.running     = False

        # YOLO state
        self._yolo       = None          # loaded model (or None)
        self._yolo_ready = False
        self._model_path = model_path

        if model_path and YOLO_OK:
            threading.Thread(target=self._load_yolo, daemon=True).start()
        elif model_path and not YOLO_OK:
            print("[Camera] YOLO overlay requested but ultralytics not installed.")

    def _load_yolo(self):
        try:
            print(f"[Camera] Loading YOLO model: {self._model_path}")
            self._yolo = _YOLO(self._model_path)
            # Warm-up pass so first real frame isn't slow
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._yolo(dummy, verbose=False)
            self._yolo_ready = True
            print("[Camera] YOLO model ready.")
        except Exception as e:
            print(f"[Camera] YOLO load failed: {e}")

    def setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', str(self.fov))

        transform = carla.Transform(carla.Location(x=1.2, y=0, z=1.6))
        self.camera = self.world.spawn_actor(
            camera_bp, transform, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)
        self.camera.listen(self._on_image)
        self.running = True
        print(f"[Camera] RGB Camera attached - {self.width}x{self.height}, {self.fov}° FOV")

    def _on_image(self, image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
            # Non-blocking put: if queue is full, discard oldest frame
            if self.image_queue.full():
                try:
                    self.image_queue.get_nowait()
                except queue.Empty:
                    pass
            self.image_queue.put_nowait(array)
        except Exception:
            pass

    def _draw_detections(self, img):
        """Run YOLO on img and draw bounding boxes + labels in-place."""
        try:
            results = self._yolo(img, verbose=False, conf=self.CONF_THRESH)
        except Exception:
            return img

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0].item())
                conf    = float(box.conf[0].item())

                color = self.CLASS_COLORS.get(cls_id, (200, 200, 200))
                label = self.CLASS_NAMES.get(cls_id, f'cls{cls_id}')

                # Box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label background + text
                text     = f"{label} {conf:.2f}"
                font     = cv2.FONT_HERSHEY_SIMPLEX
                scale    = 0.55
                thickness = 1
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                # Keep label inside frame
                lx = max(x1, 0)
                ly = max(y1 - th - 6, 0)
                cv2.rectangle(img,
                              (lx, ly),
                              (lx + tw + 4, ly + th + 6),
                              color, -1)
                cv2.putText(img, text,
                            (lx + 2, ly + th + 2),
                            font, scale,
                            (0, 0, 0),   # black text on coloured background
                            thickness, cv2.LINE_AA)

        # Status badge: show whether YOLO is active
        badge     = "YOLO ON" if self._yolo_ready else "YOLO loading..."
        badge_col = (0, 200, 60) if self._yolo_ready else (0, 140, 255)
        cv2.putText(img, badge, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    badge_col, 2, cv2.LINE_AA)
        return img

    def show(self):
        if self.camera is None:
            self.setup_camera()

        cv2.namedWindow('CARLA Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CARLA Camera', self.width, self.height)

        while self.running:
            try:
                img = self.image_queue.get(timeout=1.0)
                if img is not None:
                    if self._yolo_ready:
                        img = self._draw_detections(img)
                    cv2.imshow('CARLA Camera', img)
                    cv2.waitKey(1)
            except queue.Empty:
                continue

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.stop()
            self.camera.destroy()


# ═══════════════════════════════════════════════════════════════
# CLASS 4 — DelaunayPlanner (FIXED)
# ═══════════════════════════════════════════════════════════════
class DelaunayPlanner:
    class Result:
        def __init__(self):
            self.waypoints = []
            self.good_edges = []
            self.bad_edges = []
            self.midpoints = []
            self.visible_cones = []

    def __init__(self, max_edge_len, spline_res, max_circumcircle_radius=4.0):
        self.max_edge_len = max_edge_len
        self.spline_res = spline_res
        self.max_circumcircle_radius = max_circumcircle_radius

    def plan(self, visible_cones, car_x, car_y):
        result = self.Result()
        result.visible_cones = list(visible_cones)

        if len(visible_cones) < 4:
            return result

        coords = np.array([(c['x'], c['y']) for c in visible_cones])
        colours = np.array([c['colour'] for c in visible_cones])

        try:
            tri = ScipyDelaunay(coords)
        except:
            return result

        good_edges = []
        midpoints = []

        # ----------------------------------
        # TRIANGLES → FILTER → MIDPOINTS
        # ----------------------------------
        for simplex in tri.simplices:
            ids = simplex
            pts = coords[ids]
            cols = [colours[i] for i in ids]

            # ❌ Ignore non-track cones
            if any(c not in ('blue', 'yellow') for c in cols):
                continue

            # ✅ Only 2-1 triangle combinations
            blue_count = cols.count('blue')
            yellow_count = cols.count('yellow')

            if not ((blue_count == 2 and yellow_count == 1) or
                    (yellow_count == 2 and blue_count == 1)):
                continue

            # ❌ Remove outer/big triangles
            R = self._circumcircle_radius(pts[0], pts[1], pts[2])
            if R > self.max_circumcircle_radius:
                continue

            # ----------------------------------
            # EDGES + MIDPOINTS
            # ----------------------------------
            for i in range(3):
                a = ids[i]
                b = ids[(i + 1) % 3]

                p1 = coords[a]
                p2 = coords[b]

                col_a = colours[a]
                col_b = colours[b]

                # store for visualization
                good_edges.append(((p1[0], p1[1]), (p2[0], p2[1]), col_a, col_b))

                # ✅ midpoint ONLY for opposite colors
                if col_a != col_b:
                    mid = (p1 + p2) / 2.0
                    midpoints.append(mid)

        result.good_edges = good_edges
        result.bad_edges = []

        # ----------------------------------
        # MIDPOINT CLEANUP
        # ----------------------------------
        if len(midpoints) < 5:
            result.midpoints = midpoints
            return result

        midpoints = np.array(midpoints)

        # ✅ remove duplicates (critical)
        midpoints = np.unique(np.round(midpoints, 2), axis=0)

        # ----------------------------------
        # ORDER POINTS (GRAPH METHOD)
        # ----------------------------------
        ordered = self._order_points_graph(midpoints)
        result.midpoints = ordered

        # ----------------------------------
        # SPLINE FIT
        # ----------------------------------
        try:
            tck, _ = splprep([ordered[:, 0], ordered[:, 1]], s=0.8)
            u = np.linspace(0, 1, self.spline_res)
            x_new, y_new = splev(u, tck)
            result.waypoints = list(zip(x_new, y_new))
        except:
            result.waypoints = [tuple(p) for p in ordered]

        return result

    def _order_points_graph(self, points):
        if len(points) < 3:
            return points

        dist = np.linalg.norm(points[:, None] - points[None, :], axis=2)

        neighbors = {}
        for i in range(len(points)):
            idx = np.argsort(dist[i])[1:3]
            neighbors[i] = list(idx)

        ordered = [0]
        visited = set(ordered)

        while len(ordered) < len(points):
            current = ordered[-1]

            next_candidates = [n for n in neighbors[current] if n not in visited]

            if next_candidates:
                nxt = next_candidates[0]
            else:
                remaining = list(set(range(len(points))) - visited)
                dists = [dist[current][r] for r in remaining]
                nxt = remaining[np.argmin(dists)]

            ordered.append(nxt)
            visited.add(nxt)

        return points[ordered]

    def _circumcircle_radius(self, a, b, c):
        A = np.linalg.norm(b - c)
        B = np.linalg.norm(a - c)
        C = np.linalg.norm(a - b)
        s = (A + B + C) / 2.0
        area = max(s * (s - A) * (s - B) * (s - C), 0.0)
        if area == 0:
            return np.inf
        area = np.sqrt(area)
        return (A * B * C) / (4.0 * area)

    def _order_points(self, points):
        """Order points by nearest neighbor"""
        if len(points) < 3:
            return np.array(points)

        points = list(points)
        ordered = [points.pop(0)]

        while points:
            last = ordered[-1]
            dists = [np.linalg.norm(np.array(p) - np.array(last)) for p in points]
            idx = int(np.argmin(dists))
            ordered.append(points.pop(idx))

        return np.array(ordered)

    def _create_spline(self, points, resolution):
        """Create smooth spline (NOT looping yet)"""
        pts = np.array(points)

        if len(pts) < 4:
            return list(pts)

        # Smooth but NOT looping
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=1.5)
            u = np.linspace(0, 1, resolution)
            x, y = splev(u, tck)
            return list(zip(x, y))
        except:
            return self._catmull_rom(points, resolution)

    def _catmull_rom(self, pts, resolution):
        if len(pts) < 2:
            return pts
        if len(pts) == 2:
            return [tuple(pts[0]), tuple(pts[1])]
        
        padded = [pts[0]] + list(pts) + [pts[-1]]
        seg_res = max(4, resolution // len(pts))
        result = []
        
        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i-1], padded[i], padded[i+1], padded[i+2]
            for j in range(seg_res):
                t = j / seg_res
                t2 = t * t
                t3 = t2 * t
                pt = 0.5 * (
                    2*p1
                    + (-p0 + p2) * t
                    + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                    + (-p0 + 3*p1 - 3*p2 + p3) * t3
                )
                result.append(tuple(pt))
        
        result.append(tuple(pts[-1]))
        return result


# ═══════════════════════════════════════════════════════════════
# CLASS 5 — MPCController
# ═══════════════════════════════════════════════════════════════
class MPCController:
    def __init__(self, p):
        if not CASADI_OK:
            raise ImportError("CasADi required")

        self.N = int(p['horizon'])
        self.dt = p['dt']
        self.L = p['wheelbase']
        self.w_cte = p['w_cte']
        self.w_yaw = p['w_yaw']
        self.w_speed = p['w_speed']
        self.w_steer = p['w_steer']
        self.w_accel = p['w_accel']
        self.w_steer_rate = p['w_steer_rate']
        self.max_steer = p['max_steer']
        self.max_accel = p['max_accel']
        self.min_accel = p['min_accel']
        self.max_speed = p['max_speed']
        self.min_speed = p['min_speed']
        self.max_lat_accel = p['max_lat_accel']
        self.k_radius = p['k_radius']
        self.last_steer = 0.0

        self._build_solver()
        print(f"[MPC] Solver ready")

    def _build_solver(self):
        x, y, yaw, v = [ca.SX.sym(s) for s in ('x', 'y', 'yaw', 'v')]
        delta, a = ca.SX.sym('delta'), ca.SX.sym('a')
        state = ca.vertcat(x, y, yaw, v)
        control = ca.vertcat(delta, a)

        f = ca.Function('f', [state, control], [ca.vertcat(
            x + v * ca.cos(yaw) * self.dt,
            y + v * ca.sin(yaw) * self.dt,
            yaw + (v / self.L) * ca.tan(delta) * self.dt,
            v + a * self.dt,
        )])

        X = ca.SX.sym('X', 4, self.N + 1)
        U = ca.SX.sym('U', 2, self.N)
        P = ca.SX.sym('P', 4 + 4 * self.N)

        cost = 0
        g = [X[:, 0] - P[0:4]]

        for k in range(self.N):
            rx = P[4 + 4*k]
            ry = P[5 + 4*k]
            ryaw = P[6 + 4*k]
            rv = P[7 + 4*k]
            rx_n = P[4 + 4*(k+1)] if k < self.N-1 else rx
            ry_n = P[5 + 4*(k+1)] if k < self.N-1 else ry
            plen = ca.sqrt((rx_n-rx)**2 + (ry_n-ry)**2 + 1e-6)
            cte = ((X[0,k]-rx)*(ry_n-ry) - (X[1,k]-ry)*(rx_n-rx)) / plen
            yerr = ca.atan2(ca.sin(X[2,k]-ryaw), ca.cos(X[2,k]-ryaw))
            verr = X[3,k] - rv

            cost += self.w_cte * cte**2
            cost += self.w_yaw * yerr**2
            cost += self.w_speed * verr**2
            cost += self.w_steer * U[0,k]**2
            cost += self.w_accel * U[1,k]**2
            if k > 0:
                cost += self.w_steer_rate * (U[0,k] - U[0,k-1])**2
            g.append(X[:, k+1] - f(X[:, k], U[:, k]))

        g_vec = ca.vertcat(*g)
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {'x': opt_vars, 'f': cost, 'g': g_vec, 'p': P}

        opts = {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 50, 'ipopt.tol': 1e-2,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_states = 4 * (self.N + 1)
        self.n_constraints = g_vec.size1()
        self.lbx = ([-ca.inf, -ca.inf, -ca.inf, self.min_speed] * (self.N+1) +
                   [-self.max_steer, self.min_accel] * self.N)
        self.ubx = ([ca.inf, ca.inf, ca.inf, self.max_speed] * (self.N+1) +
                   [self.max_steer, self.max_accel] * self.N)

    def build_ref(self, state, waypoints, current_idx):
        n = len(waypoints)
        ref = []
        for k in range(self.N):
            idx = min(current_idx + k, n - 1)
            wp = waypoints[idx]
            next_idx = min(idx + 1, n - 1)
            wp_next = waypoints[next_idx]
            yaw_r = math.atan2(wp_next[1] - wp[1], wp_next[0] - wp[0])
            speed_ref = self.max_speed * 0.7
            ref += [wp[0], wp[1], yaw_r, speed_ref]
        return ref

    def solve(self, state, ref):
        if len(ref) != 4 * self.N:
            return self.last_steer, 0.0
        P_val = list(state) + ref
        x0 = list(state) * (self.N+1) + [self.last_steer, 0.0] * self.N
        lbg = [0.0] * self.n_constraints
        ubg = [0.0] * self.n_constraints
        try:
            sol = self.solver(x0=x0, p=P_val, lbx=self.lbx, ubx=self.ubx, lbg=lbg, ubg=ubg)
            flat = sol['x'].full().flatten().tolist()
            steer_raw = flat[self.n_states]
            accel = flat[self.n_states + 1]
            steer = 0.9 * steer_raw + 0.1 * self.last_steer
            self.last_steer = steer
            return steer, accel
        except:
            return self.last_steer, 0.0


# ═══════════════════════════════════════════════════════════════
# CLASS 6 — PurePursuit
# ═══════════════════════════════════════════════════════════════
class PurePursuit:
    def __init__(self, lookahead, max_steer):
        self.lookahead = lookahead
        self.max_steer = max_steer

    def compute(self, car_x, car_y, car_yaw, waypoints):
        if not waypoints:
            return 0.0
        dists = [math.hypot(wp[0]-car_x, wp[1]-car_y) for wp in waypoints]
        closest = int(np.argmin(dists))
        target = waypoints[min(closest + self.lookahead, len(waypoints)-1)]
        angle = math.atan2(target[1]-car_y, target[0]-car_x)
        steer = math.atan2(math.sin(angle-car_yaw), math.cos(angle-car_yaw))
        return max(-self.max_steer, min(self.max_steer, steer))


# ═══════════════════════════════════════════════════════════════
# CLASS 7 — Visualiser
# ═══════════════════════════════════════════════════════════════
class Visualiser:
    def __init__(self, world, life=0.2):
        self.world = world
        self.life = life
        self.last_draw_time = 0
        self.draw_interval = 0.1

    def draw(self, result, visible_cones, base_z):
        current_time = time.time()
        if current_time - self.last_draw_time < self.draw_interval:
            return
        self.last_draw_time = current_time

        z = base_z + 0.25
        zs = base_z + 0.35
        w = self.world

        for c in visible_cones:
            col = {
                'blue': carla.Color(0, 60, 180),
                'yellow': carla.Color(180, 160, 0),
                'big_orange': carla.Color(180, 80, 0),
            }.get(c['colour'], carla.Color(100, 100, 100))
            w.debug.draw_point(
                carla.Location(x=c['x'], y=c['y'], z=z+0.6),
                size=0.06, color=col, life_time=self.life)

        for (x1, y1), (x2, y2), col_a, col_b in result.good_edges:
            if col_a == 'blue' and col_b == 'blue':
                edge_col = carla.Color(0, 60, 180)
            elif col_a == 'yellow' and col_b == 'yellow':
                edge_col = carla.Color(180, 150, 0)
            else:
                edge_col = carla.Color(100, 100, 100)
            w.debug.draw_line(
                carla.Location(x=x1, y=y1, z=z),
                carla.Location(x=x2, y=y2, z=z),
                thickness=0.01, color=edge_col, life_time=self.life)

        for (x1, y1), (x2, y2) in result.bad_edges:
            w.debug.draw_line(
                carla.Location(x=x1, y=y1, z=z),
                carla.Location(x=x2, y=y2, z=z),
                thickness=0.005, color=carla.Color(40, 15, 15), life_time=self.life)

        for (mx, my) in result.midpoints:
            w.debug.draw_point(
                carla.Location(x=mx, y=my, z=z+0.15),
                size=0.03, color=carla.Color(0, 100, 40), life_time=self.life)

        pts = result.waypoints
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            w.debug.draw_line(
                carla.Location(x=x1, y=y1, z=zs),
                carla.Location(x=x2, y=y2, z=zs),
                thickness=0.01, color=carla.Color(0, 100, 40), life_time=self.life)


# ═══════════════════════════════════════════════════════════════
# CLASS 8 — FMDVController
# ═══════════════════════════════════════════════════════════════
class FMDVController:
    def __init__(self, cone_csv_path, p):
        self.cone_csv = cone_csv_path
        self.p = p
        self.use_mpc = p['use_mpc'] and CASADI_OK

        print("Connecting to CARLA...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print("Connected.")

        self.ego = self._find_ego()
        ego_loc = self.ego.get_transform().location

        center_x, center_y = self._find_center(cone_csv_path, ego_loc)

        self.colour_map = ConeColourMap(self.world, cone_csv_path, center_x, center_y)
        
        self.camera_detector = CameraConeDetector(
            self.world, 
            fov_deg=p['camera_fov'], 
            max_range=p['camera_range']
        )
        
        # Resolve YOLO model path: look next to this script file
        import inspect
        _script_dir = os.path.dirname(os.path.abspath(
            inspect.getfile(FMDVController)))
        _model_path = os.path.join(_script_dir, 'best.pt')
        if not os.path.exists(_model_path):
            _model_path = None
            print("[Camera] best.pt not found next to script — YOLO overlay disabled")
        else:
            print(f"[Camera] Found YOLO model: {_model_path}")

        self.rgb_camera = RGBCameraViewer(
            self.world, self.ego,
            width=800, height=600,
            fov=p['camera_fov'],
            model_path=_model_path)
        
        self.planner = DelaunayPlanner(
            max_edge_len=p['max_edge_len'], 
            spline_res=p['spline_res'],
            max_circumcircle_radius=p['max_circumcircle_radius']
        )
        
        self.mpc = MPCController(p) if self.use_mpc else None
        self.pursuit = PurePursuit(lookahead=p['lookahead'], max_steer=p['max_steer'])
        self.vis = Visualiser(self.world, life=p['draw_life'])

        self._all_cones = self._load_cone_actors()
        print(f"[Controller] {len(self._all_cones)} cone actors cached.")

        self._wp_idx = 0
        self._last_wps = []
        self._lap_count = 0
        self._lap_start = time.time()
        self._plot_shown = False
        self._start_position = None
        self._max_dist_traveled = 0
        self._full_lap_spline = None
        
        self._snap = {
            'cones': {},
            'good_edges': set(),
            'midpoints': [],      # accumulated list of midpoint arrays
            'midpoint_keys': set(), # dedup set of (rounded_x, rounded_y)
            'spline': [],
        }
        self._last_lap1_mps = []   # memory of last midpoints for lap-1 fallback

    def _find_ego(self):
        for a in self.world.get_actors().filter('vehicle.*'):
            if a.attributes.get('role_name') in ('hero', 'ego_vehicle'):
                return a
        actors = list(self.world.get_actors().filter('vehicle.*'))
        if actors:
            return actors[0]
        raise RuntimeError("No vehicle found.")

    def _find_center(self, cone_csv_path, ego_loc):
        car_start_x = car_start_y = None
        with open(cone_csv_path, 'r') as f:
            for row in csv.DictReader(f):
                if row['tag'].strip().lower() == 'car_start':
                    car_start_x = float(row['x'])
                    car_start_y = float(row['y'])
                    break
        if car_start_x is None:
            return ego_loc.x, ego_loc.y
        return ego_loc.x - car_start_x, ego_loc.y - car_start_y

    def _load_cone_actors(self):
        actors = list(self.world.get_actors().filter('static.prop.*cone*'))
        cones = []
        for a in actors:
            loc = a.get_location()
            cones.append({
                'actor': a,
                'id': a.id,
                'x': loc.x,
                'y': loc.y,
                'colour': self.colour_map.get(a.id),
            })
        return cones

    def run(self):
        dt = 1.0 / self.p['hz']
        mode_str = f"{'MPC' if self.use_mpc else 'PurePursuit'}"
        
        print(f"\n{'='*55}")
        print(f"FM DV CONTROLLER — Camera-based Detection + Lap Learning")
        print(f"Camera: {self.p['camera_fov']}° FOV, {self.p['camera_range']}m range")
        print(f"Lap radius: {self.p['lap_complete_radius']}m")
        print(f"Controller: {mode_str}")
        print(f"{'='*55}")
        print("Starting in 3 seconds...\n")
        time.sleep(3.0)
        
        camera_thread = threading.Thread(target=self.rgb_camera.show, daemon=True)
        camera_thread.start()

        try:
            while True:
                t0 = time.time()
                self._tick()
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\nStopping...")
            self.rgb_camera.stop()
            self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            print("Done.")

    def _tick(self):
        tf = self.ego.get_transform()
        x = tf.location.x
        y = tf.location.y
        yaw = math.radians(tf.rotation.yaw)
        vel = self.ego.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        z = tf.location.z

        # Draw FOV lines
        self.camera_detector.draw_fov_lines(x, y, yaw, z)

        # Detect cones and plan
        detected_cones = self.camera_detector.detect(self._all_cones, x, y, yaw)
        result = self.planner.plan(detected_cones, x, y)

        # Store start position at the beginning of every lap
        if self._start_position is None:
            self._start_position = (x, y)
            print(f"[Lap] Start at ({x:.2f}, {y:.2f})")

        # ── LAP 2+ : follow stored global spline with full MPC ───────────────
        if self._full_lap_spline is not None and len(self._full_lap_spline) > 0:
            result.waypoints = self._full_lap_spline
            self._last_wps   = self._full_lap_spline
            waypoints = self._full_lap_spline
            n = len(waypoints)

            # Re-find closest point then walk forward to one that is ahead of car
            fwd_x = math.cos(yaw); fwd_y = math.sin(yaw)
            dists = [math.hypot(wp[0]-x, wp[1]-y) for wp in waypoints]
            closest_idx = int(np.argmin(dists))
            self._wp_idx = closest_idx
            for offset in range(1, n):
                idx = (closest_idx + offset) % n
                wp  = waypoints[idx]
                dx  = wp[0] - x; dy = wp[1] - y
                if math.hypot(dx, dy) >= self.p['lookahead'] and dx*fwd_x + dy*fwd_y > 0:
                    self._wp_idx = idx
                    break

            # Lap detection (every lap)
            if self._start_position:
                dist_to_start = math.hypot(x - self._start_position[0], y - self._start_position[1])
                if dist_to_start > self._max_dist_traveled:
                    self._max_dist_traveled = dist_to_start
                if dist_to_start < self.p['lap_complete_radius'] and self._max_dist_traveled > 20.0:
                    self._lap_complete()

            # Full MPC at normal speed
            state = [x, y, yaw, max(self.p['min_speed'], speed)]
            if self.mpc and n > self._wp_idx + self.p['horizon']:
                ref = self.mpc.build_ref(state, waypoints, self._wp_idx)
                steer, accel = self.mpc.solve(state, ref)
                target_v = max(self.p['min_speed'], min(self.p['max_speed'],
                               speed + accel * self.p['dt']))
            else:
                steer    = self.pursuit.compute(x, y, yaw, waypoints)
                target_v = self.p['max_speed'] * 0.7

        # ── LAP 1 : slow midpoint-by-midpoint follower ────────────────────────
        else:
            lap1_cap = self.p['max_speed'] * self.p['lap1_speed_factor']
            target_v = lap1_cap
            fwd_x = math.cos(yaw); fwd_y = math.sin(yaw)

            # Midpoints from this tick
            raw_mps = result.midpoints
            midpoints = [tuple(m) for m in raw_mps] if len(raw_mps) > 0 else []

            # Fall back to memory if current tick gave nothing (e.g. mid-corner)
            if len(midpoints) == 0:
                midpoints = list(getattr(self, '_last_lap1_mps', []))
            else:
                self._last_lap1_mps = midpoints

            if len(midpoints) == 0:
                # Truly no information — creep straight ahead
                self.ego.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0, brake=0.0))
                self.vis.draw(result, detected_cones, z)
                if self._start_position:
                    d = math.hypot(x-self._start_position[0], y-self._start_position[1])
                    if d > self._max_dist_traveled: self._max_dist_traveled = d
                    if d < self.p['lap_complete_radius'] and self._max_dist_traveled > 20.0:
                        self._lap_complete()
                print(f"  [LAP1-CREEP] ({x:.1f},{y:.1f}) spd={speed:.1f}")
                return

            # Pick nearest midpoint ahead of car; fall back to closest if all behind
            best_idx, best_dist = None, float('inf')
            for i, mp in enumerate(midpoints):
                dx = mp[0]-x; dy = mp[1]-y
                d  = math.hypot(dx, dy)
                if dx*fwd_x + dy*fwd_y > 0 and d < best_dist:
                    best_dist = d; best_idx = i
            if best_idx is None:
                best_idx = int(np.argmin([math.hypot(mp[0]-x, mp[1]-y) for mp in midpoints]))

            steer    = self.pursuit.compute(x, y, yaw, [midpoints[best_idx]])
            waypoints = midpoints
            n = len(waypoints)
            self._wp_idx = best_idx

            # Lap detection
            if self._start_position:
                dist_to_start = math.hypot(x - self._start_position[0], y - self._start_position[1])
                if dist_to_start > self._max_dist_traveled:
                    self._max_dist_traveled = dist_to_start
                if dist_to_start < self.p['lap_complete_radius'] and self._max_dist_traveled > 20.0:
                    self._lap_complete()

        # ── THROTTLE / BRAKE (shared) ─────────────────────────────────────────
        verr     = target_v - speed
        throttle = max(0.0, min(1.0, 0.3 + 0.15 * verr))
        brake    = 0.0
        if verr < -1.5:
            throttle = 0.0
            brake = min(1.0, 0.3 + 0.1 * abs(verr))

        self.ego.apply_control(carla.VehicleControl(
            throttle=float(throttle),
            steer=float(max(-1.0, min(1.0, steer / self.p['max_steer']))),
            brake=float(brake),
        ))

        self.vis.draw(result, detected_cones, z)

        # ── Accumulate snapshot data during lap 1 ─────────────────────────────
        if self._lap_count == 0:
            for c in detected_cones:
                if c['id'] not in self._snap['cones']:
                    self._snap['cones'][c['id']] = {'x': c['x'], 'y': c['y'], 'colour': c['colour']}
            for (x1, y1), (x2, y2), col_a, col_b in result.good_edges:
                key = (round(x1,2), round(y1,2), round(x2,2), round(y2,2), col_a, col_b)
                self._snap['good_edges'].add(key)
            # FIX: accumulate ALL midpoints seen across lap, not just latest batch
            for mp in result.midpoints:
                key = (round(float(mp[0]),1), round(float(mp[1]),1))
                if key not in self._snap['midpoint_keys']:
                    self._snap['midpoint_keys'].add(key)
                    self._snap['midpoints'].append(mp)

        dist_to_start = math.hypot(x - self._start_position[0], y - self._start_position[1]) if self._start_position else 0
        print(f"  [LAP{self._lap_count+1}] ({x:.1f},{y:.1f}) spd={speed:.1f} cones={len(detected_cones)} wp={self._wp_idx}/{n}")

    def _lap_complete(self):
        lap_time = time.time() - self._lap_start
        self._lap_start = time.time()
        self._lap_count += 1

        print(f"\n{'='*50}")
        print(f"*** LAP {self._lap_count} COMPLETE | {lap_time:.2f}s ***")
        print(f"{'='*50}\n")

        # Build global spline on first lap completion only
        if self._lap_count == 1 and len(self._snap['midpoints']) >= 4:
            pts = np.array([np.asarray(m) for m in self._snap['midpoints']])

            # Angular sort around centroid for clean closed-loop ordering
            centroid = pts.mean(axis=0)
            angles   = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
            sorted_ccw = pts[np.argsort(angles)]

            # Match direction to car's current heading
            tf_now  = self.ego.get_transform()
            cx, cy  = tf_now.location.x, tf_now.location.y
            cyaw    = math.radians(tf_now.rotation.yaw)
            dc      = np.linalg.norm(sorted_ccw - np.array([cx, cy]), axis=1)
            nearest = int(np.argmin(dc))
            np_     = sorted_ccw[(nearest+1) % len(sorted_ccw)]
            pp_     = sorted_ccw[(nearest-1) % len(sorted_ccw)]
            fx, fy  = math.cos(cyaw), math.sin(cyaw)
            if (pp_[0]-cx)*fx+(pp_[1]-cy)*fy > (np_[0]-cx)*fx+(np_[1]-cy)*fy:
                sorted_ccw = sorted_ccw[::-1]

            ordered_closed = np.vstack([sorted_ccw, sorted_ccw[0]])
            self._full_lap_spline = self.planner._create_spline(ordered_closed, 300)
            print(f"[INFO] Global lap spline: {len(self._full_lap_spline)} pts")
        elif self._lap_count == 1:
            print("[WARN] Not enough midpoints for global spline")

        # Snapshot for plot
        snap_copy = {
            'cones':      self._snap['cones'].copy(),
            'good_edges': self._snap['good_edges'].copy(),
            'midpoints':  list(self._snap['midpoints']),
            'spline':     list(self._full_lap_spline) if self._full_lap_spline else [],
        }

        # Reset lap tracking
        self._max_dist_traveled = 0
        self._start_position    = None

        # Show plot after first lap only (non-daemon so it survives)
        if self._lap_count == 1 and not self._plot_shown:
            self._plot_shown = True
            threading.Thread(target=self._show_lap_plot,
                             args=(self._lap_count, lap_time, snap_copy),
                             daemon=False).start()

    def _show_lap_plot(self, lap_num, lap_time, snap):
        # Robust backend: try interactive ones, fall back to Agg+savefile
        import matplotlib
        for _b in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'wxAgg', 'Agg']:
            try:
                matplotlib.use(_b)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt

        n_cones = len(snap['cones'])
        n_edges = len(snap['good_edges'])
        n_mps   = len(snap['midpoints'])
        n_spl   = len(snap['spline'])
        print(f"[Plot] backend={matplotlib.get_backend()} | "
              f"{n_cones} cones  {n_edges} edges  {n_mps} midpoints  {n_spl} spline pts")

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        # ── Delaunay triangle edges (flat 6-tuple: x1,y1,x2,y2,col_a,col_b) ──
        bb_drawn = yy_drawn = gate_drawn = False
        for x1, y1, x2, y2, col_a, col_b in snap['good_edges']:
            if col_a == 'blue' and col_b == 'blue':
                lbl = 'Blue wall' if not bb_drawn else None;   bb_drawn = True
                ax.plot([x1,x2],[y1,y2], color='#1a44bb', lw=0.9, alpha=0.6, label=lbl)
            elif col_a == 'yellow' and col_b == 'yellow':
                lbl = 'Yellow wall' if not yy_drawn else None; yy_drawn = True
                ax.plot([x1,x2],[y1,y2], color='#bbaa00', lw=0.9, alpha=0.6, label=lbl)
            else:
                lbl = 'Gate (cross-track)' if not gate_drawn else None; gate_drawn = True
                ax.plot([x1,x2],[y1,y2], color='#888888', lw=0.7, alpha=0.45, label=lbl)

        # ── Cones ──────────────────────────────────────────────────────────────
        blues   = [c for c in snap['cones'].values() if c['colour'] == 'blue']
        yellows = [c for c in snap['cones'].values() if c['colour'] == 'yellow']
        if blues:
            ax.scatter([c['x'] for c in blues],   [c['y'] for c in blues],
                       c='#2255ff', s=35, zorder=5, edgecolors='#aabbff', lw=0.4,
                       label=f'Blue cones ({len(blues)})')
        if yellows:
            ax.scatter([c['x'] for c in yellows], [c['y'] for c in yellows],
                       c='#ffcc00', s=35, zorder=5, edgecolors='#ffe888', lw=0.4,
                       label=f'Yellow cones ({len(yellows)})')

        # ── Midpoints ──────────────────────────────────────────────────────────
        if snap['midpoints']:
            mxs = [float(np.asarray(p).flat[0]) for p in snap['midpoints']]
            mys = [float(np.asarray(p).flat[1]) for p in snap['midpoints']]
            ax.scatter(mxs, mys, c='#00dd55', s=22, zorder=6,
                       edgecolors='#00ff88', lw=0.3,
                       label=f'Midpoints ({len(mxs)})')

        # ── Spline ─────────────────────────────────────────────────────────────
        if snap['spline']:
            sx = [float(p[0]) for p in snap['spline']]
            sy = [float(p[1]) for p in snap['spline']]
            ax.plot(sx, sy, color='#00ff55', lw=2.5, zorder=7,
                    label=f'Racing line ({len(sx)} pts)')

        ax.set_xlabel('X (m)', color='#888888')
        ax.set_ylabel('Y (m)', color='#888888')
        ax.set_title(
            f'Lap {lap_num} complete  |  {lap_time:.1f}s  |  '
            f'Camera {self.p["camera_fov"]}° / {self.p["camera_range"]}m',
            color='#cc3333', fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, color='#1e1e1e', linewidth=0.5)
        ax.legend(facecolor='#1a1a1a', edgecolor='#444444',
                  labelcolor='#cccccc', fontsize=9)
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            out = f'/tmp/lap{lap_num}_plot.png'
            plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"[Plot] Saved to {out}")
        plt.close(fig)
        print("[Plot] Done")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]
    positional = []
    kv_args = []

    for arg in args:
        if '=' in arg:
            kv_args.append(arg)
        else:
            positional.append(arg)

    cone_file = None
    for path in positional:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r') as f:
                first_line = f.readline().lower()
            if 'tag' in first_line:
                cone_file = path
                break
        except Exception:
            continue

    if cone_file is None and positional:
        cone_file = positional[-1]

    if cone_file is None:
        print("Usage: python3 fm_dv_controller.py small_track.csv [param=value ...]")
        sys.exit(1)

    print(f"[main] Using cone CSV: {cone_file}")

    overrides = parse_args(kv_args)
    p = {k: overrides.get(k, v) for k, v in DEFAULTS.items()}
    p['use_mpc'] = bool(p['use_mpc'])
    p['horizon'] = int(p['horizon'])
    p['hz'] = int(p['hz'])
    p['lookahead'] = int(p['lookahead'])
    p['spline_res'] = int(p['spline_res'])

    print("=" * 55)
    print("FM DV CONTROLLER")
    print("=" * 55)
    for k, v in p.items():
        print(f"  {k:<22} {v}")
    print("=" * 55)

    ctrl = FMDVController(cone_file, p)
    ctrl.run()


if __name__ == '__main__':
    main()