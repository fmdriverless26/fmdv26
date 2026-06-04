#!/usr/bin/env python3

import carla
import csv
import math
import time
import sys
import os
import numpy as np

import casadi as ca


# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEFAULTS = {
    'use_mpc':       True,
    'speed':         15.0,
    'lookahead':     8,
    'max_steer':     0.5,
    'hz':            200,
    'w_cte':         50.0,
    'w_yaw':         20.0,
    'w_speed':       0.1,
    'w_steer':       10.0,
    'w_accel':       0.0,
    'w_steer_rate':  1500.0,
    'max_speed':     30.0,
    'min_speed':     11.0,
    'max_accel':     4.0,
    'min_accel':     -5.0,
    'max_lat_accel': 20.0,
    'horizon':       8,
    'dt':            0.1,
    'wheelbase':     2.495,
    'k_radius':      0.35,
}


# ─────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────
def parse_args(args):
    """Parse key=value overrides. Handles bool, int, float."""
    params = {}
    for arg in args:
        if '=' not in arg:
            continue
        k, v = arg.split('=', 1)
        k = k.strip()
        v = v.strip()
        if v.lower() == 'true':
            params[k] = True
        elif v.lower() == 'false':
            params[k] = False
        else:
            try:
                params[k] = int(v) if '.' not in v and 'e' not in v.lower() else float(v)
            except ValueError:
                params[k] = v
    return params


# ─────────────────────────────────────────────
# CENTER DETECTION — track-agnostic
#
# The cone CSV and waypoints CSV are in different coordinate systems.
# Cones use car_start as their reference point.
# Waypoints use the track geometric center as their reference point.
# This function computes the correct CENTER for WAYPOINTS by:
#   1. Computing SPAWN = ego - car_start_csv  (same as cone center)
#   2. Computing the centroid difference between cone CSV and wp CSV
#   3. Applying the additional offset so waypoints align with the cones
#
# Works for ANY pair of track CSVs automatically.
# ─────────────────────────────────────────────
def find_center_for_waypoints(cone_csv_path, wp_csv_path, ego_x, ego_y):
    """
    Compute the correct CENTER_X, CENTER_Y to apply to waypoints so
    they align with the spawned track, for any track CSV pair.

    Logic:
        SPAWN_XY = ego - car_start_csv          (the offset used for cones)
        cone_centroid = mean of all cone positions in CSV space
        wp_centroid   = mean of all waypoints in CSV space
        CENTER_for_wps = SPAWN_XY + (cone_centroid - wp_centroid)

    This corrects for the different origins of the two CSV files.
    """
    # ── Read cone CSV ──
    car_start_x = None
    car_start_y = None
    cone_xs, cone_ys = [], []

    with open(cone_csv_path, 'r') as f:
        for row in csv.DictReader(f):
            tag = row['tag'].strip().lower()
            if tag == 'car_start':
                car_start_x = float(row['x'])
                car_start_y = float(row['y'])
            else:
                try:
                    cone_xs.append(float(row['x']))
                    cone_ys.append(float(row['y']))
                except ValueError:
                    pass

    if car_start_x is None:
        print("[WARN] No car_start in cone CSV. Using ego as CENTER.")
        return ego_x, ego_y

    # SPAWN = what was added to every cone CSV coordinate when spawning
    spawn_x = ego_x - car_start_x
    spawn_y = ego_y - car_start_y

    # ── Cone centroid in CSV space ──
    if not cone_xs:
        print("[WARN] No cone positions found. Using SPAWN as CENTER.")
        return spawn_x, spawn_y

    cone_cx = sum(cone_xs) / len(cone_xs)
    cone_cy = sum(cone_ys) / len(cone_ys)

    # ── Waypoint centroid in CSV space ──
    wp_xs, wp_ys = [], []
    with open(wp_csv_path, 'r') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    wp_xs.append(float(row[0]))
                    wp_ys.append(float(row[1]))
                except ValueError:
                    pass

    if not wp_xs:
        print("[WARN] No waypoints found in CSV. Using SPAWN as CENTER.")
        return spawn_x, spawn_y

    wp_cx = sum(wp_xs) / len(wp_xs)
    wp_cy = sum(wp_ys) / len(wp_ys)

    # ── Correct CENTER for waypoints ──
    # cone_centroid + SPAWN = world centroid of cones
    # wp_centroid + CENTER_for_wps = world centroid of waypoints
    # We want them to be the same → CENTER_for_wps = cone_centroid - wp_centroid + SPAWN
    center_x = spawn_x + (cone_cx - wp_cx)
    center_y = spawn_y + (cone_cy - wp_cy)

    print(f"car_start in CSV:      ({car_start_x:.4f}, {car_start_y:.4f})")
    print(f"Ego in world:          ({ego_x:.4f}, {ego_y:.4f})")
    print(f"SPAWN (cone center):   ({spawn_x:.4f}, {spawn_y:.4f})")
    print(f"Cone CSV centroid:     ({cone_cx:.4f}, {cone_cy:.4f})")
    print(f"Waypoint CSV centroid: ({wp_cx:.4f}, {wp_cy:.4f})")
    print(f"Centroid offset:       ({cone_cx-wp_cx:.4f}, {cone_cy-wp_cy:.4f})")
    print(f"CENTER for waypoints:  ({center_x:.4f}, {center_y:.4f})")
    return center_x, center_y


# ─────────────────────────────────────────────
# WAYPOINTS
# ─────────────────────────────────────────────
def load_waypoints(filename, center_x, center_y):
    waypoints = []
    with open(filename) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    waypoints.append((float(row[0]) + center_x,
                                      float(row[1]) + center_y))
                except ValueError:
                    continue
    return waypoints


def draw_waypoints(world, waypoints, z):
    print(f"Drawing {len(waypoints)} waypoints...")
    for wp in waypoints:
        world.debug.draw_point(
            carla.Location(x=wp[0], y=wp[1], z=z + 0.5),
            size=0.12, color=carla.Color(0, 255, 0), life_time=300.0)


# ─────────────────────────────────────────────
# VEHICLE HELPERS
# ─────────────────────────────────────────────
def find_ego_vehicle(world):
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') == 'ego_vehicle':
            return actor
    vehicles = list(world.get_actors().filter('vehicle.*'))
    if vehicles:
        print("[WARN] No ego_vehicle role found, using first vehicle.")
        return vehicles[0]
    return None


def get_yaw_radians(transform):
    return math.radians(transform.rotation.yaw)


# ─────────────────────────────────────────────
# PURE PURSUIT
# ─────────────────────────────────────────────
def get_target_waypoint(x, y, waypoints, lookahead):
    distances = [math.sqrt((wp[0]-x)**2 + (wp[1]-y)**2) for wp in waypoints]
    closest   = int(np.argmin(distances))
    return waypoints[(closest + lookahead) % len(waypoints)], closest


def compute_pure_pursuit_steering(x, y, yaw_rad, target, max_steer):
    tx, ty = target
    angle  = math.atan2(ty - y, tx - x)
    steer  = math.atan2(math.sin(angle - yaw_rad), math.cos(angle - yaw_rad))
    return max(-max_steer, min(max_steer, steer))


# ─────────────────────────────────────────────
# WAYPOINT SWITCHING — dot product method
# ─────────────────────────────────────────────
def update_waypoint_dot(x, y, waypoints, idx):
    wp_curr = waypoints[idx]
    wp_next = waypoints[(idx + 1) % len(waypoints)]
    car_to_wp  = (wp_curr[0]-x, wp_curr[1]-y)
    wp_to_next = (wp_next[0]-wp_curr[0], wp_next[1]-wp_curr[1])
    dot = car_to_wp[0]*wp_to_next[0] + car_to_wp[1]*wp_to_next[1]
    if dot < 0:
        return (idx + 1) % len(waypoints), True
    return idx, False


# ─────────────────────────────────────────────
# CURVATURE SPEED — Heron's formula
# ─────────────────────────────────────────────
def herons_curvature_speed(wp1, wp2, wp3, max_lat_accel, max_speed, min_speed):
    a = math.hypot(wp2[0]-wp1[0], wp2[1]-wp1[1])
    b = math.hypot(wp3[0]-wp2[0], wp3[1]-wp2[1])
    c = math.hypot(wp3[0]-wp1[0], wp3[1]-wp1[1])
    if a < 0.01 or b < 0.01 or c < 0.01:
        return max_speed
    s       = (a + b + c) / 2.0
    area_sq = s * (s-a) * (s-b) * (s-c)
    if area_sq <= 0.45:
        return max_speed
    radius = (a * b * c) / (4.0 * math.sqrt(area_sq))
    v_ref  = math.sqrt(max_lat_accel * radius)
    return max(min_speed, min(max_speed, v_ref))


# ─────────────────────────────────────────────
# MPC CONTROLLER
# ─────────────────────────────────────────────
class MPCController:
    def __init__(self, waypoints, p):

        self.waypoints = waypoints
        self.n         = len(waypoints)
        self.N         = int(p['horizon'])
        self.dt        = p['dt']
        self.L         = p['wheelbase']

        self.w_cte        = p['w_cte']
        self.w_yaw        = p['w_yaw']
        self.w_speed      = p['w_speed']
        self.w_steer      = p['w_steer']
        self.w_accel      = p['w_accel']
        self.w_steer_rate = p['w_steer_rate']

        self.max_steer    = p['max_steer']
        self.max_accel    = p['max_accel']
        self.min_accel    = p['min_accel']
        self.max_speed    = p['max_speed']
        self.min_speed    = p['min_speed']
        self.max_lat_accel= p['max_lat_accel']
        self.k_radius     = p['k_radius']

        self.last_steer = 0.0
        self._build_solver()
        print(f"MPC initialized: N={self.N} dt={self.dt} L={self.L}")

    def _build_solver(self):
        x   = ca.SX.sym('x')
        y   = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v   = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        a     = ca.SX.sym('a')

        state   = ca.vertcat(x, y, yaw, v)
        control = ca.vertcat(delta, a)

        # Kinematic bicycle model
        f = ca.Function('f', [state, control], [ca.vertcat(
            x   + v * ca.cos(yaw) * self.dt,
            y   + v * ca.sin(yaw) * self.dt,
            yaw + (v / self.L) * ca.tan(delta) * self.dt,
            v   + a * self.dt
        )])

        X = ca.SX.sym('X', 4, self.N + 1)
        U = ca.SX.sym('U', 2, self.N)
        P = ca.SX.sym('P', 4 + 4 * self.N)

        cost = 0
        g    = [X[:, 0] - P[0:4]]

        for k in range(self.N):
            rx   = P[4 + 4*k]
            ry   = P[5 + 4*k]
            ryaw = P[6 + 4*k]
            rv   = P[7 + 4*k]

            rx_n = P[4 + 4*(k+1)] if k < self.N-1 else rx
            ry_n = P[5 + 4*(k+1)] if k < self.N-1 else ry

            px   = rx_n - rx
            py   = ry_n - ry
            plen = ca.sqrt(px**2 + py**2 + 1e-6)
            cte  = ((X[0,k]-rx)*py - (X[1,k]-ry)*px) / plen
            yerr = ca.atan2(ca.sin(X[2,k]-ryaw), ca.cos(X[2,k]-ryaw))
            verr = X[3,k] - rv

            cost += self.w_cte   * cte**2
            cost += self.w_yaw   * yerr**2
            cost += self.w_speed * verr**2
            cost += self.w_steer * U[0,k]**2
            cost += self.w_accel * U[1,k]**2
            if k > 0:
                cost += self.w_steer_rate * (U[0,k] - U[0,k-1])**2

            g.append(X[:, k+1] - f(X[:, k], U[:, k]))

        # Terminal cost
        rx   = P[4 + 4*(self.N-1)]
        ry   = P[5 + 4*(self.N-1)]
        ryaw = P[6 + 4*(self.N-1)]
        dx   = X[0,self.N] - rx
        dy   = X[1,self.N] - ry
        cte_t = -ca.cos(ryaw)*dy + ca.sin(ryaw)*dx
        cost += 2.0 * self.w_cte * cte_t**2

        opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
        g_vec    = ca.vertcat(*g)
        nlp      = {'x': opt_vars, 'f': cost, 'g': g_vec, 'p': P}

        opts = {
            'ipopt.print_level': 0,
            'print_time':        0,
            'ipopt.max_iter':    50,
            'ipopt.tol':         1e-2,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.warm_start_init_point': 'yes',
        }
        self.solver       = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.n_states     = 4 * (self.N + 1)
        self.n_constraints= g_vec.size1()

        self.lbx = ([-ca.inf,-ca.inf,-ca.inf, self.min_speed] * (self.N+1)
                  + [-self.max_steer, self.min_accel] * self.N)
        self.ubx = ([ca.inf, ca.inf, ca.inf, self.max_speed] * (self.N+1)
                  + [self.max_steer, self.max_accel] * self.N)

    def _get_waypoint_sequence(self, state, current_idx):
        indices = []
        x, y, yaw, v = state
        idx = current_idx
        for k in range(self.N):
            indices.append(idx)
            wp   = self.waypoints[idx]
            dist = math.hypot(x - wp[0], y - wp[1])
            if dist < max(v, 2.0) * self.k_radius:
                idx = (idx + 1) % self.n
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        return indices

    def build_ref(self, state, current_idx):
        ref     = []
        indices = self._get_waypoint_sequence(state, current_idx)
        for k in range(self.N):
            i      = indices[k]
            wp     = self.waypoints[i]
            wp_nxt = self.waypoints[(i+1) % self.n]
            yaw_r  = math.atan2(wp_nxt[1]-wp[1], wp_nxt[0]-wp[0])
            if k < self.N - 1:
                wp_prv  = self.waypoints[(i-1) % self.n]
                wp_nxt2 = self.waypoints[(i+2) % self.n]
                v_r = herons_curvature_speed(
                    wp_prv, wp, wp_nxt2,
                    self.max_lat_accel, self.max_speed, self.min_speed)
            else:
                v_r = self.max_speed * 0.8
            ref += [wp[0], wp[1], yaw_r, v_r]
        return ref

    def solve(self, state, ref):
        if len(ref) != 4 * self.N:
            return self.last_steer, 0.0
        P_val = list(state) + ref
        x0    = list(state) * (self.N+1) + [self.last_steer, 0.0] * self.N
        lbg   = [0.0] * self.n_constraints
        ubg   = [0.0] * self.n_constraints
        try:
            sol   = self.solver(x0=x0, p=P_val,
                                lbx=self.lbx, ubx=self.ubx,
                                lbg=lbg, ubg=ubg)
            flat  = sol['x'].full().flatten().tolist()
            steer_raw = flat[self.n_states]
            accel     = flat[self.n_states + 1]
            steer     = 0.9 * steer_raw + 0.1 * self.last_steer
            self.last_steer = steer
            return steer, accel
        except Exception as e:
            print(f"[MPC] Solver error: {e}")
            return self.last_steer, 0.0


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args       = sys.argv[1:]
    wp_file    = None
    cone_file  = None
    extra_args = []

    # Parse positional args (non key=value)
    for arg in args:
        if '=' in arg:
            extra_args.append(arg)
        elif wp_file is None:
            wp_file = arg
        elif cone_file is None:
            cone_file = arg

    if wp_file is None:
        print("[ERROR] Waypoints CSV required as first argument.")
        print("Usage: python3 carla_controller.py waypoints.csv cones.csv [param=value ...]")
        sys.exit(1)

    overrides = parse_args(extra_args)
    p = {k: overrides.get(k, v) for k, v in DEFAULTS.items()}
    # Ensure correct types
    p['use_mpc']  = bool(p['use_mpc'])
    p['lookahead']= int(p['lookahead'])
    p['hz']       = int(p['hz'])
    p['horizon']  = int(p['horizon'])

    use_mpc = p['use_mpc']

    print("=" * 55)
    print(f"FM DV CONTROLLER  —  {'MPC' if use_mpc else 'PURE PURSUIT'}")
    print("=" * 55)
    print(f"  waypoints : {wp_file}")
    print(f"  cones     : {cone_file or 'not provided (center from ego only)'}")
    print(f"  speed     : {p['speed']} m/s")
    print(f"  lookahead : {p['lookahead']}")
    print(f"  max_steer : {p['max_steer']} rad")
    print(f"  hz        : {p['hz']}")
    if use_mpc:
        print(f"  horizon N : {p['horizon']}")
        print(f"  dt        : {p['dt']}")
        print(f"  w_cte     : {p['w_cte']}")
        print(f"  w_yaw     : {p['w_yaw']}")
        print(f"  w_steer_r : {p['w_steer_rate']}")
    print("=" * 55)

    print("Connecting to CARLA...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()
    print("Connected.")

    ego = find_ego_vehicle(world)
    if ego is None:
        print("[ERROR] No ego vehicle found. Spawn track first.")
        sys.exit(1)
    print(f"Ego: {ego.type_id}  id={ego.id}")

    ego_loc  = ego.get_transform().location
    print(f"Ego position: X={ego_loc.x:.4f}, Y={ego_loc.y:.4f}")

    # ── Find center ──
    if cone_file and os.path.exists(cone_file):
        center_x, center_y = find_center_for_waypoints(cone_file, wp_file, ego_loc.x, ego_loc.y)
    else:
        print("[WARN] No cone CSV provided. Waypoints may be misaligned.")
        print("       Pass cones.csv as second argument for accurate placement.")
        center_x, center_y = ego_loc.x, ego_loc.y

    # ── Load waypoints ──
    waypoints = load_waypoints(wp_file, center_x, center_y)
    print(f"Loaded {len(waypoints)} waypoints.")
    print(f"First: ({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f})")
    print(f"Ego:   ({ego_loc.x:.2f}, {ego_loc.y:.2f})")

    # Sanity check: first waypoint should be close to ego
    dist_to_first = math.hypot(waypoints[0][0]-ego_loc.x,
                               waypoints[0][1]-ego_loc.y)
    print(f"Distance ego → first waypoint: {dist_to_first:.2f}m")
    if dist_to_first > 20:
        print("[WARN] First waypoint is far from car. Center detection may be off.")

    draw_waypoints(world, waypoints, ego_loc.z)

    # ── Init MPC if needed ──
    mpc = None
    if use_mpc:
        print("\nBuilding MPC solver (may take a few seconds)...")
        mpc = MPCController(waypoints, p)
        print("MPC ready.")

    print("\nStarting in 3 seconds...")
    time.sleep(3.0)

    dt          = 1.0 / p['hz']
    log_every   = max(1, p['hz'])
    log_counter = 0
    wp_idx      = 0
    lap_count   = 0
    lap_start   = time.time()

    print(f"Running at {p['hz']}Hz. Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.time()

            tf    = ego.get_transform()
            x     = tf.location.x
            y     = tf.location.y
            yaw   = get_yaw_radians(tf)
            vel   = ego.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # ── Waypoint advance ──
            prev_idx = wp_idx
            wp_idx, _ = update_waypoint_dot(x, y, waypoints, wp_idx)
            if wp_idx == 0 and prev_idx == len(waypoints) - 1:
                lap_count += 1
                lap_time   = time.time() - lap_start
                lap_start  = time.time()
                print(f"LAP {lap_count} | {lap_time:.3f}s")

            # ── Curvature speed ──
            wp_prv = waypoints[(wp_idx-1) % len(waypoints)]
            wp_cur = waypoints[wp_idx]
            wp_nxt = waypoints[(wp_idx+1) % len(waypoints)]
            tgt_v  = herons_curvature_speed(
                wp_prv, wp_cur, wp_nxt,
                p['max_lat_accel'], p['speed'], p['min_speed'])

            # ── Control ──
            if use_mpc and mpc:
                state  = [x, y, yaw, max(p['min_speed'], speed)]
                ref    = mpc.build_ref(state, wp_idx)
                steer, accel = mpc.solve(state, ref)
                tgt_v  = max(p['min_speed'],
                             min(p['max_speed'], speed + accel * p['dt']))
            else:
                tgt_wp, _ = get_target_waypoint(x, y, waypoints, p['lookahead'])
                steer = compute_pure_pursuit_steering(
                    x, y, yaw, tgt_wp, p['max_steer'])

            # ── Throttle / brake ──
            verr     = tgt_v - speed
            throttle = max(0.0, min(1.0, 0.3 + 0.15 * verr))
            brake    = 0.0
            if verr < -1.5:
                throttle = 0.0
                brake    = min(1.0, 0.3 + 0.1 * abs(verr))

            ego.apply_control(carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

            # ── Log ──
            log_counter += 1
            if log_counter >= log_every:
                log_counter = 0
                mode = "MPC" if (use_mpc and mpc) else "PP"
                print(f"  pos=({x:.1f},{y:.1f})  "
                      f"yaw={math.degrees(yaw):.1f}°  "
                      f"spd={speed:.1f}m/s  "
                      f"tgt={tgt_v:.1f}m/s  "
                      f"str={math.degrees(steer):.1f}°  "
                      f"thr={throttle:.2f}  "
                      f"wp={wp_idx}/{len(waypoints)}  "
                      f"[{mode}]")

            elapsed = time.time() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1.0)
        print("Done.")


if __name__ == '__main__':
    main()