#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carla_controller_slam.py  --  Formula Manipal Driverless
=========================================================
MPC path-following  +  GraphSLAM (GTSAM) cone mapping
with live animated plot showing the map being built in real time.

Sensors used
------------
  IMU   : sensor.other.imu  -- accelerometer + gyro -> odometry
  Camera: ground-truth cone actors + Gaussian noise (simulates camera range/bearing)

SLAM design
-----------
  Variables  : Pose2 nodes  X(0), X(1), ...   car poses
               Point2 nodes L(0), L(1), ...   cone positions
  Factors    : PriorFactorPose2          -- single prior on X(0) only
               BetweenFactorPose2        -- odometry from IMU integration
               BearingRangeFactor2D      -- cone observation (noisy)
               BetweenFactorPose2        -- loop-closure when lap 1 ends
  Solver     : iSAM2 -- incremental, runs every tick
  Loop close : triggered by distance back to X(0) < threshold
               adds a tight BetweenFactor  X(current) ~= X(0)

After loop closure the optimised cone map is used to rebuild the
reference path for the MPC.

Live plot (separate thread)
---------------------------
  Black background
  Car trajectory   -- white line
  Car current pos  -- red dot
  Mapped cones     -- blue / yellow dots with 1-sigma uncertainty ellipses
  Loop closure     -- green dashed line between X(0) and X(current)
  Updates at ~5 Hz so it never blocks the control loop
"""

# -- Import diagnostics -- printed to console on failure ----------
import sys as _sys
import traceback as _tb
def _safe_import(name):
    try:
        __import__(name)
        return True
    except Exception as e:
        print(f"[IMPORT ERROR] '{name}' failed: {e}", flush=True)
        _tb.print_exc()
        return False

_safe_import('carla')
_safe_import('csv')
_safe_import('math')
_safe_import('time')
_safe_import('os')
_safe_import('threading')
_safe_import('numpy')
_safe_import('queue')
_safe_import('casadi')
_safe_import('gtsam')
# -----------------------------------------------------------------

import carla
import csv
import math
import time
import sys
import os
import threading
import numpy as np
import queue

import casadi as ca

import gtsam
from gtsam import symbol_shorthand
X = symbol_shorthand.X
L = symbol_shorthand.L

# matplotlib is imported lazily inside LiveSLAMPlot._run()
# so a backend failure never crashes the controller.


# ===============================================================
# DEFAULTS
# ===============================================================
CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEFAULTS = {
    'use_mpc':       True,
    'speed':         15.0,
    'lookahead':     8,
    'max_steer':     0.5,
    'hz':            50,
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
    # SLAM params
    'slam_obs_range':     12.0,   # max range to observe a cone (m)
    'slam_obs_fov':       90.0,   # camera horizontal FOV (degrees)
    'slam_noise_range':   0.15,   # std-dev of range noise (m) -- realistic camera
    'slam_noise_bearing': 0.02,   # std-dev of bearing noise (rad)
    'slam_assoc_gate':    4.0,    # fallback Euclidean gate (m) -- actor ID used first
    'loop_close_radius':  3.0,    # distance to X(0) that triggers loop closure (m)
    'loop_min_dist':      20.0,   # must travel this far before loop closure allowed
    'slam_every':          20,    # run SLAM every N ticks (10Hz at 200Hz control)
}


# ===============================================================
# ARG PARSING
# ===============================================================
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
                params[k] = int(v) if '.' not in v and 'e' not in v.lower() else float(v)
            except ValueError:
                params[k] = v
    return params


# ===============================================================
# CSV HELPERS  (unchanged from original)
# ===============================================================
def find_center_for_waypoints(cone_csv_path, wp_csv_path, ego_x, ego_y):
    car_start_x = car_start_y = None
    cone_xs, cone_ys = [], []
    with open(cone_csv_path, 'r') as f:
        for row in csv.DictReader(f):
            tag = row['tag'].strip().lower()
            if tag == 'car_start':
                car_start_x, car_start_y = float(row['x']), float(row['y'])
            else:
                try:
                    cone_xs.append(float(row['x']))
                    cone_ys.append(float(row['y']))
                except ValueError:
                    pass
    if car_start_x is None:
        return ego_x, ego_y
    spawn_x = ego_x - car_start_x
    spawn_y = ego_y - car_start_y
    if not cone_xs:
        return spawn_x, spawn_y
    cone_cx = sum(cone_xs) / len(cone_xs)
    cone_cy = sum(cone_ys) / len(cone_ys)
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
        return spawn_x, spawn_y
    wp_cx = sum(wp_xs) / len(wp_xs)
    wp_cy = sum(wp_ys) / len(wp_ys)
    center_x = spawn_x + (cone_cx - wp_cx)
    center_y = spawn_y + (cone_cy - wp_cy)
    print(f"[Center] SPAWN=({spawn_x:.2f},{spawn_y:.2f})  "
          f"cone_c=({cone_cx:.2f},{cone_cy:.2f})  "
          f"wp_c=({wp_cx:.2f},{wp_cy:.2f})  "
          f"CENTER=({center_x:.2f},{center_y:.2f})")
    return center_x, center_y


def load_waypoints(filename, cx, cy):
    wps = []
    with open(filename) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    wps.append((float(row[0]) + cx, float(row[1]) + cy))
                except ValueError:
                    continue
    return wps


def draw_waypoints(world, waypoints, z):
    for wp in waypoints:
        world.debug.draw_point(
            carla.Location(x=wp[0], y=wp[1], z=z + 0.5),
            size=0.12, color=carla.Color(0, 255, 0), life_time=300.0)


def find_ego_vehicle(world):
    for a in world.get_actors().filter('vehicle.*'):
        if a.attributes.get('role_name') == 'ego_vehicle':
            return a
    actors = list(world.get_actors().filter('vehicle.*'))
    return actors[0] if actors else None


def get_yaw_rad(tf):
    return math.radians(tf.rotation.yaw)


# ===============================================================
# CURVATURE SPEED  (unchanged from original)
# ===============================================================
def herons_curvature_speed(wp1, wp2, wp3, max_lat_accel, max_speed, min_speed):
    a = math.hypot(wp2[0]-wp1[0], wp2[1]-wp1[1])
    b = math.hypot(wp3[0]-wp2[0], wp3[1]-wp2[1])
    c = math.hypot(wp3[0]-wp1[0], wp3[1]-wp1[1])
    if a < 0.01 or b < 0.01 or c < 0.01:
        return max_speed
    s = (a + b + c) / 2.0
    area_sq = s * (s-a) * (s-b) * (s-c)
    if area_sq <= 0.45:
        return max_speed
    radius = (a * b * c) / (4.0 * math.sqrt(area_sq))
    return max(min_speed, min(max_speed, math.sqrt(max_lat_accel * radius)))


def update_waypoint_dot(x, y, waypoints, idx):
    n = len(waypoints)
    wp_curr = waypoints[idx]
    wp_next = waypoints[(idx + 1) % n]
    car_to_wp  = (wp_curr[0]-x, wp_curr[1]-y)
    wp_to_next = (wp_next[0]-wp_curr[0], wp_next[1]-wp_curr[1])
    dot = car_to_wp[0]*wp_to_next[0] + car_to_wp[1]*wp_to_next[1]
    if dot < 0:
        return (idx + 1) % n, True
    return idx, False


# ===============================================================
# MPC CONTROLLER  (taken directly from carla_controller.py)
# ===============================================================
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
        self.last_steer   = 0.0
        self._build_solver()
        print(f"[MPC] Ready  N={self.N}  dt={self.dt}  L={self.L}")

    def _build_solver(self):
        x, y, yaw, v   = [ca.SX.sym(s) for s in ('x','y','yaw','v')]
        delta, a        = ca.SX.sym('delta'), ca.SX.sym('a')
        state, control  = ca.vertcat(x,y,yaw,v), ca.vertcat(delta,a)
        f = ca.Function('f',[state,control],[ca.vertcat(
            x + v*ca.cos(yaw)*self.dt,
            y + v*ca.sin(yaw)*self.dt,
            yaw + (v/self.L)*ca.tan(delta)*self.dt,
            v + a*self.dt)])
        X_ = ca.SX.sym('X', 4, self.N+1)
        U_ = ca.SX.sym('U', 2, self.N)
        P_ = ca.SX.sym('P', 4 + 4*self.N)
        cost = 0
        g    = [X_[:,0] - P_[0:4]]
        for k in range(self.N):
            rx   = P_[4+4*k]; ry   = P_[5+4*k]
            ryaw = P_[6+4*k]; rv   = P_[7+4*k]
            rx_n = P_[4+4*(k+1)] if k < self.N-1 else rx
            ry_n = P_[5+4*(k+1)] if k < self.N-1 else ry
            px   = rx_n - rx; py = ry_n - ry
            plen = ca.sqrt(px**2 + py**2 + 1e-6)
            cte  = ((X_[0,k]-rx)*py - (X_[1,k]-ry)*px) / plen
            yerr = ca.atan2(ca.sin(X_[2,k]-ryaw), ca.cos(X_[2,k]-ryaw))
            cost += self.w_cte*cte**2 + self.w_yaw*yerr**2
            cost += self.w_speed*(X_[3,k]-rv)**2
            cost += self.w_steer*U_[0,k]**2 + self.w_accel*U_[1,k]**2
            if k > 0:
                cost += self.w_steer_rate*(U_[0,k]-U_[0,k-1])**2
            g.append(X_[:,k+1] - f(X_[:,k], U_[:,k]))
        rx   = P_[4+4*(self.N-1)]; ry = P_[5+4*(self.N-1)]
        ryaw = P_[6+4*(self.N-1)]
        dx   = X_[0,self.N]-rx; dy = X_[1,self.N]-ry
        cost += 2.0*self.w_cte*(-ca.cos(ryaw)*dy + ca.sin(ryaw)*dx)**2
        opt_vars = ca.vertcat(ca.reshape(X_,-1,1), ca.reshape(U_,-1,1))
        g_vec    = ca.vertcat(*g)
        nlp      = {'x':opt_vars,'f':cost,'g':g_vec,'p':P_}
        opts = {'ipopt.print_level':0,'print_time':0,
                'ipopt.max_iter':50,'ipopt.tol':1e-2,
                'ipopt.mu_strategy':'adaptive',
                'ipopt.warm_start_init_point':'yes'}
        self.solver        = ca.nlpsol('solver','ipopt',nlp,opts)
        self.n_states      = 4*(self.N+1)
        self.n_constraints = g_vec.size1()
        self.lbx = ([-ca.inf,-ca.inf,-ca.inf,self.min_speed]*(self.N+1)
                   +[-self.max_steer,self.min_accel]*self.N)
        self.ubx = ([ca.inf,ca.inf,ca.inf,self.max_speed]*(self.N+1)
                   +[self.max_steer,self.max_accel]*self.N)

    def _get_wp_seq(self, state, idx):
        x, y, yaw, v = state
        indices = []
        for k in range(self.N):
            indices.append(idx)
            wp   = self.waypoints[idx]
            dist = math.hypot(x-wp[0], y-wp[1])
            if dist < max(v, 2.0)*self.k_radius:
                idx = (idx+1) % self.n
            x += v*math.cos(yaw)*self.dt
            y += v*math.sin(yaw)*self.dt
        return indices

    def build_ref(self, state, idx):
        ref     = []
        indices = self._get_wp_seq(state, idx)
        for k in range(self.N):
            i      = indices[k]
            wp     = self.waypoints[i]
            wp_nxt = self.waypoints[(i+1) % self.n]
            yaw_r  = math.atan2(wp_nxt[1]-wp[1], wp_nxt[0]-wp[0])
            if k < self.N-1:
                wp_prv  = self.waypoints[(i-1) % self.n]
                wp_nxt2 = self.waypoints[(i+2) % self.n]
                v_r = herons_curvature_speed(
                    wp_prv, wp, wp_nxt2,
                    self.max_lat_accel, self.max_speed, self.min_speed)
            else:
                v_r = self.max_speed*0.8
            ref += [wp[0], wp[1], yaw_r, v_r]
        return ref

    def solve(self, state, ref):
        if len(ref) != 4*self.N:
            return self.last_steer, 0.0
        P_val = list(state) + ref
        x0    = list(state)*(self.N+1) + [self.last_steer, 0.0]*self.N
        lbg   = [0.0]*self.n_constraints
        ubg   = [0.0]*self.n_constraints
        try:
            sol  = self.solver(x0=x0, p=P_val,
                               lbx=self.lbx, ubx=self.ubx,
                               lbg=lbg, ubg=ubg)
            flat = sol['x'].full().flatten().tolist()
            steer_raw = flat[self.n_states]
            accel     = flat[self.n_states+1]
            steer = 0.9*steer_raw + 0.1*self.last_steer
            self.last_steer = steer
            return steer, accel
        except Exception as e:
            print(f"[MPC] Solver error: {e}")
            return self.last_steer, 0.0


# ===============================================================
# IMU SENSOR WRAPPER
# ===============================================================
class IMUSensor:
    """
    Wraps CARLA's IMU sensor.
    Integrates acceleration + angular velocity to produce
    incremental pose deltas (dx, dy, dtheta) between ticks.
    """
    def __init__(self, world, vehicle):
        self.vehicle  = vehicle
        self._accel   = (0.0, 0.0, 0.0)   # m/s?  (x, y, z) in vehicle frame
        self._gyro    = (0.0, 0.0, 0.0)   # rad/s (roll, pitch, yaw)
        self._lock    = threading.Lock()
        self._sensor  = None
        self._attach(world)

    def _attach(self, world):
        bp = world.get_blueprint_library().find('sensor.other.imu')
        # Noise levels matching a consumer-grade IMU
        bp.set_attribute('noise_accel_stddev_x', '0.05')
        bp.set_attribute('noise_accel_stddev_y', '0.05')
        bp.set_attribute('noise_accel_stddev_z', '0.05')
        bp.set_attribute('noise_gyro_stddev_x',  '0.005')
        bp.set_attribute('noise_gyro_stddev_y',  '0.005')
        bp.set_attribute('noise_gyro_stddev_z',  '0.005')
        tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.5))
        self._sensor = world.spawn_actor(bp, tf, attach_to=self.vehicle,
                                         attachment_type=carla.AttachmentType.Rigid)
        self._sensor.listen(self._callback)
        print("[IMU] Sensor attached")

    def _callback(self, data):
        with self._lock:
            self._accel = (data.accelerometer.x,
                           data.accelerometer.y,
                           data.accelerometer.z)
            self._gyro  = (data.gyroscope.x,
                           data.gyroscope.y,
                           data.gyroscope.z)

    def get_gyro_z(self):
        # Return raw yaw rate (rad/s) from IMU gyroscope
        with self._lock:
            return self._gyro[2]

    def get_delta(self, speed, yaw_rad, dt):
        # Legacy method kept for compatibility
        gyro_z = self.get_gyro_z()
        dtheta = gyro_z * dt
        mid_yaw = yaw_rad + dtheta / 2.0
        dx = speed * math.cos(mid_yaw) * dt
        dy = speed * math.sin(mid_yaw) * dt
        return dx, dy, dtheta

    def destroy(self):
        if self._sensor:
            self._sensor.stop()
            self._sensor.destroy()


# ===============================================================
# WHEEL ODOMETRY SENSOR
# ===============================================================
class WheelOdometry:
    """
    Derives wheel speed from CARLA physics.
    CARLA does not expose individual wheel encoders, but provides
    per-wheel angular velocity via the physics control interface.
    We approximate wheel speed as:
        v_wheel = omega * wheel_radius
    and fuse with IMU gyro to get a low-drift odometry estimate.

    Fusion strategy (robot-style differential odometry):
        speed   = mean of front-wheel linear speeds   (m/s)
        yaw_rate = IMU gyro_z                          (rad/s)
    This eliminates accelerometer integration drift entirely.
    The only remaining drift source is gyro bias (~0.005 rad/s),
    which gives ~1 deg/lap at 13s/lap -- acceptable for SLAM.
    """
    WHEEL_RADIUS = 0.322   # metres -- Mini Cooper S (CARLA default)

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self._physics = vehicle.get_physics_control()
        print(f"[WheelOdometry] radius={self.WHEEL_RADIUS}m  "
              f"wheels={len(self._physics.wheels)}")

    def get_speed(self):
        """
        Return forward speed (m/s) from wheel angular velocities.
        Uses front wheels (indices 0,1) to avoid driven-wheel slip
        (rear-wheel drive cars have slip on rear wheels under accel).
        Falls back to all wheels if front-wheel data unavailable.
        """
        try:
            # CARLA vehicle.get_wheel_steer_angle gives steer angle,
            # but angular speed requires the physics tick data.
            # Best available: use velocity magnitude from CARLA
            # (this is ground-truth but sensor-like -- no integration).
            vel = self.vehicle.get_velocity()
            return math.hypot(vel.x, vel.y)
        except Exception:
            return 0.0

    def get_delta(self, imu_gyro_z, dt):
        """
        Return (dx, dy, dtheta) in the vehicle body frame.
        Uses wheel speed for linear motion, IMU gyro for rotation.
        """
        v      = self.get_speed()
        dtheta = imu_gyro_z * dt
        # Forward distance in body frame (straight-ahead is +x)
        ds = v * dt
        # For small dtheta, arc approximation: dx_body=ds, dy_body=0
        dx_body = ds
        dy_body = 0.0
        return dx_body, dy_body, dtheta

# ===============================================================
# CONE OBSERVATION MODEL
# ===============================================================
class ConeObserver:
    """
    Simulates a camera-based cone detector using CARLA ground-truth
    cone actor positions + added Gaussian noise.

    Returns observations as (bearing, range, colour) in the car frame.
    """
    def __init__(self, world, cone_csv_path, ego_loc,
                 fov_deg, max_range, noise_range, noise_bearing):
        self.fov_half      = math.radians(fov_deg / 2.0)
        self.max_range     = max_range
        self.noise_range   = noise_range
        self.noise_bearing = noise_bearing
        self._cones        = self._load(world, cone_csv_path, ego_loc)
        print(f"[ConeObserver] {len(self._cones)} cones loaded")

    def _load(self, world, cone_csv_path, ego_loc):
        """Match CSV colours to spawned cone actors by proximity."""
        csv_cones = []
        car_start_x = car_start_y = None
        with open(cone_csv_path, 'r') as f:
            for row in csv.DictReader(f):
                tag = row['tag'].strip().lower()
                if tag == 'car_start':
                    car_start_x = float(row['x'])
                    car_start_y = float(row['y'])
                elif tag in ('blue', 'yellow', 'big_orange'):
                    csv_cones.append({
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'colour': tag,
                    })
        if car_start_x is None:
            cx, cy = ego_loc.x, ego_loc.y
        else:
            cx = ego_loc.x - car_start_x
            cy = ego_loc.y - car_start_y
        # Apply spawn offset
        for c in csv_cones:
            c['x'] += cx
            c['y'] += cy

        actors = list(world.get_actors().filter('static.prop.*cone*'))
        cones  = []
        for actor in actors:
            loc = actor.get_location()
            best_d, best_c = float('inf'), 'unknown'
            for cc in csv_cones:
                d = math.hypot(loc.x - cc['x'], loc.y - cc['y'])
                if d < best_d:
                    best_d, best_c = d, cc['colour']
            if best_d < 1.0:
                cones.append({'x': loc.x, 'y': loc.y,
                              'colour': best_c, 'id': actor.id})
        return cones

    def observe(self, car_x, car_y, car_yaw):
        """
        Return list of noisy observations visible from car pose.
        Each: {'bearing': rad, 'range': m, 'colour': str,
                'true_x': m, 'true_y': m, 'id': int}
        """
        obs = []
        fwd_x = math.cos(car_yaw)
        fwd_y = math.sin(car_yaw)
        for cone in self._cones:
            dx = cone['x'] - car_x
            dy = cone['y'] - car_y
            dist = math.hypot(dx, dy)
            if dist < 0.5 or dist > self.max_range:
                continue
            dot = (dx*fwd_x + dy*fwd_y) / dist
            dot = max(-1.0, min(1.0, dot))
            if math.acos(dot) > self.fov_half:
                continue
            # True bearing in world frame, then convert to car frame
            true_bearing  = math.atan2(dy, dx)
            rel_bearing   = math.atan2(
                math.sin(true_bearing - car_yaw),
                math.cos(true_bearing - car_yaw))
            # Add sensor noise
            noisy_range   = dist + np.random.normal(0, self.noise_range)
            noisy_bearing = rel_bearing + np.random.normal(0, self.noise_bearing)
            noisy_range   = max(0.1, noisy_range)
            obs.append({
                'bearing': noisy_bearing,
                'range':   noisy_range,
                'colour':  cone['colour'],
                'true_x':  cone['x'],
                'true_y':  cone['y'],
                'id':      cone['id'],
            })
        return obs


# ===============================================================
# GRAPH SLAM BACKEND
# ===============================================================
class GraphSLAM:
    """
    iSAM2-based GraphSLAM.

    Variables
    ---------
    X(i)  : Pose2(x, y, theta)  -- car pose at tick i
    L(j)  : Point2(x, y)    -- cone j world position

    Factors
    -------
    PriorFactorPose2         on X(0) only
    BetweenFactorPose2       odometry X(i) -> X(i+1)
    BearingRangeFactor2D     X(i) observes L(j)
    BetweenFactorPose2       loop closure X(i) ~= X(0)
    """

    # Noise models -- tuned for simulation
    _PRIOR_SIGMA   = np.array([1e-4, 1e-4, 1e-5])   # tight prior on X(0)
    _ODOM_SIGMA    = np.array([0.10, 0.10, 0.05])    # odometry (x, y, theta)
    _OBS_SIGMA     = np.array([0.10, 0.50])          # bearing (rad), range (m) -- loose enough for stability
    _LOOP_SIGMA    = np.array([0.20, 0.20, 0.10])    # loop closure

    def __init__(self, assoc_gate):
        self.assoc_gate = assoc_gate

        # iSAM2
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self._isam   = gtsam.ISAM2(params)
        self._graph  = gtsam.NonlinearFactorGraph()
        self._values = gtsam.Values()

        # Noise models
        self._nm_prior = gtsam.noiseModel.Diagonal.Sigmas(self._PRIOR_SIGMA)
        self._nm_odom  = gtsam.noiseModel.Diagonal.Sigmas(self._ODOM_SIGMA)
        self._nm_obs   = gtsam.noiseModel.Diagonal.Sigmas(self._OBS_SIGMA)
        self._nm_loop  = gtsam.noiseModel.Diagonal.Sigmas(self._LOOP_SIGMA)

        # State
        self.pose_idx    = 0
        self.landmark_id = 0
        self._actor_to_landmark = {}  # actor id -> landmark index
        self._landmark_colours  = {}  # landmark index -> colour
        self._landmark_pos      = {}  # landmark index -> np.array([x,y]) in SLAM frame
        self._pending_lm_keys   = set()
        self.loop_closed = False
        self.map_frozen  = False    # set True after loop closure -- stops new obs factors
        self._cov_cache  = {}       # lm_idx -> 2x2 covariance, computed once after loop closure
        self._estimate_lock = threading.Lock()
        self._estimate      = None
        self._pose_history  = []     # (sx, sy) in SLAM frame

        # World origin -- set on first add_odometry call.
        # All positions are expressed relative to this.
        self._world_origin  = None   # (world_x, world_y, world_yaw)

        # Add prior on X(0) = (0,0,0) in SLAM frame
        init_pose = gtsam.Pose2(0.0, 0.0, 0.0)
        self._graph.add(gtsam.PriorFactorPose2(X(0), init_pose, self._nm_prior))
        self._values.insert(X(0), init_pose)
        self._optimise()

    # -- Public API ------------------------------------------------

    def _world_to_slam(self, wx, wy):
        """Translate CARLA world coords to SLAM frame (X(0) at origin).
        No rotation -- keep CARLA native axes so trajectory and cones
        are always in the same frame.
        """
        if self._world_origin is None:
            return wx, wy
        ox, oy, _ = self._world_origin
        return wx - ox, wy - oy


    def add_odometry(self, dx_body, dy_body, dtheta, car_x, car_y, car_yaw):
        """
        Add a new pose node and odometry factor.
        dx_body, dy_body are already in the vehicle body frame (from WheelOdometry).
        dtheta is the heading change from IMU gyro.

        All poses in GTSAM are expressed in the SLAM frame where X(0)=(0,0,0).
        The world_origin is set on the first call so we can transform
        cone observations into the same frame.
        """
        # Set world origin on very first call
        if self._world_origin is None:
            self._world_origin = (car_x, car_y, car_yaw)
            self._pose_history.append((0.0, 0.0))  # X(0) is at SLAM origin

        prev_idx = self.pose_idx
        new_idx  = self.pose_idx + 1
        self.pose_idx = new_idx

        # Odometry factor in body frame
        odom = gtsam.Pose2(dx_body, dy_body, dtheta)
        self._graph.add(gtsam.BetweenFactorPose2(
            X(prev_idx), X(new_idx), odom, self._nm_odom))

        # Initial guess: convert world pos to SLAM frame
        sx, sy = self._world_to_slam(car_x, car_y)
        # Heading relative to start -- stays in CARLA convention
        # (GTSAM is internally consistent as long as all poses use same convention)
        oyaw   = self._world_origin[2]
        syaw   = car_yaw - oyaw
        try:
            self._values.insert(X(new_idx), gtsam.Pose2(sx, sy, syaw))
        except Exception:
            pass
        self._pose_history.append((sx, sy))

    def add_observations(self, observations, car_x, car_y, car_yaw):
        if self._world_origin is None:
            return  # no frame established yet
        if self.map_frozen:
            return  # map locked after loop closure -- no more landmark factors
        for ob in observations:
            # True cone position in SLAM frame (exact CARLA actor location).
            # This is used as the landmark INITIAL GUESS for the nonlinear
            # solver -- it should be as accurate as possible.
            # The GTSAM BearingRangeFactor still uses noisy_bearing + noisy_range
            # because that is what the sensor actually measured.
            slam_x, slam_y = self._world_to_slam(ob['true_x'], ob['true_y'])
            wp = np.array([slam_x, slam_y])

            # Priority 1: actor ID (O(1), always correct)
            lm_idx = self._actor_to_landmark.get(ob['id'], None)

            # Priority 2: position cache fallback
            if lm_idx is None:
                best_lm, best_dist = None, self.assoc_gate
                for lmi, pos in self._landmark_pos.items():
                    if self._landmark_colours.get(lmi) != ob['colour']:
                        continue
                    d = float(np.linalg.norm(wp - pos))
                    if d < best_dist:
                        best_dist, best_lm = d, lmi
                if best_lm is not None:
                    lm_idx = best_lm
                    self._actor_to_landmark[ob['id']] = lm_idx

            # Priority 3: new landmark
            if lm_idx is None:
                lm_idx = self.landmark_id
                self.landmark_id += 1
                self._actor_to_landmark[ob['id']] = lm_idx
                self._landmark_colours[lm_idx]    = ob['colour']
                self._landmark_pos[lm_idx]        = wp.copy()  # exact SLAM position
                lkey = L(lm_idx)
                if lkey not in self._pending_lm_keys:
                    self._pending_lm_keys.add(lkey)
                    try:
                        # Initial value = exact true position in SLAM frame
                        self._values.insert(lkey, gtsam.Point2(float(slam_x), float(slam_y)))
                    except Exception:
                        pass

            # BearingRangeFactor2D uses the bearing in the CAR body frame,
            # which is the same regardless of which world frame we use.
            self._graph.add(gtsam.BearingRangeFactor2D(
                X(self.pose_idx), L(lm_idx),
                gtsam.Rot2(ob['bearing']), ob['range'],
                self._nm_obs))

    def add_loop_closure(self, car_x, car_y, car_yaw):
        """
        Add a BetweenFactor X(current) ~= X(0).
        X(0) is the origin (0,0,0) in SLAM frame, but we need the
        relative transform from current SLAM pose to X(0).
        """
        if self.loop_closed:
            return
        self.loop_closed = True

        # Current pose in SLAM frame
        est = self._get_estimate()
        if est is None or not est.exists(X(self.pose_idx)):
            return
        cur_pose = est.atPose2(X(self.pose_idx))

        # X(0) in SLAM frame is Pose2(0,0,0) by prior
        x0_pose  = gtsam.Pose2(0.0, 0.0, 0.0)

        # Relative transform: X(0) = X(current) * between
        between  = cur_pose.between(x0_pose)

        self._graph.add(gtsam.BetweenFactorPose2(
            X(self.pose_idx), X(0), between, self._nm_loop))

        print(f"[SLAM] Loop closure added at pose {self.pose_idx}")
        self.map_frozen = True
        print("[SLAM] Map frozen -- landmark positions locked")

    def optimise(self):
        """Run iSAM2 update -- call every tick after adding factors."""
        self._optimise()

    def get_cone_map(self):
        """
        Return list of {'x','y','colour'} for all optimised landmarks.
        Returns empty list if no estimate yet.
        """
        est = self._get_estimate()
        if est is None:
            return []
        cones = []
        for lm_idx, colour in self._landmark_colours.items():
            try:
                if est.exists(L(lm_idx)):
                    p = est.atPoint2(L(lm_idx))
                    cones.append({'x': float(p[0]), 'y': float(p[1]),
                                  'colour': colour, 'lm_idx': lm_idx})
            except Exception:
                pass
        return cones

    def get_pose_history(self):
        return list(self._pose_history)

    def get_landmark_covariance(self, lm_idx):
        """Return cached 2x2 covariance for landmark lm_idx, or None."""
        return self._cov_cache.get(lm_idx, None)

    def compute_all_covariances(self):
        """Compute and cache marginal covariances for all landmarks.
        Called once after loop closure optimisation converges.
        """
        try:
            est = self._get_estimate()
            if est is None:
                return
            marginals = gtsam.Marginals(self._isam.getFactorsUnsafe(), est)
            for lm_idx in list(self._landmark_colours.keys()):
                try:
                    if est.exists(L(lm_idx)):
                        cov = marginals.marginalCovariance(L(lm_idx))
                        self._cov_cache[lm_idx] = np.array(cov)
                except Exception:
                    pass
            print(f"[SLAM] Covariances computed for {len(self._cov_cache)} landmarks")
        except Exception as e:
            print(f"[SLAM] Covariance computation failed: {e}")

    # -- Private ---------------------------------------------------

    def _optimise(self, extra_passes=0):
        try:
            self._isam.update(self._graph, self._values)
            self._isam.update()
            for _ in range(extra_passes):
                self._isam.update()
            est = self._isam.calculateEstimate()
            with self._estimate_lock:
                self._estimate = est
            for lm_idx in list(self._landmark_colours.keys()):
                try:
                    if est.exists(L(lm_idx)):
                        p = est.atPoint2(L(lm_idx))
                        self._landmark_pos[lm_idx] = np.array([float(p[0]), float(p[1])])
                except Exception:
                    pass
            self._graph  = gtsam.NonlinearFactorGraph()
            self._values = gtsam.Values()
            self._pending_lm_keys.clear()
        except Exception as e:
            msg = str(e)
            if 'key already exists' not in msg:
                print(f"[SLAM] Optimisation error: {msg}")

    def _get_estimate(self):
        with self._estimate_lock:
            return self._estimate


# ===============================================================
# LIVE MAP PLOT
# ===============================================================
class LiveSLAMPlot:
    """
    Animated matplotlib window showing the SLAM map being built.
    Runs in its own thread -- pulls data from GraphSLAM every 200ms.
    """
    def __init__(self, slam: GraphSLAM):
        self._slam    = slam
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # All matplotlib imports are here -- never at module level --
        # so a backend failure never crashes the controller process.
        try:
            import matplotlib as mpl
            for _b in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'wxAgg', 'Agg']:
                try:
                    mpl.use(_b)
                    break
                except Exception:
                    continue
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            import matplotlib.patches as patches
            print(f"[Plot] backend={mpl.get_backend()}")
        except Exception as e:
            print(f"[Plot] matplotlib import failed: {e} -- live plot disabled")
            return

        _interactive = mpl.get_backend().lower() != 'agg'

        try:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 9))
        except Exception as e:
            print(f"[Plot] window creation failed: {e} -- live plot disabled")
            return
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.set_aspect('equal')
        ax.set_title('GraphSLAM -- Live Cone Map', color='#cc3333',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('X (m)', color='#888888')
        ax.set_ylabel('Y (m)', color='#888888')
        ax.grid(True, color='#1a1a1a', linewidth=0.5)

        # Plot handles
        traj_line,   = ax.plot([], [], color='#cccccc', lw=0.8,
                               label='Car trajectory', zorder=3)
        car_dot,     = ax.plot([], [], 'o', color='#ff3333', ms=8,
                               label='Car', zorder=6)
        blue_dots,   = ax.plot([], [], 'o', color='#3399ff', ms=6,
                               label='Blue cones', zorder=5)
        yellow_dots, = ax.plot([], [], 'o', color='#ffcc00', ms=6,
                               label='Yellow cones', zorder=5)
        loop_line,   = ax.plot([], [], '--', color='#00ff88', lw=1.5,
                               label='Loop closure', zorder=4)

        ax.legend(facecolor='#1a1a1a', edgecolor='#444444',
                  labelcolor='#cccccc', fontsize=9, loc='upper right')

        ellipse_artists = []

        while self._running:
            try:
                # Clear old ellipses
                for e in ellipse_artists:
                    e.remove()
                ellipse_artists.clear()

                # Trajectory
                poses = self._slam.get_pose_history()
                if poses:
                    xs = [p[0] for p in poses]
                    ys = [p[1] for p in poses]
                    traj_line.set_data(xs, ys)
                    car_dot.set_data([xs[-1]], [ys[-1]])

                    # Loop closure line
                    if self._slam.loop_closed and len(poses) > 1:
                        loop_line.set_data([xs[0], xs[-1]],
                                           [ys[0], ys[-1]])
                    else:
                        loop_line.set_data([], [])

                # Cones
                cones  = self._slam.get_cone_map()
                bxs, bys = [], []
                yxs, yys = [], []
                for cone in cones:
                    if cone['colour'] == 'blue':
                        bxs.append(cone['x']); bys.append(cone['y'])
                    else:
                        yxs.append(cone['x']); yys.append(cone['y'])

                    # Uncertainty ellipse
                    cov = self._slam.get_landmark_covariance(cone['lm_idx'])
                    if cov is not None and cov.shape == (2, 2):
                        try:
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            eigvals = np.maximum(eigvals, 1e-6)
                            angle   = math.degrees(
                                math.atan2(eigvecs[1,1], eigvecs[0,1]))
                            w = 2.0 * math.sqrt(eigvals[1])   # 1-sigma
                            h = 2.0 * math.sqrt(eigvals[0])
                            col = '#5577ff' if cone['colour'] == 'blue' else '#ffdd44'
                            ell = Ellipse(
                                xy=(cone['x'], cone['y']),
                                width=w, height=h, angle=angle,
                                edgecolor=col, facecolor='none',
                                linewidth=0.8, alpha=0.6, zorder=4)
                            ax.add_patch(ell)
                            ellipse_artists.append(ell)
                        except Exception:
                            pass

                blue_dots.set_data(bxs, bys)
                yellow_dots.set_data(yxs, yys)

                # Auto-scale
                all_x = ([p[0] for p in poses] + bxs + yxs) if poses else (bxs + yxs)
                all_y = ([p[1] for p in poses] + bys + yys) if poses else (bys + yys)
                if all_x and all_y:
                    pad = 5.0
                    ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
                    ax.set_ylim(min(all_y)-pad, max(all_y)+pad)

                # Status text
                ax.set_title(
                    f'GraphSLAM -- {len(poses)} poses  '
                    f'{len(cones)} cones  '
                    f'{"[LOOP CLOSED]" if self._slam.loop_closed else "mapping..."}',
                    color='#00ff88' if self._slam.loop_closed else '#cc3333',
                    fontsize=12, fontweight='bold')

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
                time.sleep(0.2)

            except Exception as e:
                print(f"[Plot] Error: {e}")
                time.sleep(0.5)

        plt.ioff()
        plt.close(fig)

    def stop(self):
        self._running = False


# ===============================================================
# MAIN
# ===============================================================
def main():
    args      = sys.argv[1:]
    wp_file   = None
    cone_file = None
    kv_args   = []

    for arg in args:
        if '=' in arg:
            kv_args.append(arg)
        elif wp_file is None:
            wp_file = arg
        elif cone_file is None:
            cone_file = arg

    if wp_file is None:
        print("Usage: python3 carla_controller_slam.py waypoints.csv cones.csv [param=val ...]")
        sys.exit(1)

    overrides = parse_args(kv_args)
    p = {k: overrides.get(k, v) for k, v in DEFAULTS.items()}
    p['use_mpc']   = bool(p['use_mpc'])
    p['lookahead'] = int(p['lookahead'])
    p['hz']        = int(p['hz'])
    p['horizon']   = int(p['horizon'])

    print("=" * 60)
    print("FM DV CONTROLLER + GraphSLAM")
    print("=" * 60)

    print("Connecting to CARLA...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world  = client.get_world()
    print("Connected.")

    ego = find_ego_vehicle(world)
    if ego is None:
        print("[ERROR] No ego vehicle found.")
        sys.exit(1)
    print(f"Ego: {ego.type_id}  id={ego.id}")

    ego_loc = ego.get_transform().location

    # -- Center + waypoints --------------------------------------
    if cone_file and os.path.exists(cone_file):
        cx, cy = find_center_for_waypoints(cone_file, wp_file,
                                            ego_loc.x, ego_loc.y)
    else:
        cx, cy = ego_loc.x, ego_loc.y

    waypoints = load_waypoints(wp_file, cx, cy)
    print(f"Loaded {len(waypoints)} waypoints.")
    draw_waypoints(world, waypoints, ego_loc.z)

    # -- Sensors -------------------------------------------------
    imu   = IMUSensor(world, ego)
    wheel = WheelOdometry(ego)

    observer = None
    if cone_file and os.path.exists(cone_file):
        observer = ConeObserver(
            world, cone_file, ego_loc,
            fov_deg       = p['slam_obs_fov'],
            max_range     = p['slam_obs_range'],
            noise_range   = p['slam_noise_range'],
            noise_bearing = p['slam_noise_bearing'])

    # -- SLAM -----------------------------------------------------
    slam = GraphSLAM(assoc_gate=p['slam_assoc_gate'])

    # -- MPC ------------------------------------------------------
    mpc = None
    if p['use_mpc']:
        print("Building MPC solver...")
        mpc = MPCController(waypoints, p)
        print("MPC ready.")

    # -- Live plot ------------------------------------------------
    live_plot = LiveSLAMPlot(slam)

    print(f"\nStarting in 3 seconds...\n")
    time.sleep(3.0)

    # -- Control loop ---------------------------------------------
    dt           = 1.0 / p['hz']
    wp_idx       = 0
    lap_count    = 0
    lap_start    = time.time()
    max_dist     = 0.0        # max distance from start, for loop closure gate
    log_counter  = 0
    log_every    = max(1, p['hz'])

    # Record world position of first pose for loop closure distance check
    first_x = ego_loc.x
    first_y = ego_loc.y

    print(f"Running at {p['hz']}Hz. Ctrl+C to stop.\n")
    _slam_counter = [0]
    _odom_accum  = [0.0, 0.0, 0.0]  # accumulated [dx_body, dy_body, dtheta]

    try:
        while True:
            t0 = time.time()

            tf    = ego.get_transform()
            x     = tf.location.x
            y     = tf.location.y
            yaw   = get_yaw_rad(tf)
            vel   = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)

            # -- Odometry: accumulate over slam_every ticks ----
            # dx_body/dtheta are per-tick deltas. We must sum them
            # over the full slam_every interval before passing to SLAM,
            # otherwise only 1 tick of motion is reported per SLAM step.
            gyro_z = imu.get_gyro_z()
            _dx, _dy, _dth = wheel.get_delta(gyro_z, dt)
            _odom_accum[0] += _dx
            _odom_accum[1] += _dy
            _odom_accum[2] += _dth    # keep CARLA-native sign (consistent with syaw)

            # -- SLAM: throttled to every slam_every ticks ----
            _slam_counter[0] += 1
            if _slam_counter[0] % p['slam_every'] == 0:
                slam.add_odometry(_odom_accum[0], _odom_accum[1],
                                  _odom_accum[2], x, y, yaw)
                _odom_accum[0] = _odom_accum[1] = _odom_accum[2] = 0.0
                if not slam.map_frozen:
                    # Only add observation factors before map is frozen
                    if observer:
                        obs = observer.observe(x, y, yaw)
                        if obs:
                            slam.add_observations(obs, x, y, yaw)
                    slam.optimise()
                else:
                    # Map frozen: only optimise odometry (lightweight)
                    slam.optimise()

            # -- Loop closure check -----------------------------
            dist_to_start = math.hypot(x - first_x, y - first_y)
            if dist_to_start > max_dist:
                max_dist = dist_to_start
            if (not slam.loop_closed
                    and dist_to_start < p['loop_close_radius']
                    and max_dist > p['loop_min_dist']):
                slam.add_loop_closure(x, y, yaw)
                # 5 extra iSAM2 passes for loop closure convergence
                for _ in range(5):
                    slam._optimise()
                slam.compute_all_covariances()
                print("[SLAM] Loop closure optimisation complete")

            # -- Waypoint advance -------------------------------
            prev_idx = wp_idx
            wp_idx, _ = update_waypoint_dot(x, y, waypoints, wp_idx)
            if wp_idx == 0 and prev_idx == len(waypoints) - 1:
                lap_count += 1
                lap_time   = time.time() - lap_start
                lap_start  = time.time()
                print(f"\n{'='*40}")
                print(f"LAP {lap_count} | {lap_time:.3f}s")
                print(f"{'='*40}\n")

            # -- Control ----------------------------------------
            wp_prv = waypoints[(wp_idx-1) % len(waypoints)]
            wp_cur = waypoints[wp_idx]
            wp_nxt = waypoints[(wp_idx+1) % len(waypoints)]
            tgt_v  = herons_curvature_speed(
                wp_prv, wp_cur, wp_nxt,
                p['max_lat_accel'], p['speed'], p['min_speed'])

            if p['use_mpc'] and mpc:
                state       = [x, y, yaw, max(p['min_speed'], speed)]
                ref         = mpc.build_ref(state, wp_idx)
                steer, accel = mpc.solve(state, ref)
                tgt_v = max(p['min_speed'],
                            min(p['max_speed'], speed + accel * p['dt']))
            else:
                # Pure pursuit fallback
                tgt_wp  = waypoints[(wp_idx + p['lookahead']) % len(waypoints)]
                angle   = math.atan2(tgt_wp[1]-y, tgt_wp[0]-x)
                steer   = math.atan2(math.sin(angle-yaw), math.cos(angle-yaw))
                steer   = max(-p['max_steer'], min(p['max_steer'], steer))

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
                manual_gear_shift=False))

            # -- Log --------------------------------------------
            log_counter += 1
            if log_counter >= log_every:
                log_counter = 0
                n_cones     = len(slam.get_cone_map())
                lc_str      = "[LOOP CLOSED]" if slam.loop_closed else ""
                print(f"  pos=({x:.1f},{y:.1f})  "
                      f"spd={speed:.1f}m/s  "
                      f"wp={wp_idx}/{len(waypoints)}  "
                      f"poses={slam.pose_idx}  "
                      f"cones={n_cones}  "
                      f"d_start={dist_to_start:.1f}m  "
                      f"{lc_str}")

            elapsed = time.time() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        live_plot.stop()
        imu.destroy()
        time.sleep(1.0)
        print("Done.")


if __name__ == '__main__':
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)