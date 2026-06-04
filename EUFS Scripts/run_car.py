import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math
import csv
import time
import casadi as ca


class MPC:
    def __init__(self, N, dt, L):
        self.N  = N
        self.dt = dt
        self.L  = L

        # Weights
        w_cte       = 80.0
        w_yaw       = 20.0
        w_speed     = 8.0
        w_steer     = 20.0
        w_accel     = 2.0
        w_steer_rate = 5000.0
        w_accel_rate = 0.0

        x   = ca.SX.sym('x')
        y   = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v   = ca.SX.sym('v')
        state = ca.vertcat(x, y, yaw, v)

        delta = ca.SX.sym('delta')
        a     = ca.SX.sym('a')
        control = ca.vertcat(delta, a)

        # Bicycle model - linearized, tan(delta) ≈ delta
        x_next   = x + v * ca.cos(yaw) * dt
        y_next   = y + v * ca.sin(yaw) * dt
        yaw_next = yaw + (v / L) * delta * dt
        v_next   = v + a * dt

        f = ca.Function('f', [state, control],
                        [ca.vertcat(x_next, y_next, yaw_next, v_next)])

        self.f = f

        X = ca.SX.sym('X', 4, N + 1)
        U = ca.SX.sym('U', 2, N)
        P = ca.SX.sym('P', 4 + 4 * N)  # [x, y, yaw, v, x0, y0, yaw0, v0, ...]

        cost = 0
        g    = []

        g.append(X[:, 0] - P[0:4])

        for k in range(N):
            rx   = P[4 + 4*k]
            ry   = P[5 + 4*k]
            ryaw = P[6 + 4*k]
            rv   = P[7 + 4*k]

            if k < N - 1:
                rx_next = P[4 + 4*(k+1)]
                ry_next = P[5 + 4*(k+1)]
                # Perpendicular CTE using cross product
                path_x   = rx_next - rx
                path_y   = ry_next - ry
                car_x    = X[0, k] - rx
                car_y    = X[1, k] - ry
                path_len = ca.sqrt(path_x**2 + path_y**2 + 1e-6)
                cte      = (car_x * path_y - car_y * path_x) / path_len
                yaw_error = ca.atan2(
                    ca.sin(X[2,k] - ryaw),
                    ca.cos(X[2,k] - ryaw)
                )

            else:
                # Last step: path direction comes from stored ryaw
                car_x = X[0, k] - rx
                car_y = X[1, k] - ry
                cte   = -ca.sin(ryaw) * car_x + ca.cos(ryaw) * car_y  # perpendicular using ryaw
                yaw_error = ca.atan2(
                    ca.sin(X[2,k] - ryaw),
                    ca.cos(X[2,k] - ryaw)
                )


            v_err = X[3, k] - rv

            cost += w_cte        * cte**2
            cost += w_yaw        * yaw_error**2
            cost += w_speed      * v_err**2
            cost += w_steer      * U[0, k]**2
            cost += w_accel      * U[1, k]**2

            if k > 0:
                cost += w_steer_rate * (U[0, k] - U[0, k-1])**2
                cost += w_accel_rate * (U[1, k] - U[1, k-1])**2

            g.append(X[:, k+1] - f(X[:, k], U[:, k]))

        # Terminal cost - heavier penalty on final position
        rx_f = P[4 + 4*(N-1)]
        ry_f = P[5 + 4*(N-1)]
        dx_f = X[0, N] - rx_f
        dy_f = X[1, N] - ry_f
        cost += 3.0 * w_cte * (dx_f**2 + dy_f**2)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g        = ca.vertcat(*g)
        nlp      = {'x': opt_vars, 'f': cost, 'g': g, 'p': P}

        opts = {
            'ipopt.max_iter':              50,
            'ipopt.print_level':            0,
            'ipopt.tol':                   1e-2,
            'ipopt.acceptable_tol':        5e-2,
            'print_time':                   0,
            'ipopt.linear_solver':        'mumps',
            'ipopt.mu_strategy':          'adaptive',
            'ipopt.warm_start_init_point': 'yes'
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbx = []
        self.ubx = []
        for _ in range(N + 1):
            self.lbx += [-ca.inf, -ca.inf, -ca.inf, 1.5]
            self.ubx += [ ca.inf,  ca.inf,  ca.inf, 13.0]
        for _ in range(N):
            self.lbx += [-0.5, -5.0]
            self.ubx += [ 0.5,  5.0]

        self.lbg = [0] * g.size1()
        self.ubg = [0] * g.size1()

        self.prev_solution = None
        self.last_steer    = 0.0

    def solve(self, state, ref_traj):
        p = [float(s) for s in state] + [float(r) for r in ref_traj]

        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            x0 = []
            for _ in range(self.N + 1):
                x0 += [float(state[0]), float(state[1]),
                       float(state[2]), float(state[3])]
            for _ in range(self.N):
                x0 += [self.last_steer, 0.0]

        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)
        except Exception:
            return self.last_steer * 0.8, -1.0

        sol_list = [float(v) for v in sol['x'].full().flatten()]

        control_start  = 4 * (self.N + 1)
        delta_raw      = sol_list[control_start]
        a              = sol_list[control_start + 1]

        delta              = 0.6 * delta_raw + 0.4 * self.last_steer
        self.last_steer    = delta
        self.prev_solution = sol_list

        return delta, a


class Navigator(Node):
    def __init__(self):
        super().__init__('mpc_navigator')

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_cb, 10)

        try:
            self.waypoints = self.load_wp('small_track_waypoints.csv')
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            rclpy.shutdown()
            return

        if len(self.waypoints) < 10:
            self.get_logger().error(f"Need at least 10 waypoints, got {len(self.waypoints)}")
            rclpy.shutdown()
            return

        self.idx = 0
        self.n   = len(self.waypoints)

        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.v   = 0.0

        self.mpc = MPC(N=8, dt=0.1, L=1.58)

        self.lap_start     = time.time()
        self.laps          = 0
        self.base_speed    = 6.0
        self.max_lat_accel = 3.0

        self.total_cte = 0.0
        self.cte_count = 0

        self.timer = self.create_timer(0.1, self.loop)

    def load_wp(self, name):
        with open(name) as f:
            return [(float(r[0]), float(r[1]))
                    for r in csv.reader(f) if len(r) >= 2]

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

        self.v = math.hypot(msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y)

    def is_waypoint_behind(self, wp_idx):
        wp_curr    = self.waypoints[wp_idx]
        wp_next    = self.waypoints[(wp_idx + 1) % self.n]
        car_to_wp  = (wp_curr[0] - self.x,  wp_curr[1] - self.y)
        wp_to_next = (wp_next[0] - wp_curr[0], wp_next[1] - wp_curr[1])
        dot = car_to_wp[0] * wp_to_next[0] + car_to_wp[1] * wp_to_next[1]
        return dot < 0

    def update_current_waypoint(self):
        if self.is_waypoint_behind(self.idx):
            prev     = self.idx
            self.idx = (self.idx + 1) % self.n
            if self.idx == 0 and prev == self.n - 1:
                self.lap_complete()

    def calculate_radius_herons(self, wp1, wp2, wp3):
        a = math.hypot(wp2[0] - wp1[0], wp2[1] - wp1[1])
        b = math.hypot(wp3[0] - wp2[0], wp3[1] - wp2[1])
        c = math.hypot(wp3[0] - wp1[0], wp3[1] - wp1[1])

        if a < 0.01 or b < 0.01 or c < 0.01:
            return float('inf')

        s       = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 0:
            return float('inf')

        return (a * b * c) / (4.0 * math.sqrt(area_sq))

    def calculate_reference_speed(self, wp_idx):
        radius = self.calculate_radius_herons(
            self.waypoints[(wp_idx - 1) % self.n],
            self.waypoints[wp_idx],
            self.waypoints[(wp_idx + 1) % self.n]
        )
        radius = max(radius, 8.0)
        v_ref  = math.sqrt(self.max_lat_accel * radius)

        # Lookahead pre-braking: also check next 2 waypoints ahead
        # This makes the car slow down BEFORE a corner, not during it
        for lookahead in range(1, 3):
            future_idx    = (wp_idx + lookahead) % self.n
            future_radius = self.calculate_radius_herons(
                self.waypoints[(future_idx - 1) % self.n],
                self.waypoints[future_idx],
                self.waypoints[(future_idx + 1) % self.n]
            )
            future_radius = max(future_radius, 8.0)
            future_v      = math.sqrt(self.max_lat_accel * future_radius)
            v_ref         = min(v_ref, future_v)  # Take minimum - brake early

        return max(2.5, min(self.base_speed * 0.9, v_ref))

    def build_ref(self):
        ref    = []
        x, y   = self.x, self.y
        wp_idx = self.idx

        for _ in range(self.mpc.N):
            wp   = self.waypoints[wp_idx]
            dist = math.hypot(x - wp[0], y - wp[1])

            if dist < max(self.v, 2.0) * self.mpc.dt * 1.5:
                wp_idx = (wp_idx + 1) % self.n
                wp     = self.waypoints[wp_idx]

            x += self.v * math.cos(self.yaw) * self.mpc.dt
            y += self.v * math.sin(self.yaw) * self.mpc.dt

            # Yaw reference from current waypoint to next
            wp_next  = self.waypoints[(wp_idx + 1) % self.n]
            yaw_ref  = math.atan2(wp_next[1] - wp[1], wp_next[0] - wp[0])

            v_ref = self.calculate_reference_speed(wp_idx)
            ref  += [float(wp[0]), float(wp[1]), float(yaw_ref), float(v_ref)]

        return ref

    def calculate_cte(self):
        min_dist = float('inf')

        for i in range(self.n):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[(i + 1) % self.n]

            A = self.x - wp1[0]
            B = self.y - wp1[1]
            C = wp2[0] - wp1[0]
            D = wp2[1] - wp1[1]

            len_sq = C * C + D * D
            if len_sq < 1e-6:
                continue

            param = (A * C + B * D) / len_sq
            param = max(0.0, min(1.0, param))

            xx = wp1[0] + param * C
            yy = wp1[1] + param * D

            min_dist = min(min_dist, math.hypot(self.x - xx, self.y - yy))

        self.total_cte += min_dist
        self.cte_count += 1

        return min_dist

    def lap_complete(self):
        t       = time.time() - self.lap_start
        avg_cte = self.total_cte / max(1, self.cte_count)
        self.laps += 1

        if avg_cte < 0.8 and t < 30.0:
            self.base_speed = min(13.0, self.base_speed + 0.3)
            self.get_logger().info(
                f"Lap {self.laps}: {t:.1f}s | Avg CTE: {avg_cte:.2f}m | Speed ↑ {self.base_speed:.1f}m/s"
            )
        elif avg_cte > 1.5 or t > 45.0:
            self.base_speed = max(4.0, self.base_speed - 0.5)
            self.get_logger().info(
                f"Lap {self.laps}: {t:.1f}s | Avg CTE: {avg_cte:.2f}m | Speed ↓ {self.base_speed:.1f}m/s"
            )
        else:
            self.get_logger().info(f"Lap {self.laps}: {t:.1f}s | Avg CTE: {avg_cte:.2f}m")

        self.lap_start = time.time()
        self.total_cte = 0.0
        self.cte_count = 0

    def loop(self):
        try:
            self.update_current_waypoint()

            cte = self.calculate_cte()

            if cte > 10.0:
                self.get_logger().error(f"Too far off track! CTE: {cte:.2f}m")
                msg = AckermannDriveStamped()
                msg.drive.speed        = 0.0
                msg.drive.acceleration = -3.0
                self.cmd_pub.publish(msg)
                return

            state = [float(self.x), float(self.y),
                     float(self.yaw), max(1.5, float(self.v))]
            ref   = self.build_ref()

            steer, accel = self.mpc.solve(state, ref)

            target_v = max(1.5, min(self.base_speed, self.v + accel * self.mpc.dt))

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = float(steer)
            msg.drive.speed          = float(target_v)
            msg.drive.acceleration   = float(accel)
            self.cmd_pub.publish(msg)

            if self.cte_count % 40 == 0:
                self.get_logger().info(
                    f"V: {self.v:.1f}m/s | Steer: {steer:.3f}rad | "
                    f"CTE: {cte:.2f}m | WP: {self.idx}/{self.n}"
                )

        except Exception as e:
            self.get_logger().error(f"Control loop failed: {str(e)}")
            msg = AckermannDriveStamped()
            msg.drive.speed        = 0.0
            msg.drive.acceleration = -5.0
            self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()