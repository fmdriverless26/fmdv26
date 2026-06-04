import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math
import csv
import time
import casadi as ca
import os


class MPC:
    def __init__(self):

        # MPC Parameters
        self.N = 10
        self.dt = 0.1

        # wheelbase
        self.L = 1.58

        # weights
        self.w_cte = 20.0
        self.w_yaw = 25.0
        self.w_speed = 5.0
        self.w_steer = 20.0
        self.w_accel = 0.5

        # Constraints
        self.max_steer = 0.5
        self.max_accel = 5.0
        self.max_speed = 8.0
        self.min_speed = 0.0

        # 🔥 added for ABC/4A curvature speed
        self.max_lat_accel = 2.5

        # ----- CasADi model -----
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')

        steer = ca.SX.sym('steer')
        a = ca.SX.sym('a')

        state = ca.vertcat(x, y, yaw, v)
        control = ca.vertcat(steer, a)

        x_next = x + v * ca.cos(yaw) * self.dt
        y_next = y + v * ca.sin(yaw) * self.dt
        yaw_next = yaw + (v / self.L) * ca.tan(steer) * self.dt
        v_next = v + a * self.dt

        self.f = ca.Function('f', [state, control],
                             [ca.vertcat(x_next, y_next, yaw_next, v_next)])

        X = ca.SX.sym('X', 4, self.N + 1)
        U = ca.SX.sym('U', 2, self.N)

        P = ca.SX.sym('P', 4 + self.N * 4)

        cost = 0
        g = []

        g.append(X[:, 0] - P[0:4])

        for k in range(self.N):

            rx = P[4 + k*4]
            ry = P[5 + k*4]
            ryaw = P[6 + k*4]
            rv = P[7 + k*4]

            dx = X[0, k+1] - rx
            dy = X[1, k+1] - ry
            yaw_error = X[2, k+1] - ryaw
            v_error = X[3, k+1] - rv

            cte_error = ca.sqrt(dx**2 + dy**2)

            cost += self.w_cte * cte_error**2
            cost += self.w_yaw * yaw_error**2
            cost += self.w_speed * v_error**2

            x_next_pred = self.f(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next_pred)

        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        g = ca.vertcat(*g)

        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.n_states = 4 * (self.N + 1)
        self.n_controls = 2 * self.N
        self.n_vars = self.n_states + self.n_controls
        self.n_constraints = g.size()[0]

    # ============================================================
    # ABC / 4A Curvature Speed
    # ============================================================

    def curvature_speed(self, wp1, wp2, wp3):

        a = math.hypot(wp2[0] - wp1[0], wp2[1] - wp1[1])
        b = math.hypot(wp3[0] - wp2[0], wp3[1] - wp2[1])
        c = math.hypot(wp3[0] - wp1[0], wp3[1] - wp1[1])

        if a < 0.01 or b < 0.01 or c < 0.01:
            return self.max_speed

        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 0:
            return self.max_speed

        area = math.sqrt(area_sq)

        radius = (a * b * c) / (4.0 * area)

        v_ref = math.sqrt(self.max_lat_accel * radius)

        v_ref = min(self.max_speed, max(self.min_speed, v_ref))

        return v_ref

    def solve(self, state, ref):

        P_val = state + ref
        x0 = [0.0] * self.n_vars

        lbg = [0.0] * self.n_constraints
        ubg = [0.0] * self.n_constraints

        sol = self.solver(
            x0=x0,
            p=P_val,
            lbg=lbg,
            ubg=ubg
        )

        sol_x = sol['x'].full().flatten()

        steer = sol_x[self.n_states + 0]
        accel = sol_x[self.n_states + 1]

        return steer, accel

    def build_ref(self, x, y , v, idx):
        ref = []

        v = max(v, 2.0)

        for k in range(self.N):

            wp = self.waypoints[idx]
            wp_next = self.waypoints[(idx + 1) % self.n]
            wp_next2 = self.waypoints[(idx + 2) % self.n]

            # --- advance waypoint if passed ---
            car_to_wp = (wp[0] - x, wp[1] - y)
            wp_dir = (wp_next[0] - wp[0], wp_next[1] - wp[1])

            dot = car_to_wp[0]*wp_dir[0] + car_to_wp[1]*wp_dir[1]

            if dot < 0:
                idx = (idx + 1) % self.n
                wp = self.waypoints[idx]
                wp_next = self.waypoints[(idx + 1) % self.n]
                wp_next2 = self.waypoints[(idx + 2) % self.n]

            # --- reference yaw ---
            dx = wp_next[0] - wp[0]
            dy = wp_next[1] - wp[1]
            yaw_ref = math.atan2(dy, dx)

            # --- curvature based speed ---
            v_ref = self.curvature_speed(wp, wp_next, wp_next2)
            v_ref = min(v_ref, self.max_speed)

            ref += [wp[0], wp[1], yaw_ref, v_ref]

            # --- forward simulate position ---
            x += v * math.cos(yaw_ref) * self.dt
            y += v * math.sin(yaw_ref) * self.dt

        return ref


class Controller(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_callback, 10)

        self.waypoints = self.load_wp('small_track_waypoints.csv')
        self.idx = 0

        self.mpc = MPC()

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0

        self.prev = 0

        self.timer = self.create_timer(0.02, self.loop)

    def load_wp(self, filename):
        file_path = os.path.join(
            os.path.dirname(__file__),
            filename
        )
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            return [(float(r[0]), float(r[1])) for r in reader if len(r) >= 2]

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v = math.hypot(vx, vy)

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

    def wp_reached(self):
        self.prev = self.idx

        wp_curr = self.waypoints[self.idx]
        wp_next = self.waypoints[(self.idx + 1) % len(self.waypoints)]

        car_to_wp = (wp_curr[0] - self.x, wp_curr[1] - self.y)
        wp_to_next = (wp_next[0] - wp_curr[0], wp_next[1] - wp_curr[1])

        dot = car_to_wp[0] * wp_to_next[0] + car_to_wp[1] * wp_to_next[1]

        if dot < 0:
            self.idx = (self.idx + 1) % len(self.waypoints)

            if self.idx == 0 and self.prev == len(self.waypoints) - 1:
                self.lap_complete()

    def lap_complete(self):
        lap_time = time.time() - self.lap_start_time
        self.lap_times.append(lap_time)
        self.lap_count += 1

        self.get_logger().info(f"LAP {self.lap_count} COMPLETE")
        self.get_logger().info(f"Time: {lap_time:.2f}s \n Max Speed: {self.max_speed_seen:.2f} m/s")

        self.lap_start_time = time.time()

    def loop(self):
        try:
            self.wp_reached()

            state = [self.x, self.y, self.yaw, self.v]

            self.mpc.waypoints = self.waypoints
            self.mpc.n = len(self.waypoints)
            ref = self.mpc.build_ref(self.x, self.y, self.v, self.idx)

            steer, accel = self.mpc.solve(state, ref)

            target_v = self.v + accel * self.mpc.dt

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = float(steer)
            msg.drive.speed = float(target_v)
            msg.drive.acceleration = float(accel)
            self.cmd_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"MPC error: {e}")


def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
