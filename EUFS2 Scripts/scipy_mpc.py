import rclpy
import math
import numpy as np
import csv
from scipy.optimize import minimize

from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        # Publisher (CONFIRMED WORKING)
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            '/cmd',
            10
        )

        # Subscriber (FIXED)
        self.subscriber_ = self.create_subscription(
            Odometry,
            '/odom',
            self.car_callback,
            10
        )

        # MPC params
        self.N = 5
        self.dt = 0.1
        self.Lf = 1.58 / 2

        self.ref_v = 6.0

        # Weights
        self.w_cte = 30
        self.w_epsi = 30
        self.w_v = 10
        self.w_delta = 23
        self.w_a = 5

        # Reference path
        self.path = [
            (4, 0.5), (6.1, 0.46), (8.96, 0.63), (12.9, 0.54), (16.5, 0.91),
            (20.14, 2.02), (23.03, 3.01), (25.72, 4.12), (27.05, 3.85),
            (29.24, 3.03), (30.23, 2.15), (31.74, 0.11), (32.04, -1.34),
            (31.85, -4.60), (30.93, -7.44), (30.10, -9.54), (28.42, -11.91),
            (26.17, -13.64), (23.26, -15.09), (20.75, -16.53),
            (16.42, -18.71), (12.72, -20.56), (9.16, -22.59),
            (6.34, -24.41), (3.44, -25.23), (2.01, -24.97),
            (-0.24, -23.59), (-1.13, -22.77), (-3.03, -20.33),
            (-3.38, -16.96), (-3.54, -15.76), (-3.28, -12.89),
            (-2.86, -10.3), (-2.56, -7.25), (-2.59, -4.27),
            (-1.47, -1.72), (-0.7, -0.64)
        ]

        self.current_waypoint_idx = 0

        # CSV logging
        self.csv_file = open('mpc_controls.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['steering_angle', 'acceleration'])

        self.get_logger().info('✅ MPC Controller initialized')

    def car_callback(self, msg):
        self.msg = msg
        self.run_mpc()

    def advance_waypoint(self, x, y, waypoint_idx):
        path_length = len(self.path)

        while True:
            current_idx = waypoint_idx % path_length
            next_idx = (waypoint_idx + 1) % path_length

            A = self.path[(current_idx - 1) % path_length]
            B = (x, y)
            C = self.path[current_idx]

            AB = (B[0] - A[0], B[1] - A[1])
            AC = (C[0] - A[0], C[1] - A[1])

            dot_product = AB[0] * AC[0] + AB[1] * AC[1]
            AC_mag_sq = AC[0]**2 + AC[1]**2

            projection_ratio = dot_product / AC_mag_sq if AC_mag_sq != 0 else 0

            if projection_ratio > 1.0:
                waypoint_idx = next_idx
            else:
                break

        return waypoint_idx

    def next_points(self, x, y):
        self.current_waypoint_idx = self.advance_waypoint(
            x, y, self.current_waypoint_idx
        )
        return self.path[self.current_waypoint_idx]

    def run_mpc(self):
        # Extract state (FIXED FOR ODOM)
        px = self.msg.pose.pose.position.x
        py = self.msg.pose.pose.position.y

        q = self.msg.pose.pose.orientation
        psi = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        vx = self.msg.twist.twist.linear.x
        vy = self.msg.twist.twist.linear.y
        v = math.sqrt(vx**2 + vy**2)

        # Get target
        target_x, target_y = self.next_points(px, py)

        path_angle = math.atan2(target_y - py, target_x - px)

        theta = psi - path_angle
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi

        theta = abs(theta)

        dynamic_velocity = self.ref_v * (math.cos(theta) ** 4)

        cte = math.sqrt((px - target_x)**2 + (py - target_y)**2)

        heading_error = path_angle - psi
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        state = [px, py, psi, v, cte, heading_error, dynamic_velocity]

        solution = self.solve_mpc(state)

        steer = float(np.clip(solution[0], -0.5, 0.5))
        accel = float(max(solution[1], 0.5))  # prevent stall

        # Log
        self.csv_writer.writerow([steer, accel])
        self.csv_file.flush()

        # Publish
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.drive.steering_angle = steer
        msg.drive.acceleration = accel

        self.publisher_.publish(msg)

        self.get_logger().info(
            f"steer={steer:.3f}, accel={accel:.3f}, v={v:.2f}, cte={cte:.2f}"
        )

    def solve_mpc(self, state):
        x0, y0, psi0, v0, cte0, epsi0, target_v = state

        N = self.N
        dt = self.dt
        Lf = self.Lf

        initial_idx = self.current_waypoint_idx

        def cost_fn(u):
            x, y, psi, v = x0, y0, psi0, v0
            cost = 0
            idx = initial_idx

            for _ in range(N):
                delta, a = u

                x += v * math.cos(psi) * dt
                y += v * math.sin(psi) * dt
                psi += v * delta / Lf * dt
                v += a * dt

                idx = self.advance_waypoint(x, y, idx)
                ref_x, ref_y = self.path[idx % len(self.path)]

                cte = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
                epsi = math.atan2(ref_y - y, ref_x - x) - psi

                while epsi > math.pi:
                    epsi -= 2 * math.pi
                while epsi < -math.pi:
                    epsi += 2 * math.pi

                cost += self.w_cte * cte**2
                cost += self.w_epsi * epsi**2
                cost += self.w_v * (v - target_v)**2
                cost += self.w_delta * delta**2
                cost += self.w_a * a**2

            return cost

        bounds = [(-0.5, 0.5), (-2.0, 5.0)]
        initial_guess = [0.0, 1.0]

        result = minimize(cost_fn, initial_guess, bounds=bounds, method='SLSQP')

        if result.success:
            return result.x
        else:
            self.get_logger().warn(f"MPC failed: {result.message}")
            return [0.0, 0.5]

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()


def main(args=None):
    rclpy.init(args=args)

    node = MPCController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
