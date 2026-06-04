import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
import os
import math
import csv
import time

class BaseWaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_callback, 10)

        self.waypoints = self.load_wp('small_track_waypoints.csv')
        self.current_waypoint_idx = 0

        self.kp_steering = 0.5
        self.ki_steering = 0.01
        self.kd_steering = 0.05
        self.max_steering = 0.5

        self.k_stanley = 0.1
        self.k_soft = 1.0
        self.k_radius=0.35

        self.base_speed = 13.0
        self.min_speed = 5.0
        self.max_safe_speed = 13.0
        self.error_sensitivity = 1.0

        self.kp_speed = 2.0
        self.max_accel = 5.0
        self.min_accel = -5.0

        self.lap_times = []
        self.lap_count = 0
        self.max_speed_seen = 0.0
        self.consecutive_clean_laps = 0
        self.last_lap_had_issues = False

        self.steer_integral = 0.0
        self.prev_yaw_error = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.yaw_error = 0.0
        self.last_pos = [0.0, 0.0]
        self.total_distance = 0.0

        self.lap_start_time = time.time()
        self.timer = self.create_timer(0.02, self.control_loop)

    def load_wp(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            return [(float(r[0]), float(r[1])) for r in reader if len(r) >= 2]

    def next_wp(self, idx):
        return self.waypoints[(idx + 1) % len(self.waypoints)]

    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = math.hypot(vx, vy)
        self.max_speed_seen = max(self.max_speed_seen, self.speed)

        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        dx = new_x - self.last_pos[0]
        dy = new_y - self.last_pos[1]
        self.total_distance += math.hypot(dx, dy)

        self.x, self.y = new_x, new_y
        self.last_pos = [new_x, new_y]

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

    def wp_reached(self):
        wp = self.waypoints[self.current_waypoint_idx]

        distance_to_wp = math.hypot(self.x - wp[0], self.y - wp[1])

        if distance_to_wp < self.k_radius*self.speed:
            prev_idx = self.current_waypoint_idx
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)

            if self.current_waypoint_idx == 0 and prev_idx == len(self.waypoints) - 1:
                self.lap_completion()

    def lap_completion(self):
        lap_time = time.time() - self.lap_start_time
        self.lap_times.append(lap_time)
        self.lap_count += 1

        self.get_logger().info(f"LAP {self.lap_count} COMPLETE")
        self.get_logger().info(f"Time: {lap_time:.2f}s \n Max Speed: {self.max_speed_seen:.2f} m/s")

        if lap_time > 16.0 or self.max_speed_seen > self.max_safe_speed * 1.1:
            self.last_lap_had_issues = True
            self.consecutive_clean_laps = 0
            self.base_speed = max(11.0, self.base_speed - 0.3)
            self.get_logger().info("Slowing down")
        else:
            self.last_lap_had_issues = False
            self.consecutive_clean_laps += 1

            if self.consecutive_clean_laps >= 2 and self.base_speed < self.max_safe_speed:
                self.base_speed = min(self.max_safe_speed, self.base_speed + 0.2)
                self.get_logger().info("Increasing speed")

        self.max_speed_seen = 0.0
        self.yaw_error = 0.0
        self.steer_integral = 0.0
        self.lap_start_time = time.time()

    def crosstract_error(self, wp, wp_next):
        path_x = wp_next[0] - wp[0]
        path_y = wp_next[1] - wp[1]
        car_x = self.x - wp[0]
        car_y = self.y - wp[1]

        path_len = math.hypot(path_x, path_y)

        return (car_x * path_y - car_y * path_x) / path_len

    def steering(self):
        wp = self.waypoints[self.current_waypoint_idx]
        wp_next = self.next_wp(self.current_waypoint_idx)

        desired_yaw = math.atan2(wp_next[1] - self.y, wp_next[0] - self.x)
        yaw_error = desired_yaw - self.yaw

        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        self.yaw_error = yaw_error

        dt = 0.02
        self.steer_integral += yaw_error * dt
        self.steer_integral = max(-1.0, min(1.0, self.steer_integral))

        yaw_error_rate = (yaw_error - self.prev_yaw_error) / dt
        self.prev_yaw_error = yaw_error

        steer = (self.kp_steering * yaw_error + self.ki_steering * self.steer_integral + self.kd_steering * yaw_error_rate)

        if self.speed > 11.0:
            steer *= 1.1

        crosstract_error = self.crosstract_error(wp, wp_next)
        steer += math.atan2(self.k_stanley * crosstract_error, self.speed + self.k_soft)

        return max(-self.max_steering, min(self.max_steering, steer))

    def desired_speed(self):
        base = self.base_speed * (0.9 if self.last_lap_had_issues else 1.0)
        return max(self.min_speed, base / (1 + self.error_sensitivity * abs(self.yaw_error)))

    def acceleration_cmd(self, desired_speed):
        kp = self.kp_speed * (0.7 if self.last_lap_had_issues else 1.0)
        accel = kp * (desired_speed - self.speed)
        return max(self.min_accel, min(self.max_accel, accel))

    def control_loop(self):
        self.wp_reached()

        steer = self.steering()
        desired_speed = self.desired_speed()
        accel = self.acceleration_cmd(desired_speed)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer
        msg.drive.speed = desired_speed
        msg.drive.acceleration = accel
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = BaseWaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
