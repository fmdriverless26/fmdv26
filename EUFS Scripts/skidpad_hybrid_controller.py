"""

THis code makes the car go forward until it sees the origin.

Once it sees origin and crosses it, it follows the waypoints according to that origin. 



"""





import rclpy

from rclpy.node import Node

import numpy as np

import math

import casadi as ca



# ROS 2 Messages

from nav_msgs.msg import Odometry

from ackermann_msgs.msg import AckermannDriveStamped

from eufs_msgs.msg import ConeArrayWithCovariance



# Import the raw TU Berlin waypoints from your local file

try:

    from skidpad_path_data import BASE_SKIDPAD_PATH

except ImportError:

    print("ERROR: tu_berlin_path.py not found in the same directory.")



# ==========================================

# CASADI MPC OPTIMIZER

# ==========================================

class CasadiMPC:

    def __init__(self):

        # Increased resolution (dt) for tighter, faster curves

        self.N = 6

        self.dt = 0.15

        self.L = 1.58



        # TIGHT SKIDPAD WEIGHTS

        self.w_cte = 200.0        # HUGE penalty for drifting wide off the line

        self.w_yaw = 20.0         # Keep heading aligned

        self.w_speed = 0.1

        self.w_steer = 1.0        # Let the car steer deeply (low penalty)

        self.w_accel = 0.1

        self.w_steer_rate = 100.0 # Let the car turn the wheel FAST (was 1500!)



        # Limits

        self.max_steer = 0.5

        self.max_accel = 2.0

        self.min_accel = -3.0

        self.max_speed = 5.0      # Safe skidpad speed

        self.min_speed = 1.0

        self.max_lat_accel = 17.0

        

        self.k_radius = 0.35

        self.last_steer = 0.0



        self._build_solver()



    def _build_solver(self):

        x = ca.SX.sym('x')

        y = ca.SX.sym('y')

        yaw = ca.SX.sym('yaw')

        v = ca.SX.sym('v')

        delta = ca.SX.sym('delta')

        a = ca.SX.sym('a')



        state = ca.vertcat(x, y, yaw, v)

        control = ca.vertcat(delta, a)



        # Kinematic Bicycle Model

        x_next = x + v * ca.cos(yaw) * self.dt

        y_next = y + v * ca.sin(yaw) * self.dt

        yaw_next = yaw + (v / self.L) * ca.tan(delta) * self.dt  

        v_next = v + a * self.dt



        f = ca.Function('f', [state, control], [ca.vertcat(x_next, y_next, yaw_next, v_next)])



        X = ca.SX.sym('X', 4, self.N + 1)

        U = ca.SX.sym('U', 2, self.N)

        P = ca.SX.sym('P', 4 + 4 * self.N)



        cost = 0

        g = [X[:,0] - P[0:4]]



        for k in range(self.N):

            rx, ry, ryaw, rv = P[4 + 4*k], P[5 + 4*k], P[6 + 4*k], P[7 + 4*k]



            if k < self.N - 1:

                rx_next, ry_next = P[4 + 4*(k+1)], P[5 + 4*(k+1)]

            else:

                rx_next, ry_next = rx, ry



            path_x, path_y = rx_next - rx, ry_next - ry

            car_x, car_y = X[0, k] - rx, X[1, k] - ry



            path_len = ca.sqrt(path_x**2 + path_y**2 + 1e-6)

            cte = (car_x * path_y - car_y * path_x) / path_len

            yaw_error = ca.atan2(ca.sin(X[2,k] - ryaw), ca.cos(X[2,k] - ryaw))

            v_err = X[3,k] - rv



            cost += self.w_cte * cte**2 + self.w_yaw * yaw_error**2 + self.w_speed * v_err**2

            cost += self.w_steer * U[0,k]**2 + self.w_accel * U[1,k]**2



            if k > 0:

                cost += self.w_steer_rate * (U[0,k] - U[0,k-1])**2



            g.append(X[:,k+1] - f(X[:,k], U[:,k]))



        # Terminal Cost

        rx, ry, ryaw = P[4 + 4*(self.N-1)], P[5 + 4*(self.N-1)], P[6 + 4*(self.N-1)]

        dx, dy = X[0,self.N] - rx, X[1,self.N] - ry

        cte_term = -ca.cos(ryaw) * dy + ca.sin(ryaw) * dx

        cost += 2.0 * self.w_cte * cte_term**2



        opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))

        g = ca.vertcat(*g)



        nlp = {'x': opt_vars, 'f': cost, 'g': g, 'p': P}

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 50, 'ipopt.tol': 1e-2}



        self.solver = ca.nlpsol('solver','ipopt',nlp,opts)

        self.n_states = 4*(self.N+1)

        self.n_constraints = g.size1()

        

        self.lbx, self.ubx = [], []

        for _ in range(self.N + 1):

            self.lbx += [-ca.inf, -ca.inf, -ca.inf, self.min_speed]

            self.ubx += [ca.inf, ca.inf, ca.inf, self.max_speed]

        for _ in range(self.N):

            self.lbx += [-self.max_steer, self.min_accel]

            self.ubx += [self.max_steer, self.max_accel]



    def build_ref(self, current_state, waypoints, start_idx):

        ref = []

        x, y, yaw, v = current_state

        n = len(waypoints)

        

        idx = start_idx

        

        for k in range(self.N):

            wp = waypoints[idx]

            # Advance index if we predict we will pass this waypoint

            if math.hypot(x - wp[0], y - wp[1]) < max(v, 2.0) * self.k_radius:

                idx = min(idx + 5, n - 1) # Jump ahead slightly on the dense array

                wp = waypoints[idx]

            

            x += v * math.cos(yaw) * self.dt

            y += v * math.sin(yaw) * self.dt



            next_idx = min(idx + 5, n - 1)

            wp_next = waypoints[next_idx]

            

            # Heading target

            if next_idx == idx and idx > 0:

                yaw_ref = math.atan2(waypoints[idx][1] - waypoints[idx-1][1], waypoints[idx][0] - waypoints[idx-1][0])

            else:

                yaw_ref = math.atan2(wp_next[1] - wp[1], wp_next[0] - wp[0])

            

            # Use max speed (MPC handles lateral constraints implicitly)

            v_ref = self.max_speed * 0.9 

            

            ref += [wp[0], wp[1], yaw_ref, v_ref]

            

        return ref



    def solve(self, state, ref):

        if len(ref) != 4 * self.N: return self.last_steer, 0.0

        P_val = state + ref

        x0 = state[0:4] * (self.N + 1) + [self.last_steer, 0.0] * self.N



        try:

            sol = self.solver(x0=x0, p=P_val, lbx=self.lbx, ubx=self.ubx, 

                              lbg=[0.0]*self.n_constraints, ubg=[0.0]*self.n_constraints)

            sol_x = sol['x'].full().flatten().tolist()

            steer_raw, accel = sol_x[self.n_states], sol_x[self.n_states + 1]

            

            self.last_steer = 0.9 * steer_raw + 0.1 * self.last_steer

            return self.last_steer, accel

        except Exception as e:

            return self.last_steer, 0.0





# ==========================================

# MAIN ROS 2 NODE 

# ==========================================

class SimpleMPCController(Node):

    def __init__(self):

        super().__init__('simple_mpc_controller')

        

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/cmd', 10)

        self.create_subscription(Odometry, '/ground_truth/odom', self.odom_callback, 10)

        self.create_subscription(ConeArrayWithCovariance, '/ground_truth/cones', self.cone_callback, 10)



        self.mpc = CasadiMPC()

        

        self.x = self.y = self.yaw = self.v = 0.0

        self.current_wp_idx = None 

        self.path_generated = False

        self.is_finished = False  # <--- ADDED FINISH FLAG

        self.waypoints = np.zeros((0, 2))



        self.timer = self.create_timer(0.02, self.control_loop) 



    def odom_callback(self, msg):

        self.x = msg.pose.pose.position.x

        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

        self.v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)



    def cone_callback(self, msg):

        # WAIT FOR THE TIMING GATE (4 Big Orange Cones)

        if not self.path_generated and len(msg.big_orange_cones) >= 4:

            # Calculate the exact center of the gate

            gate_x = np.mean([c.point.x for c in msg.big_orange_cones])

            gate_y = np.mean([c.point.y for c in msg.big_orange_cones])

            

            

            cx = gate_x 

            cy = gate_y

            

            # Shift the entire array to the true origin.

            self.waypoints = BASE_SKIDPAD_PATH + np.array([cx+1.0, cy])

            self.path_generated = True

            

            self.get_logger().info(f"Loaded Raw 4-Lap Array! Shifted to true origin ({cx:.2f}, {cy:.2f})")



    def get_closest_waypoint_index(self):

        """ 

        Strictly searches FORWARD in the array. 

        This prevents the MPC from jumping across the figure-8 intersection.

        """

        car_pos = np.array([self.x, self.y])

        n = len(self.waypoints)



        if self.current_wp_idx is None:

            distances = np.linalg.norm(self.waypoints - car_pos, axis=1)

            self.current_wp_idx = int(np.argmin(distances))

            return self.current_wp_idx



        # Search window of 150 points (~7.5 meters forward on a dense array)

        search_window = min(150, n - self.current_wp_idx)

        indices = [self.current_wp_idx + i for i in range(search_window)]

        window_points = self.waypoints[indices]

        

        distances = np.linalg.norm(window_points - car_pos, axis=1)

        local_min = int(np.argmin(distances))

        

        self.current_wp_idx = indices[local_min]

        return self.current_wp_idx



    def control_loop(self):

        # 1. STOPPING LOGIC: If we reached the end, command full brakes!

        if self.is_finished:

            msg = AckermannDriveStamped()

            msg.drive.steering_angle = 0.0

            msg.drive.speed = 0.0

            msg.drive.acceleration = -3.0  # Max Brake

            self.cmd_pub.publish(msg)

            return



        # 2. Wait for path to generate

        if not self.path_generated or len(self.waypoints) == 0:

            msg = AckermannDriveStamped()

            msg.drive.steering_angle = 0.0

            msg.drive.speed = 3.0

            msg.drive.acceleration = 1.0

            self.cmd_pub.publish(msg)

            return



        # 3. Path is generated, find our spot

        start_idx = self.get_closest_waypoint_index()



        # Check if we are within the last 20 waypoints of the exit straight

        if start_idx >= len(self.waypoints) - 20:

            self.get_logger().info("🏁 Track Complete! Applying Brakes. 🏁")

            self.is_finished = True

            return



        # 4. Feed to MPC and Drive

        state = [self.x, self.y, self.yaw, max(self.mpc.min_speed, self.v)]

        ref = self.mpc.build_ref(state, self.waypoints, start_idx)

        steer, accel = self.mpc.solve(state, ref)



        msg = AckermannDriveStamped()

        msg.drive.steering_angle = float(steer) * 0.92

        msg.drive.acceleration = float(accel)

        self.cmd_pub.publish(msg)



def main():

    rclpy.init()

    node = SimpleMPCController()

    try: 

        rclpy.spin(node)

    except KeyboardInterrupt: 

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()



if __name__ == '__main__':

    main()
