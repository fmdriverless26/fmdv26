import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import casadi as ca
import math
import csv
import os
import time


#MPC

class MPC:

    def __init__(self, waypoints):

        # Horizon 
        self.N = 4
        self.dt = 0.3
        self.L = 1.58

        # Weights 
        self.w_cte = 50.0
        self.w_yaw = 20.0
        self.w_speed = 0.1
        self.w_steer = 10.0
        self.w_accel = 0.1
        self.w_steer_rate = 1500.0 

        # Limits 
        self.max_steer = 0.5
        self.max_accel = 2.0
        self.min_accel = -3.0
        self.max_speed = 14.0
        self.min_speed = 1.0

        self.max_lat_accel = 17.0
        
        self.k_radius = 0.35

        self.waypoints = waypoints
        self.n = len(waypoints)

        self.last_steer = 0.0
        #self.prev_solution = None

        self._build_solver()

    #SOLVER
    def _build_solver(self):

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')

        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')

        state = ca.vertcat(x, y, yaw, v)
        control = ca.vertcat(delta, a)

        # Bicycle model
        x_next = x + v * ca.cos(yaw) * self.dt
        y_next = y + v * ca.sin(yaw) * self.dt
        yaw_next = yaw + (v / self.L) * ca.tan(delta) * self.dt  
        v_next = v + a * self.dt

        f = ca.Function('f', [state, control], [ca.vertcat(x_next, y_next, yaw_next, v_next)])

        X = ca.SX.sym('X', 4, self.N + 1)
        U = ca.SX.sym('U', 2, self.N)

        P = ca.SX.sym('P', 4 + 4 * self.N)

        cost = 0
        g = []

        g.append(X[:,0] - P[0:4])

        for k in range(self.N):

            rx = P[4 + 4*k]
            ry = P[5 + 4*k]
            ryaw = P[6 + 4*k]
            rv = P[7 + 4*k]

            if k < self.N - 1:
                rx_next = P[4 + 4*(k+1)]
                ry_next = P[5 + 4*(k+1)]
            else:
                # For last step, reuse current direction
                rx_next = rx
                ry_next = ry

            path_x = rx_next - rx
            path_y = ry_next - ry

            car_x = X[0, k] - rx
            car_y = X[1, k] - ry

            path_len = ca.sqrt(path_x**2 + path_y**2 + 1e-6)

            cte = (car_x * path_y - car_y * path_x) / path_len

            yaw_error = ca.atan2(
                ca.sin(X[2,k] - ryaw),
                ca.cos(X[2,k] - ryaw)
            )


            v_err = X[3,k] - rv

            # Cost
            cost += self.w_cte * cte**2
            cost += self.w_yaw * yaw_error**2
            cost += self.w_speed * v_err**2
            cost += self.w_steer * U[0,k]**2
            cost += self.w_accel * U[1,k]**2

            if k > 0:
                cost += self.w_steer_rate * (U[0,k] - U[0,k-1])**2

            # Dynamics
            g.append(X[:,k+1] - f(X[:,k], U[:,k]))

        # Terminal cost
        rx = P[4 + 4*(self.N-1)]
        ry = P[5 + 4*(self.N-1)]
        ryaw = P[6 + 4*(self.N-1)]
        
        dx = X[0,self.N] - rx
        dy = X[1,self.N] - ry
        path_dx = ca.cos(ryaw)
        path_dy = ca.sin(ryaw)
        cte_term = -path_dx * dy + path_dy * dx
        cost += 2.0 * self.w_cte * cte_term**2

        opt_vars = ca.vertcat(ca.reshape(X,-1,1),
                              ca.reshape(U,-1,1))

        g = ca.vertcat(*g)

        nlp = {'x': opt_vars, 'f': cost, 'g': g, 'p': P}

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-2,
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.warm_start_init_point': 'yes'
        }

        self.solver = ca.nlpsol('solver','ipopt',nlp,opts)

        self.n_states = 4*(self.N+1)
        self.n_controls = 2*self.N
        self.n_vars = self.n_states + self.n_controls
        self.n_constraints = g.size1()
        
        # Bounds
        self.lbx = []
        self.ubx = []
        for _ in range(self.N + 1):
            self.lbx += [-ca.inf, -ca.inf, -ca.inf, self.min_speed]
            self.ubx += [ca.inf, ca.inf, ca.inf, self.max_speed]
        for _ in range(self.N):
            self.lbx += [-self.max_steer, self.min_accel]
            self.ubx += [self.max_steer, self.max_accel]

     #CURVATURE SPEED USING HERONS FORMULA
    
    def herons_curvature_speed(self, wp1, wp2, wp3):

        a = math.hypot(wp2[0] - wp1[0], wp2[1] - wp1[1])
        b = math.hypot(wp3[0] - wp2[0], wp3[1] - wp2[1])
        c = math.hypot(wp3[0] - wp1[0], wp3[1] - wp1[1])
        
        if a < 0.01 or b < 0.01 or c < 0.01:
            return self.max_speed
        
        # Semi-perimeter
        s = (a + b + c) / 2.0
        
        # Heron's formula for area
        area_sq = s * (s - a) * (s - b) * (s - c)
        
        if area_sq <= 0.45:
            return self.max_speed
        
        area = math.sqrt(area_sq)
        
        # Radius from circumradius formula: R = (abc) / (4*Area)
        radius = (a * b * c) / (4.0 * area)
                
        v_ref = math.sqrt(self.max_lat_accel * radius)
        
        # Clip to bounds
        v_ref = max(self.min_speed, min(self.max_speed, v_ref))
        
        return v_ref
        
    #DYNAMIC WAYPOINT DETECTION
    
    def get_waypoint_sequence(self, current_state, current_idx):

        waypoint_indices = []
        x, y, yaw, v = current_state
        idx = current_idx
        
        for k in range(self.N):
            waypoint_indices.append(idx)
            
            # Check if we'll reach this waypoint in one timestep
            wp = self.waypoints[idx]
            dist = math.hypot(x - wp[0], y - wp[1])
            
            # If close enough, advance to next waypoint
            if dist < max(v, 2.0) * self.k_radius :
                idx = (idx + 1) % self.n
            
            # Rough prediction of next position
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
        
        return waypoint_indices

    #REFERENCE TRAJECTORY

    def build_ref(self, current_state, current_idx):

        ref = []
        
        # Get predicted waypoint sequence
        wp_indices = self.get_waypoint_sequence(current_state, current_idx)
        
        for k in range(self.N):
            # Use predicted waypoint for this step
            i = wp_indices[k]
            
            wp = self.waypoints[i]
            wp_next = self.waypoints[(i+1) % self.n]
            
            # Path heading
            yaw_ref = math.atan2(wp_next[1] - wp[1],
                                 wp_next[0] - wp[0])
            
            # Speed using Heron's formula
            if k < self.N - 1:
                wp_prev = self.waypoints[(i-1) % self.n]
                wp_next2 = self.waypoints[(i+2) % self.n]
                # Use Heron's formula for more accurate curvature
                v_ref = self.herons_curvature_speed(wp_prev, wp, wp_next2)
            else:
                v_ref = self.max_speed * 0.8
            
            ref += [wp[0], wp[1], yaw_ref, v_ref]
        
        return ref

    #SOLVE

    def solve(self, state, ref):

        # Validate inputs
        if len(ref) != 4 * self.N:
            print(f"Error: ref has {len(ref)} elements, expected {4 * self.N}")
            return self.last_steer, 0.0
        
        P_val = state + ref
        
        # Initial guess
       
        x0 = []
        for _ in range(self.N + 1):
                x0 += state[0:4]
        for _ in range(self.N):
                x0 += [self.last_steer, 0.0]
       

        lbg = [0.0] * self.n_constraints
        ubg = [0.0] * self.n_constraints

        try:
            sol = self.solver(x0=x0, p=P_val,
                              lbx=self.lbx, ubx=self.ubx,
                              lbg=lbg, ubg=ubg)

            sol_x = sol['x'].full().flatten().tolist()
            

            steer_raw = sol_x[self.n_states]
            accel = sol_x[self.n_states + 1]

            # Warm start
            steer = 0.9 * steer_raw + 0.1 * self.last_steer
            self.last_steer = steer

            return steer, accel
            
        except Exception as e:
            print(f"MPC failed: {e}")


#CONTROLLER

class Controller(Node):

    def __init__(self):
        super().__init__('mpc_controller')

        self.cmd_pub = self.create_publisher(
            AckermannDriveStamped, '/cmd', 10)

        self.create_subscription(
            Odometry, '/ground_truth/odom',
            self.odom_callback, 10)

        self.waypoints = self.load_wp('small_track_waypoints.csv')
        self.idx = 0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0

        self.lap_start = time.time()
        self.lap_count = 0

        self.mpc = MPC(self.waypoints)
        
        # Start SLOW
        self.mpc.max_speed = 30.0

        self.timer = self.create_timer(0.01, self.loop)  
        
        self.get_logger().info(f"MPC Controller initialized with {len(self.waypoints)} waypoints")

    def load_wp(self, filename):
        with open(filename,'r') as f:
            return [(float(r[0]),float(r[1]))
                    for r in csv.reader(f) if len(r)>=2]

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1-2*(q.y*q.y + q.z*q.z))

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v = math.hypot(vx,vy)

    #DOT PRODUCT 
    
    def update_waypoint(self):

        wp_curr = self.waypoints[self.idx]
        wp_next = self.waypoints[(self.idx + 1) % len(self.waypoints)]
        
        # Vector from car to current waypoint
        car_to_wp = (wp_curr[0] - self.x, wp_curr[1] - self.y)
        
        # Vector from current waypoint to next waypoint
        wp_to_next = (wp_next[0] - wp_curr[0], wp_next[1] - wp_curr[1])
        
        # Dot product
        dot = car_to_wp[0] * wp_to_next[0] + car_to_wp[1] * wp_to_next[1]
        
        # If dot product is negative, waypoint is behind us
        if dot < 0:
            prev_idx = self.idx
            self.idx = (self.idx + 1) % len(self.waypoints)
            
            # Lap completion detection
            if self.idx == 0 and prev_idx == len(self.waypoints) - 1:
                self.lap_complete()

    def lap_complete(self):
        lap_time = time.time() - self.lap_start
        self.lap_start = time.time()
        self.lap_count += 1
        
        self.get_logger().info(
                f"LAP {self.lap_count} | {lap_time:.2f}s"
            )
        
    def loop(self):
        try:
            # YOUR FEATURE: Dot product waypoint switching
            self.update_waypoint()

            state = [self.x, self.y, self.yaw, max(self.mpc.min_speed, self.v)]
            
            # YOUR FEATURE: build_ref now uses dynamic waypoint tracking
            ref = self.mpc.build_ref(state, self.idx)

            steer, accel = self.mpc.solve(state, ref)

            target_v = self.v + accel * self.mpc.dt

            msg = AckermannDriveStamped()
            msg.drive.steering_angle = float(steer) * 0.92
            msg.drive.speed = float(target_v)
            
            msg.drive.acceleration = float(accel)
            self.cmd_pub.publish(msg)
        
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            # Emergency stop
            msg = AckermannDriveStamped()
            msg.drive.speed = 0.0
            msg.drive.acceleration = -3.0
            self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()