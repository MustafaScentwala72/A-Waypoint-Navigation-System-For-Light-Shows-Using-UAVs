import sys
import os
import numpy as np
import math
import calendar
import time
import sys
from scipy.spatial.transform import Rotation as R

from controller import Robot, Supervisor
from controller import Keyboard

sys.path.append(r'C:\Users\musta\OneDrive - Aston University\Desktop\230295100_Simulation_Files\230295100_Simulation_Files\230295100_Simulation_Files\230295100_controllers\cf_controller\controllers\Drone1')

from kalman_filter import kalman_filter as KF
from cf_pid_control import cf_waypoint_pid, cf_velocity_pid

class Crazyflie(Supervisor):
    def __init__(self, cfg):
        super().__init__()
       
        self.robot = self.getSelf()
        self.target = self.getFromDef("Target")         
        self.timestep = int(self.getBasicTimeStep())
        
        ## init motors
        self.motor1 = self.getDevice('m1_motor')
        self.motor1.setPosition(float('inf'))
        self.motor1.setVelocity(-1)
        self.motor2 = self.getDevice('m2_motor')
        self.motor2.setPosition(float('inf'))
        self.motor2.setVelocity(-1)
        self.motor3 = self.getDevice('m3_motor')
        self.motor3.setPosition(float('inf'))
        self.motor3.setVelocity(-1)
        self.motor4 = self.getDevice('m4_motor')
        self.motor4.setPosition(float('inf'))
        self.motor4.setVelocity(-1)

        self.pose = None

        self.useKF = True
        self.KF = KF()
        self.sensor_flag = 0
        self.dt_accel = 0.0
        self.dt_gps = 0.0
        self.dt_propagate = 0.0
        self.meas_state_gps = np.zeros((2,1))
        self.meas_state_accel = np.zeros((3,1))

        self.accel_read_last_time = 0.0
        self.gps_read_last_time = 0.0 
        self.g = 9.81 
        self.ctrl_update_period = int(self.timestep*2)
        self.gps_update_period = int(self.timestep*2) 
        self.accel_update_period = int(self.timestep*1) 

        ## init sensors
        self.imu = self.getDevice('inertial unit')
        self.imu.enable(self.timestep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.gps_update_period)
        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(self.accel_update_period)
        self.gyro = self.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.camera = self.getDevice('cf_camera')
        self.camera.enable(self.timestep)
        self.range_front = self.getDevice('range_front')
        self.range_front.enable(self.timestep)
        self.range_left = self.getDevice("range_left")
        self.range_left.enable(self.timestep)
        self.range_back = self.getDevice("range_back")
        self.range_back.enable(self.timestep)
        self.range_right = self.getDevice("range_right")
        self.range_right.enable(self.timestep)
        self.laser_down = self.getDevice("laser_down")
        self.laser_down.enable(self.timestep)

        self.WP_PID = cf_waypoint_pid()
        self.V_PID = cf_velocity_pid()
        self.PID_update_last_time = self.getTime()
        self.sensor_read_last_time = self.getTime()
        self.dt_ctrl = 0.0

        self.x_global_last = 0.0
        self.y_global_last = 0.0
        self.z_global_last = 0.0
        self.vx_global = 0.0
        self.vy_global = 0.0
        self.vz_global = 0.0
                       
        self.max_pitch = 1.0
        self.max_roll = 1.0
        self.max_altitude = 2.0
        self.min_altitude = 0.5
        self.max_distance = 4.0

        # Defining the square and triangle waypoints for Drone 4
        self.square_positions = [
            [0.0, 1.0, 1.6],  # Top left (Initial)
            [0.0, 0.0, 1.6],  # Bottom left
            [1.0, 0.0, 1.6],  # Bottom right
            [1.0, 1.0, 1.6]   # Top right
        ]
        
        self.triangle_positions = [
            [0.5, 1.0, 1.6],  # Top middle
            [0.0, 1.0, 1.6],  # Top left
            [1.0, 0.0, 1.6],  # Bottom right
            [0.5, 0.5, 1.6]   # Center
        ]

        # Assigning a fixed height to avoid collisions
        self.flight_height = 1.6  # Drone 4's fixed flight height

        self.current_shape = "square"  # Starting with the square
        self.position_index = 0        # Starting at the first position
        self.waypoints = self.square_positions  # Starting with square waypoints
        self.hover_time = 10.0  # 10 seconds hover time
        self.last_time = time.time()  # Tracking the last time the drone reached a waypoint
        
        self.waypointpos = np.zeros(3)
                       
        self.step_physics()

    def setWaypointPos(self, waypoint=None):
        current_time = time.time()

        # Checking if the drone has hovered at the current waypoint for the required time
        if current_time - self.last_time >= self.hover_time:
            if waypoint is None:
                # Using predefined waypoints
                if self.position_index < len(self.waypoints):
                    self.waypointpos = np.array(self.waypoints[self.position_index])
                    self.position_index += 1
                else:
                    # Switching between shapes and shift positions in a loop
                    if self.current_shape == "square":
                        self.current_shape = "triangle"
                        self.waypoints = self.triangle_positions
                    elif self.current_shape == "triangle":
                        self.current_shape = "square"
                        self.waypoints = self.square_positions

                        # Shifting to the next corner for the square
                        self.waypoints = self.waypoints[-1:] + self.waypoints[:-1]  # Rotate positions

                    self.position_index = 0
                    self.waypointpos = np.array(self.waypoints[self.position_index])
                    self.position_index += 1

                # Adjusting the height to avoid collisions
                self.waypointpos[2] = self.flight_height

                # Updating the target translation field
                translation = np.copy(self.waypointpos)
                target_tr = self.target.getField('translation')
                target_tr.setSFVec3f(translation.tolist())

                # Reseting the timer after setting the new waypoint
                self.last_time = time.time()
                
                # Printing the waypoint
                print(f"Drone 4 has reached waypoint: {self.waypointpos}")
                
            else:
                self.waypointpos[0] = waypoint[0]
                self.waypointpos[1] = waypoint[1]
                self.waypointpos[2] = self.flight_height  # Ensuring fixed height

                # Printing the custom waypoint
                print(f"Drone 4 set to custom waypoint: {self.waypointpos}")

        return self.waypointpos.copy()
                
    def canFly(self):
        if self.imu.getRollPitchYaw()[0] > self.max_roll or self.imu.getRollPitchYaw()[0] < -self.max_roll or self.imu.getRollPitchYaw()[1] > self.max_pitch or self.imu.getRollPitchYaw()[1] < -self.max_pitch:
            print("crashed")
            return False
        elif self.isLost():
            print("lost")
            return False
        return True

    def isLost(self):
        return True if self.dist2Target() > self.max_distance else False
             
    def dist2Target(self, waypoint=None):
        if waypoint is not None:
            waypointpos = np.copy(waypoint)
        else:
            waypointpos = np.copy(self.waypointpos)
                        
        globalpos = np.array([self.gps.getValues()[0], self.gps.getValues()[1], self.gps.getValues()[2]])
        distance = np.linalg.norm(waypointpos - globalpos)    
        return distance       
     
    def step_physics(self):
        for _ in range(3):
            if super(Supervisor, self).step(self.timestep) == -1:
                exit()

    def update_pose(self, useKF=True):
        KF_pose, raw_pose = self.read_KF_estimates() # updating KF filter even if you dont use it for stabilization
        if useKF:
            self.pose = KF_pose
        else:
            self.pose = raw_pose
        return self.pose
                
    def read_KF_estimates(self):
        
        # Updating time intervals for sensing and propagation
        self.dt_accel = self.getTime() - self.accel_read_last_time
        self.dt_gps = self.getTime() - self.gps_read_last_time

        # Data dictionary - read sensors and add noise
        measured_data_raw = self.read_sensors().copy()
        measured_noisy_data = self.KF.add_noise(measured_data_raw.copy(), self.dt_gps, self.dt_accel, self.gps.getSamplingPeriod(), self.accelerometer.getSamplingPeriod())

        self.sensor_flag = 0

        self.dt_propagate = min(self.dt_accel, self.dt_gps)
    
        if np.round(self.dt_accel, 3) >= self.accelerometer.getSamplingPeriod()/1000 and np.round(self.dt_gps, 3) >= self.gps.getSamplingPeriod()/1000:
            self.sensor_flag = 3
            self.meas_state_accel = np.array([[measured_noisy_data['ax_global'], measured_noisy_data['ay_global'], measured_noisy_data['az_global']]]).transpose()
            self.accel_read_last_time = self.getTime()
            self.meas_state_gps = np.array([[measured_noisy_data['x_global'], measured_noisy_data['y_global'], measured_noisy_data['z_global']]]).transpose()
            self.gps_read_last_time = self.getTime()
        else:
            if np.round(self.dt_gps,3) >= self.gps.getSamplingPeriod()/1000:
                self.sensor_flag = 1
                self.meas_state_gps = np.array([[measured_noisy_data['x_global'], measured_noisy_data['y_global'], measured_noisy_data['z_global']]]).transpose()
                self.gps_read_last_time = self.getTime()
            if np.round(self.dt_accel,3) >= self.accelerometer.getSamplingPeriod()/1000: 
                self.sensor_flag = 2
                self.meas_state_accel = np.array([[measured_noisy_data['ax_global'], measured_noisy_data['ay_global'], measured_noisy_data['az_global']]]).transpose()
                self.accel_read_last_time = self.getTime()
    
        estimated_state, estimated_covariance = self.KF.KF_estimate(self.meas_state_gps, self.meas_state_accel, self.dt_propagate, self.sensor_flag)

        x_g_est, v_x_g_est, a_x_g_est, y_g_est, v_y_g_est, a_y_g_est, z_g_est, v_z_g_est, a_z_g_est = estimated_state.flatten()
        KF_state_outputs = measured_noisy_data.copy()
        KF_state_outputs['x_global'] = x_g_est
        KF_state_outputs['y_global'] = y_g_est
        KF_state_outputs['z_global'] = z_g_est
        KF_state_outputs['v_x'] = v_x_g_est
        KF_state_outputs['v_y'] = v_y_g_est
        KF_state_outputs['v_z'] = v_z_g_est
        KF_state_outputs['v_forward'] = v_x_g_est * np.cos(KF_state_outputs['yaw']) + v_y_g_est * np.sin(KF_state_outputs['yaw'])
        KF_state_outputs['v_sideways'] = -v_x_g_est * np.sin(KF_state_outputs['yaw']) + v_y_g_est * np.cos(KF_state_outputs['yaw'])
        KF_state_outputs['v_down'] = v_z_g_est
        KF_state_outputs['ax_global'] = a_x_g_est
        KF_state_outputs['ay_global'] = a_y_g_est
        KF_state_outputs['az_global'] = a_z_g_est

        self.KF.aggregate_states(measured_data_raw, measured_noisy_data, KF_state_outputs, self.getTime())
     
        output_measurement = KF_state_outputs.copy()

        return output_measurement, measured_data_raw

    def read_sensors(self):

        # Sensor data call values
        # "x_global": Global X position
        # "y_global": Global Y position
        # "z_global": Global Z position
        # "roll": Roll angle (rad)
        # "pitch": Pitch angle (rad)
        # "yaw": Yaw angle (rad)
        # "q_x": Quaternion x value
        # "q_y": Quaternion y value
        # "q_z": Quaternion z value
        # "q_w": Quaternion w value
        # "v_x": Global X velocity
        # "v_y": Global Y velocity
        # "v_z": Global Z velocity
        # "v_forward": Forward velocity (body frame)
        # "v_sideways": Leftward velocity (body frame)
        # "v_down": Downward velocity (body frame)
        # "ax_global": Global X acceleration
        # "ay_global": Global Y acceleration
        # "az_global": Global Z acceleration
        # "range_front": Front range finder distance
        # "range_down": Downward range finder distance
        # "range_left": Leftward range finder distance 
        # "range_back": Backward range finder distance
        # "range_right": Rightward range finder distance
        # "rate_roll": Roll rate (rad/s)
        # "rate_pitch": Pitch rate (rad/s)
        # "rate_yaw": Yaw rate (rad/s)

        # Data dictionary
        data = {}

        # Time interval
        dt = self.getTime() - self.sensor_read_last_time
        data['t'] = self.getTime()
        self.sensor_read_last_time = self.getTime()

        # Position
        data['x_global'] = self.gps.getValues()[0]
        data['y_global'] = self.gps.getValues()[1]
        data['z_global'] = self.gps.getValues()[2]

        # Attitude
        data['roll'] = self.imu.getRollPitchYaw()[0]
        data['pitch'] = self.imu.getRollPitchYaw()[1]
        data['yaw'] = self.imu.getRollPitchYaw()[2]

        data['q_x'] = self.imu.getQuaternion()[0]
        data['q_y'] = self.imu.getQuaternion()[1]
        data['q_z'] = self.imu.getQuaternion()[2]
        data['q_w'] = self.imu.getQuaternion()[3]

        ax_body = self.accelerometer.getValues()[0]
        ay_body = self.accelerometer.getValues()[1]
        az_body = self.accelerometer.getValues()[2] 

        # Velocity
        if np.round(self.dt_gps, 3) >= self.gps_update_period/1000:
            self.vx_global = (data['x_global'] - self.x_global_last) / self.dt_gps
            self.vy_global = (data['y_global'] - self.y_global_last) / self.dt_gps
            self.vz_global = (data['z_global'] - self.z_global_last) / self.dt_gps
            self.x_global_last = data['x_global']
            self.y_global_last = data['y_global']
            self.z_global_last = data['z_global']
        else:
            data['x_global'] = self.x_global_last
            data['y_global'] = self.y_global_last
            data['z_global'] = self.z_global_last

        data['v_x'] = self.vx_global
        data['v_y'] = self.vy_global
        data['v_z'] = self.vz_global

        data['v_forward'] =  self.vx_global * np.cos(data['yaw']) + self.vy_global * np.sin(data['yaw'])
        data['v_sideways'] =  -self.vx_global * np.sin(data['yaw']) + self.vy_global * np.cos(data['yaw'])
        data['v_down'] =  self.vz_global

        # Accleration from body to global frame
        r = R.from_euler('xyz', [data['roll'], data['pitch'], data['yaw']])
        R_T = r.as_matrix()

        a_global = (R_T @ np.array([[ax_body, ay_body, az_body]]).transpose()).flatten()

        data['ax_global'] = a_global[0]
        data['ay_global'] = a_global[1]
        data['az_global'] = a_global[2] - self.g     

        # Range sensor
        data['range_front'] = self.range_front.getValue() / 1000.0
        data['range_left']  = self.range_left.getValue() / 1000.0
        data['range_back']  = self.range_back.getValue() / 1000.0
        data['range_right'] = self.range_right.getValue() / 1000.0
        data['range_down'] = self.laser_down.getValue() / 1000.0

        # Yaw rate
        data['rate_roll'] = self.gyro.getValues()[0]
        data['rate_pitch'] = self.gyro.getValues()[1]
        data['rate_yaw'] = self.gyro.getValues()[2]

        return data

    def step_wp(self, sensor_data, waypoint):
        
        self.dt_ctrl = self.getTime() - self.PID_update_last_time

        if np.round(self.dt_ctrl, 3) >= self.ctrl_update_period/1000:
            
            self.PID_update_last_time = self.getTime()

            motorPower = self.WP_PID.pid(self.dt_ctrl, waypoint, sensor_data)
        
            self.motor1.setPosition(float('inf'))
            self.motor2.setPosition(float('inf'))
            self.motor3.setPosition(float('inf'))
            self.motor4.setPosition(float('inf'))
            self.motor1.setVelocity(-motorPower[0])
            self.motor2.setVelocity(motorPower[1])
            self.motor3.setVelocity(-motorPower[2])
            self.motor4.setVelocity(motorPower[3])

        self.step_physics()

    def step_vel(self, state_data, action):  

        self.dt_ctrl = self.getTime() - self.PID_update_last_time
        
        if np.round(self.dt_ctrl, 3) >= self.ctrl_update_period/1000:            
            self.PID_update_last_time = self.getTime()
        
            motor_power = self.V_PID.pid(self.dt_ctrl, action, state_data)
            
            self.motor1.setPosition(float('inf'))
            self.motor2.setPosition(float('inf'))
            self.motor3.setPosition(float('inf'))
            self.motor4.setPosition(float('inf'))
            self.motor1.setVelocity(-motor_power[0])
            self.motor2.setVelocity(motor_power[1])
            self.motor3.setVelocity(-motor_power[2])
            self.motor4.setVelocity(motor_power[3])
                
        self.step_physics()
        
seed = calendar.timegm(time.gmtime())
      
cfg = { 
    "seed": seed,
}
    
if __name__ == '__main__':

    cf = Crazyflie(cfg)

    while True:
        state_data = cf.update_pose(useKF=False)
        # action: desired_vx, desired_vy, desired_yaw_rate, desired_alt
        action = np.array([0.0, 0.0, 0.0, 0.3])
        cf.step_vel(state_data, action)
        if state_data["z_global"] > 0.3 - 1e-3:
            break
    
    cf.setWaypointPos(waypoint=[1.5932573, -1.3207, 2.50949])    
    while True:
        state_data = cf.update_pose()
        waypoint = np.append(cf.waypointpos, 0.0) # adding heading
        cf.step_wp(state_data, waypoint)

        if not cf.canFly():
            break
        if cf.dist2Target() <= 0.08:
            cf.setWaypointPos()     

                
    sys.exit()