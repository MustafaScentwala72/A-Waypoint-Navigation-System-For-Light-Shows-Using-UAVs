# Original author: Dr. Alexandros Giagkos. 
# Based on Bitcraze's PID controller, 
# refined in collaboration with Mr. Mohammad Hasnain Parray and myself, 
# enhancing both the PID controller and Kalman filter.

import os
import numpy as np
from cf_pid_control import cf_waypoint_pid as pid

class kalman_filter():
    def __init__(self):
        self.noise_std_GPS = 0.2
        self.noise_std_ACCEL = 0.02

        #Tuning parameter
        self.q_tr = 0.8 # Original: 1.0

        #Initializing KF state and model uncertainty
        self.initialize_KF(self.noise_std_GPS, self.noise_std_ACCEL)

        # Flags for use cases to test
        self.use_accel_only = False
        self.use_ground_truth_measurement = False
        self.use_noisy_measurement = False

        #Variables for Plotting
        self.raw_data_vec = []
        self.noisy_data_vec = []
        self.KF_estimate_vec = []
        self.time = []

        # Variable for noise generation
        self.x_noisy_global_last = 0.0
        self.y_noisy_global_last = 0.0
        self.z_noisy_global_last = 0.0
        self.ax_noisy_global_last = 0.0
        self.ay_noisy_global_last = 0.0
        self.az_noisy_global_last = 0.0
        self.v_x_noisy = 0.0
        self.v_y_noisy = 0.0
        self.v_z_noisy = 0.0

        # Good Kalman Filter PID Gains
        # KF gains
        # gains = {"P_vel_z": 6.0,     "I_vel_z": 1.0,     "D_vel_z": 0.8,
        #             "P_pos_z": 2.5,     "I_pos_z": 0.0,     "D_pos_z": 1.0,
        #             "P_rate_rp": 0.2,     "I_rate_rp":0.0,      "D_rate_rp": 0.03,
        #             "P_rate_y": 0.01,      "I_rate_y": 0.0,      "D_rate_y": 0.001,
        #             "P_att_rp": 16.0,     "I_att_rp":0.0,      "D_att_rp": 0.3,
        #             "P_att_y": 5.0,      "I_att_y": 0.0,      "D_att_y": 0.1,
        #             "P_vel_xy": 2.0,     "I_vel_xy": 0.0,     "D_vel_xy": 0.10,
        #             "P_pos_xy": 1.5,     "I_pos_xy": 0.0,     "D_pos_xy": 0.02}
    
    def initialize_KF(self, noise_std_GPS, noise_std_ACCEL):
        # Function to initialize the following:
        #   Optimal state vector (self.X_opt)
        #   Optimal prediction covariance (self.P_opt)
        #   Measurement Matrices (self.H_GPS and self.H_ACCEL)
        #   Measurement Covariance Matrices (self.R_GPS and self.R_ACCEL)

        # IMPORTANT: Assuming the state definition in the order: X = [x, v_x, a_x, y, v_y, a_y, z, v_z, a_z], Shape: (9,1), n_states = 9

        # Initializing the state vector (self.X_opt) and the covariance matrix (self.P_opt) of the state estimate
        self.X_opt = np.random.rand(9,1)
        self.P_opt = 1000.0*np.diag(np.ones(9))

        # Defining the Measurement Matrices (H) for both GPS and ACCELEROMETER measurements - Shape: (n_measurements x n_states)
        self.H_GPS = np.array([[1,0,0,0,0,0,0,0,0],
                               [0,0,0,1,0,0,0,0,0],
                               [0,0,0,0,0,0,1,0,0]
                              ])
        self.H_ACCEL = np.array([[0,0,1,0,0,0,0,0,0],
                                 [0,0,0,0,0,1,0,0,0],
                                 [0,0,0,0,0,0,0,0,1]])

        # Defining the Measurement Covariance Matrices (R) for both GPS and ACCELEROMETER measurements - Shape: (n_measurements x n_measurements)
        self.R_GPS = (noise_std_GPS**2)*np.eye(3)
        self.R_ACCEL = (noise_std_ACCEL**2)*np.eye(3)

    def reset(self):
        self.initialize_KF(self.noise_std_GPS, self.noise_std_ACCEL)
        #Variables for Plotting
        self.raw_data_vec = []
        self.noisy_data_vec = []
        self.KF_estimate_vec = []
        self.time = []

        # Variable for noise generation
        self.x_noisy_global_last = 0.0
        self.y_noisy_global_last = 0.0
        self.z_noisy_global_last = 0.0
        self.ax_noisy_global_last = 0.0
        self.ay_noisy_global_last = 0.0
        self.az_noisy_global_last = 0.0
        self.v_x_noisy = 0.0
        self.v_y_noisy = 0.0
        self.v_z_noisy = 0.0
            
    def KF_state_propagation(self, dt):
        # Function that propagates the last fused state over a time-interval dt
        # Inputs:
        #   dt: Time-interval for propagation
        # Outputs:
        #   X_pred: Predicted state after a progpagation time dt (n_states x 1)
        #   P_pred: Predicted covariance after a propagation time dt (n_states x n_states)

        # Call Process Covariance matrix (Q): DO NOT MODIFY
        Q_trans = self.calculate_Q(dt, self.q_tr)

        # Defining the state transition matrix A_trans (n_states x n_states)
        A_trans_sub = np.array([[1, dt, np.power(dt,2)/2],
                                [0, 1, dt],
                                [0, 0, 1]
                               ])
        A_trans = np.block([[A_trans_sub, np.zeros((3,6))],
                            [np.zeros((3,3)), A_trans_sub, np.zeros((3,3,))],
                            [np.zeros((3,6)), A_trans_sub]])
        
        # Calculating the propagated state (X_pred) and the propagated covariance (P_pred) using the last fused state (self.X_opt) and covariance (self.P_opt)
        X_pred = A_trans @ self.X_opt
        P_pred = A_trans @ self.P_opt @ A_trans.transpose() + Q_trans

        return X_pred, P_pred

    def KF_sensor_fusion(self, X_pred, P_pred, H, R, Z):
        # Function that performs sensor fusion when a measurement is received
        # Inputs:
        #   X_pred: State propagated to time of fusion (n_states x 1)
        #   P_pred: Covariance matrix propagated to time of fusion (n_states x n_states)
        #   H: Measurement Matrix of measured sensor (n_measurements x n_states)
        #   R: Measurement Covariance of measured sensor (n_measurements x n_states)
        #   Z: Measurement vector received from the sensor (n_measurements x 1)
        # Returns:
        #   self.X_opt: Fused state estimate at sensor readout time (n_states x 1)
        #   self.P_opt: Fused covariance matrix at sensor readout time (n_states x n_states)

        K = P_pred @ H.transpose() @ (np.linalg.inv(((H @ P_pred @ H.transpose()) + R)))
        self.X_opt = X_pred + K @ (Z - (H @ X_pred))
        self.P_opt = ((np.eye(9)) - K @ H) @ P_pred

        return self.X_opt, self.P_opt

    def KF_estimate(self, measured_state_gps, measured_state_accel, dt_last_measurement, sensor_state_flag):
        # Function that outputs the state estimate wehn requested
        # Inputs:
        #   dt_last_measurement: Time elapsed since last acceleration measurements received from either acclerometer or GPS (always > 0)
        #   sensor_state_flag: Possible values are [0,1,2,3]
        #       -> 0: No sensor measurement received at current requested time
        #       -> 1: GPS measurement received at current requested time
        #       -> 2: Accelerometer measurement received at current requested time
        #       -> 3: Accelerometer and GPS measurements received simultaneously
        #   measured_state_gps: The latest GPS position measurement (X,Y,Z) in inertial world frame (n_measurements x 1)
        #   measured_state_accel: The latest ACCELEROMETER measurement (A_X, A_Y, A_Z) in  world frame (n_measurements x 1)
        # Returns:
        #   X_est: Estimated drone state (n_states x 1)
        #   P_est: Estimated covariance (n_states x n_states)

        # Propagate 
        X_prop, P_prop = self.KF_state_propagation(dt_last_measurement)
        
        # Calculating estimate depending on sensor state
        if sensor_state_flag == 0:
            X_est, P_est = X_prop, P_prop
        if sensor_state_flag == 1:
            H = self.H_GPS
            R = self.R_GPS
            Z = measured_state_gps
            X_est, P_est = self.KF_sensor_fusion(X_prop,P_prop,H,R,Z)
            # print("In GPS meas step")
        if sensor_state_flag == 2:
            H = self.H_ACCEL
            R = self.R_ACCEL
            Z = measured_state_accel
            X_est, P_est = self.KF_sensor_fusion(X_prop,P_prop,H,R,Z)
            # print("In ACCEL meas step")
        if sensor_state_flag == 3:
            X_opt_gps, P_opt_gps = self.KF_sensor_fusion(X_prop, P_prop, self.H_GPS, self.R_GPS, measured_state_gps)
            X_est, P_est = self.KF_sensor_fusion(X_opt_gps, P_opt_gps, self.H_ACCEL, self.R_ACCEL, measured_state_accel)

        return X_est, P_est
 
    def calculate_Q(self, dt_tr, q_tr):

        # Calculating Q submatrix
        Q_sub = q_tr * np.array([[np.power(dt_tr,5)/20, np.power(dt_tr,4)/8, np.power(dt_tr,3)/6],
                                   [np.power(dt_tr,4)/8, np.power(dt_tr,3)/3, np.power(dt_tr,2)/2],
                                   [np.power(dt_tr,3)/6, np.power(dt_tr,2)/2, dt_tr]
                                ])
        
        Q = np.block([[Q_sub, np.zeros((3,6))],
                      [np.zeros((3,3)), Q_sub, np.zeros((3,3,))],
                      [np.zeros((3,6)), Q_sub]])
        
        return Q
    
    def add_noise(self, raw_sensor_data, dt_gps, dt_accel, gps_period, accel_period, accel_bias = 0.0, gps_bias = 0.0):
        noisy_sensor_data = raw_sensor_data.copy()

        #Adding Gaussian noise with specified noise parameters to sensor data
        noisy_sensor_data['x_global'] += np.random.normal(loc = gps_bias, scale = self.noise_std_GPS)
        noisy_sensor_data['y_global'] += np.random.normal(loc = gps_bias, scale = self.noise_std_GPS)
        noisy_sensor_data['z_global'] += np.random.normal(loc = gps_bias, scale = self.noise_std_GPS)

        noisy_sensor_data['ax_global'] += np.random.normal(loc = accel_bias, scale = self.noise_std_ACCEL)
        noisy_sensor_data['ay_global'] += np.random.normal(loc = accel_bias, scale = self.noise_std_ACCEL)
        noisy_sensor_data['az_global'] += np.random.normal(loc = accel_bias, scale = self.noise_std_ACCEL) 

        if self.use_accel_only:
            if np.round(dt_accel,3) >= accel_period/1000:
                self.v_x_noisy += ((noisy_sensor_data['ax_global'] + self.ax_noisy_global_last)/2)*dt_accel
                self.v_y_noisy += ((noisy_sensor_data['ay_global'] + self.ay_noisy_global_last)/2)*dt_accel
                self.v_z_noisy += ((noisy_sensor_data['az_global'] + self.az_noisy_global_last)/2)*dt_accel
                self.ax_noisy_global_last = noisy_sensor_data['ax_global']
                self.ay_noisy_global_last = noisy_sensor_data['ay_global']
                self.az_noisy_global_last = noisy_sensor_data['az_global']
        else:
            if np.round(dt_gps,3) >= gps_period/1000:
                self.v_x_noisy = (noisy_sensor_data['x_global'] - self.x_noisy_global_last)/dt_gps
                self.v_y_noisy = (noisy_sensor_data['y_global'] - self.y_noisy_global_last)/dt_gps
                self.v_z_noisy = (noisy_sensor_data['z_global'] - self.z_noisy_global_last)/dt_gps
                self.x_noisy_global_last = noisy_sensor_data['x_global']
                self.y_noisy_global_last = noisy_sensor_data['y_global']
                self.z_noisy_global_last = noisy_sensor_data['z_global']

        noisy_sensor_data['v_x'] = self.v_x_noisy
        noisy_sensor_data['v_y'] = self.v_y_noisy
        noisy_sensor_data['v_z'] = self.v_z_noisy
        noisy_sensor_data['v_forward'] =  self.v_x_noisy * np.cos(noisy_sensor_data['yaw']) + self.v_y_noisy * np.sin(noisy_sensor_data['yaw'])
        noisy_sensor_data['v_sideways'] = -self.v_x_noisy * np.sin(noisy_sensor_data['yaw']) + self.v_y_noisy * np.cos(noisy_sensor_data['yaw'])
        noisy_sensor_data['v_down'] = self.v_z_noisy

        return noisy_sensor_data
    
    def aggregate_states(self, raw_data, noisy_data, KF_data, time):
        keys = ['x_global', 'y_global', 'z_global', 'v_forward', 'v_sideways', 'v_down', 'ax_global', 'ay_global' , 'az_global']
        self.raw_data_vec.append(list(raw_data[key] for key in keys))
        self.noisy_data_vec.append(list(noisy_data[key] for key in keys))
        self.KF_estimate_vec.append(list(KF_data[key] for key in keys))
        self.time.append(time)