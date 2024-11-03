
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulation_and_control import (
    pb, MotorCommands, PinWrapper, 
    velocity_to_wheel_angular_velocity
)
import pinocchio as pin
from regulator_model import RegulatorModel
from robot_localization_system import FilterConfiguration, Map, RobotEstimator

map = Map()
landmarks = map.landmarks  # Use landmarks from Map class

# Measurement noise variances
W_range = 0.5 ** 2  # Range measurement noise variance
W_bearing = (np.pi * 0.5 / 180.0) ** 2  # Bearing measurement noise variance

# Calculate range measurements to landmarks
def landmark_range_observations(base_position, W_range):
    y = []
    for lm in landmarks:
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(W_range))
        y.append(range_meas)
    return np.array(y)

# Calculate bearing (angle) measurements to landmarks
def landmark_bearing_observations(base_position, base_bearing):
    y = []
    for lm in landmarks:
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        bearing_true = np.arctan2(dy, dx) - base_bearing
        bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))
        # Constrain angle to [-π, π]
        bearing_meas = np.arctan2(np.sin(bearing_meas), np.cos(bearing_meas))
        y.append(bearing_meas)
    return np.array(y)

# Convert quaternion to yaw (bearing)
def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    rot_quat = quat.toRotationMatrix()
    base_euler = pin.rpy.matrixToRpy(rot_quat)
    return base_euler[2]

# Initialize simulator
def init_simulator(conf_file_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

def main():
    conf_file_name = "robotnik.json"
    sim, dyn_model, num_joints = init_simulator(conf_file_name)

    # Set floor friction
    sim.SetFloorFriction(100)
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []

    # Initialize MPC
    num_states = 3
    num_controls = 2
    C = np.eye(num_states)
    N_mpc = 10

    # Initialize regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)

    # Initial state and control
    init_pos = np.array([2.0, 3.0])
    init_quat = np.array([0, 0, 0.3827, 0.9239])
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2])
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_]
    cur_u_for_linearization = np.zeros(num_controls)
    regulator.updateSystemMatrices(sim, cur_state_x_for_linearization, cur_u_for_linearization)

    # Define cost matrices
    Qcoeff = np.array([310, 310, 80.0])
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff, Rcoeff)
    u_mpc = np.zeros(num_controls)

    # Initialize EKF
    filter_config = FilterConfiguration()
    map = Map()
    estimator = RobotEstimator(filter_config, map)
    estimator.start()
    
    ##### Robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
    cmd = MotorCommands()
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # Main control loop
    while True:
        sim.Step(cmd, "torque")
        # Simulation step
        time_step = sim.GetTimeStep()
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # LINES CHANGED IN THE LAST COMMIT (1 November 2024, 16:45)
        y_range = landmark_range_observations(base_pos_no_noise,W_range)
        y_bearing = landmark_bearing_observations(base_pos, base_bearing_)

        # EKF prediction and update
        estimator.set_control_input(u_mpc)  # Set control input
        estimator.predict_to(time_step)     # Predict next state
        estimator.update_from_landmark_observations(y_range, y_bearing)  # Update range and bearing measurements

        # Get current state estimate
        x_est, Sigma_est = estimator.estimate()

        # Update regulator model matrices
        cur_state_x_for_linearization = [x_est[0], x_est[1], x_est[2]]
        regulator.updateSystemMatrices(sim, x_est, u_mpc)
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H, F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.hstack((base_pos[:2], base_bearing_)).flatten()
        
        # Compute optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x0_mpc
        u_mpc = u_mpc[:num_controls]

        # Prepare control command
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(u_mpc[0], u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd,interface_all_wheels)

        sim.Step(cmd, "torque")
        print(f"u_mpc: {u_mpc}")
        print(f"position X: {base_pos}")

        # Check exit logic
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        # Store data for plotting
        base_pos_all.append(base_pos_no_noise)
        base_bearing_all.append(base_bearing_no_noise_)

        # Update current time
        current_time += time_step

