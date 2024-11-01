#!/usr/bin/env python3

import os

import numpy as np
import time
import matplotlib.pyplot as plt


import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
# 从 standalone_localization_tester 文件导入类
from standalone_localization_tester import SimulatorConfiguration, Controller, Simulator
from robot_localization_system import FilterConfiguration, Map, RobotEstimator
def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    # Reset the simulator to the start conditions
def start(self):
        self._time = 0
        self._x_true = np.random.multivariate_normal(self._filter_config.x0,
                                                     self._filter_config.Sigma0)
        self._u = [0, 0]

def set_control_input(self, u):
        self._u = u

    # Predict the state forwards to the next timestep
def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        v = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=self._filter_config.V * dt)
        self._x_true = self._x_true + np.array([
            v_c * np.cos(self._x_true[2]) * dt,
            v_c * np.sin(self._x_true[2]) * dt,
            omega_c * dt
        ]) + v
        self._x_true[-1] = np.arctan2(np.sin(self._x_true[-1]),
                                      np.cos(self._x_true[-1]))
        self._time += dt
        return self._time

def landmark_range_observations(self):
        y = []
        C = []
        W = self._filter_config.W_range
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W))
            y.append(range_meas)

        y = np.array(y)
        return y
    
def landmark_bearing_observations(self):
        y = []
        C = []
        W = self._filter_config.W_bearing
        for lm in self._map.landmarks:
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W))
            #wrap the angle
            bearing_meas = np.arctan2(np.sin(bearing_meas), np.cos(bearing_meas))
            y.append(bearing_meas)

        y = np.array(y)
        return y

def x_true(self):
        return self._x_true

# 定义四元数转偏航角的函数
def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()
    rot_quat = quat.toRotationMatrix()
    base_euler = pin.rpy.matrixToRpy(rot_quat)
    return base_euler[2]

def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

# 主程序逻辑
def main():
    conf_file_name = "robotnik.json"
    sim,dyn_model,num_joints=init_simulator(conf_file_name)
    sim.SetFloorFriction(100)
    time_step = sim.GetTimeStep()
    
    # 初始化模拟配置、地图和 EKF 估计器
    sim_config = SimulatorConfiguration()
    filter_config = FilterConfiguration()
    map = Map()
    estimator = RobotEstimator(filter_config, map)
    estimator.start()
    
    # 初始化控制器
    controller = Controller(sim_config)
    u = controller.next_control_input(*estimator.estimate())

    base_pos_all = []
    base_bearing_all = []

    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)
    
    # 进入主循环
    for _ in range(sim_config.time_steps):
        # 模拟时间推进
        sim.Step(cmd, "torque")

        # 获取无噪声的真实状态
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise = quaternion2bearing(*base_ori_no_noise)
        
        # 获取带噪声的当前状态
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing = quaternion2bearing(*base_ori)
        
        # 获取观测
        y_range =  landmark_range_observations(base_pos)
        y_bearing = landmark_bearing_observations(base_ori)

        # EKF 预测和更新
        estimator.set_control_input(u)
        estimator.predict_to(sim_config.dt)
        estimator.update_from_landmark_observations(y_range, y_bearing)

        # 获取估计结果
        x_est, Sigma_est = estimator.estimate()
        
        # 更新控制器输入
        u = controller.next_control_input(x_est, Sigma_est)

        # 控制机器人轮子
        left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(
            u[0], u[1], 0.46, 0.11
        )
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        cmd = MotorCommands()
        cmd.SetControlCmd(angular_wheels_velocity_cmd, ["velocity", "velocity", "velocity", "velocity"])
        
        # 存储位置数据
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing)

    # 绘图显示结果
    plt.figure()
    plt.plot([pos[0] for pos in base_pos_all], [pos[1] for pos in base_pos_all], label="Estimated Path")
    plt.scatter([lm[0] for lm in map.landmarks], [lm[1] for lm in map.landmarks], marker='x', color='red', label='Landmarks')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("EKF Localization with Estimated Path")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
