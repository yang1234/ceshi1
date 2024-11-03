import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from tracker_model import TrackerModel

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    

def getSystemMatricesContinuos(num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    
    # Initialize A matrix
    A = np.zeros((num_states,num_states))
    
    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints) 
    
    # Lower right quadrant of A (velocity affected by damping)
    #if damping_coefficients is not None:
    #    damping_matrix = np.diag(damping_coefficients)
    #    A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
    
    # Initialize B matrix
    B = np.zeros((num_states, num_controls))
    
    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls) 
    
    return A, B

# Example usage:
# sim = YourSimulationObject()
# num_joints = 6  # Example: 6-DOF robot
# damping_coefficients = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # Example damping coefficients
# A, B = getSystemMatrices(sim, num_joints, damping_coefficients)


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # Q = 1 * np.eye(num_states)  # State cost matrix
    p_w = 100000
    v_w = 10
    Q_diag = np.array([p_w, p_w, p_w,p_w, p_w, p_w,p_w, v_w, v_w, v_w,v_w, v_w, v_w,v_w])
    Q = np.diag(Q_diag)
    
    print(Q)

    R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R

class LinearReference:
    def __init__(self, slope, intercept, initial_positions):
        """
        Initialize the linear reference trajectory.
        
        slope: Slope for the linear reference, it can be a list or array for each joint.
        intercept: Intercept for the linear reference, same size as slope.
        initial_positions: The initial position of each joint (starting point).
        """
        self.slope = slope
        self.intercept = intercept
        self.initial_positions = initial_positions

    def get_values(self, current_time):
        """
        Compute the reference position at the given time using a linear function.
        
        current_time: The current simulation time.
        
        Returns:
        q_d: Desired position for each joint (linear reference).
        qd_d: Desired velocity for each joint (constant slope).
        """
        q_d = self.slope * current_time + self.intercept
        qd_d = self.slope  # Velocity is constant, equal to the slope.
        return q_d, qd_d

def main():
    # Configuration
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # Initialize simulation and dynamic model
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # Print joint information
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    regressor_all = np.array([])

    # Define the matrices
    A, B = getSystemMatricesContinuos(num_joints)
    Q, R = getCostMatrices(num_joints)
    
    # Measuring all the state
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep())
    # Compute the matrices needed for MPC optimization
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    
    # Linear reference
    slope = np.array([0.5] * num_joints)  # Example slope for each joint: 5
    intercept = np.array([0.8] * num_joints)  # Example intercept for each joint: 6
    
    ref = LinearReference(slope, intercept, sim.GetInitMotorAngles())  # Initialize linear reference

    # Main control loop
    episode_duration = 5  # duration in seconds
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    
    # Testing loop
    u_mpc = np.zeros(num_joints)
    
    for i in range(steps):
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        
        # Compute linear reference trajectory
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        x_ref = []
        
        # Generate the predictive trajectory for N steps
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j * time_step)
            # Stack q_d and qd_d
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
        
        x_ref = np.vstack(x_ref).flatten()

        # Compute the optimal control sequence
        u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra)
        
        # Apply the optimal control
        u_mpc += u_star[:num_joints]
       
        # Control command
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d, qd_d = ref.get_values(current_time)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        # Advance time
        current_time += time_step
    
    # Plotting the results
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()


    
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()