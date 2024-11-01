import numpy as np
    
class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.N = N
        self.q = q #  output dimension  3
        self.m = m #  input dimension   4
        self.n = n #  state dimension   3 xytheta




    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

        return H, F

    def newABC(self):
        # Compute matrix A
        A_upper_left = np.eye(self.n) + self.delta * self.A
        A_upper_right = self.delta * self.B
        A_lower_left = np.zeros((self.m, self.n))
        A_lower_right = np.eye(self.m)

         # Combine blocks to form A
        A_upper = np.hstack((A_upper_left, A_upper_right))
        A_lower = np.hstack((A_lower_left, A_lower_right))
        A = np.vstack((A_upper, A_lower))


    def propagation_model_regulator_fixed_std(self):
        S_bar = np.zeros((self.N*self.q, self.N*self.m))
        T_bar = np.zeros((self.N*self.q, self.n))
        Q_bar = np.zeros((self.N*self.q, self.N*self.q))
        R_bar = np.zeros((self.N*self.m, self.N*self.m))
        # print(f"S_bar is :{S_bar.shape}")
        # print(f"self.q is :{self.q}")
        # print(f"self.m is :{self.m}")
        # print(f"self.C is :{self.C.shape}")  # Should be (12, 12)
        # print(f"self.A is :{self.A.shape}")  # Check its shape
        # print(f"states self.n is :{self.n}")
        # print(f"self.B is :{self.B.shape}")  # Check its shape
        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[(k-1)*self.q:k*self.q, (k-j)*self.m:(k-j+1)*self.m] = np.dot(np.dot(self.C, np.linalg.matrix_power(self.A, j-1)), self.B)

            T_bar[(k-1)*self.q:k*self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))

            Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.Q
            R_bar[(k-1)*self.m:k*self.m, (k-1)*self.m:k*self.m] = self.R

        return S_bar, T_bar, Q_bar, R_bar
    
    def updateSystemMatrices(self,sim,cur_x,cur_u):
        """
        Get the system matrices A and B according to the dimensions of the state and control input.
        
        Parameters:
        num_states, number of system states
        num_controls, number oc conttrol inputs
        cur_x, current state around which to linearize
        cur_u, current control input around which to linearize
       
        
        Returns:
        A: State transition matrix
        B: Control input matrix
        """
        # Check if state_x_for_linearization and cur_u_for_linearization are provided
        if cur_x is None or cur_u is None:
            raise ValueError(
                "state_x_for_linearization and cur_u_for_linearization are not specified.\n"
                "Please provide the current state and control input for linearization.\n"
                "Hint: Use the goal state (e.g., zeros) and zero control input at the beginning.\n"
                "Also, ensure that you implement the linearization logic in the updateSystemMatrices function."
            )
        num_controls = self.m # 4个关节
        num_outputs = self.q # 3个输出xytheta

        num_states = self.n
        # self.n=num_states

        delta_t = sim.GetTimeStep()
        v0 = cur_x[0]
        theta0 = cur_x[2]
        
        # get A and B matrices by linearinzing the cotinuous system dynamics
        # The linearized continuous-time system is:
        # 提取当前状态和控制输入

        # 动态计算 A 矩阵
        # A = np.block(
        #     [[np.eye(num_controls),np.zeros((num_controls,num_controls)),-s * np.sin(theta) * delta_t * np.eye(num_controls)], 
        #      [np.zeros((num_controls,num_controls)), np.eye(num_controls),s * np.cos(theta) * delta_t * np.eye(num_controls)],
        #      [np.zeros((num_controls,num_controls)), np.zeros((num_controls,num_controls)), np.eye(num_controls)]]
        #     )
        A = np.array([
            [1, 0, -v0 * np.sin(theta0) * delta_t],
            [0, 1,  v0 * np.cos(theta0) * delta_t],
            [0, 0,  1]
        ])
        
        # B = np.block(
        #     [[np.cos(theta) * delta_t* np.eye(num_controls),np.zeros((num_controls,num_controls))], 
        #      [np.sin(theta) * delta_t* np.eye(num_controls),np.zeros((num_controls,num_controls))],
        #      [np.zeros((num_controls,num_controls)),delta_t*np.eye(num_controls)]]
        #     )
        # 动态计算 B 矩阵
        B = np.array([
            [np.cos(theta0) * delta_t, 0],
            [np.sin(theta0) * delta_t, 0],
            [0, delta_t]
        ])
        #计算C矩阵
        C = np.eye(num_states)

        # print(f"C is :{C.shape}")  # Should be (12, 12)
        # print(f"A is :{A.shape}")  # Check its shape
        # print(f"B is :{B.shape}")  # Check its shape


        # 更新系统矩阵
        self.A = A
        self.B = B
        self.C = C  # 假设输出矩阵为单位矩阵
        



        # TODO you can change this function to allow for more passing a vector of gains
    
    def setCostMatrices(self, Qcoeff, Rcoeff):
        """
        Set the cost matrices Q and R for the MPC controller.
        Parameters:
        Qcoeff: float or array-like
            State cost coefficient(s). If scalar, the same weight is applied to all states.
            If array-like, should have a length equal to the number of states.
        Rcoeff: float or array-like
            Control input cost coefficient(s). If scalar, the same weight is applied to all control inputs.
            If array-like, should have a length equal to the number of control inputs.
        Sets:
        self.Q: ndarray
            State cost matrix.
        self.R: ndarray
            Control input cost matrix.
        """
        import numpy as np
        num_states = self.n
        num_controls = self.m
        # Process Qcoeff
        if np.isscalar(Qcoeff):
            # If Qcoeff is a scalar, create an identity matrix scaled by Qcoeff
            Q = Qcoeff * np.eye(num_states)
        else:
            # Convert Qcoeff to a numpy array
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(f"Qcoeff must be a scalar or a 1D array of length {num_states}")
            # Create a diagonal matrix with Qcoeff as the diagonal elements
            Q = np.diag(Qcoeff)
        # Process Rcoeff
        if np.isscalar(Rcoeff):
            # If Rcoeff is a scalar, create an identity matrix scaled by Rcoeff
            R = Rcoeff * np.eye(num_controls)
        else:
            # Convert Rcoeff to a numpy array
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(f"Rcoeff must be a scalar or a 1D array of length {num_controls}")
            # Create a diagonal matrix with Rcoeff as the diagonal elements
            R = np.diag(Rcoeff)
        # Assign the matrices to the object's attributes
        self.Q = Q
        self.R = R