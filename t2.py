import numpy as np

class KalmanFilter:
    def __init__(self, dt, measurement_var):
        self.dt = dt
        self.measurement_var = measurement_var
        self.R = np.diag([measurement_var, measurement_var])  # Measurement covariance matrix
        self.Q = np.eye(4)  # Process covariance matrix
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])  # Measurement matrix
        self.P = np.eye(4)  # Initial state covariance matrix
        self.x = np.zeros((4, 1))  # Initial state vector

    def predict(self, vx, vy, vz):
        F = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])  # State transition matrix
        # Update state estimate
        self.x = np.dot(F, self.x) + np.array([[vx * self.dt],
                                                [vy * self.dt],
                                                [vz * self.dt],
                                                [0]])  # Assuming constant velocity in z direction
        # Update process covariance matrix
        self.Q[0, 0] = (vx * self.dt) ** 2
        self.Q[1, 1] = (vy * self.dt) ** 2
        self.Q[2, 2] = (vz * self.dt) ** 2
        # Predicted covariance estimate
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        # Update state estimate
        self.x = self.x + np.dot(K, y)
        # Update covariance estimate
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)

class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, dt, measurement_var, x, y, z, vx, vy, vz, time, sig_r, sig_a, sig_e):
        super().__init__(dt, measurement_var)
        # Initialize state vector
        self.Sf = np.zeros((6, 1))
        self.Sf[0] = x
        self.Sf[1] = y
        self.Sf[2] = z
        self.Sf[3] = vx
        self.Sf[4] = vy
        self.Sf[5] = vz
        self.Filtered_Time = time

        # Initialize covariance matrix R
        self.R = np.zeros((2, 2))
        self.R[0, 0] = sig_r * sig_r * np.cos(sig_e) * np.cos(sig_e) * np.sin(sig_a) * np.sin(sig_a) + x * x * np.cos(sig_e) * np.cos(sig_e) * np.cos(sig_a) * np.cos(sig_a) + sig_a * sig_a + x * x * np.sin(sig_e) * np.sin(sig_e) * np.sin(sig_a) * np.sin(sig_a) * sig_e * sig_e
        self.R[1, 1] = sig_r * sig_r * np.cos(sig_e) * np.cos(sig_e) * np.cos(sig_a) * np.cos(sig_a) + x * x * np.cos(sig_e) * np.cos(sig_e) * np.sin(sig_a) * np.sin(sig_a) + sig_a * sig_a + x * x * np.sin(sig_e) * np.sin(sig_e) * np.cos(sig_a) * np.cos(sig_a) * sig_e * sig_e

        # Initialize pf matrix
        self.pf = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                k = i % 3
                l = j % 3
                self.pf[i, j] = self.R[k, l]

    def predict_state_covariance(self, delt, plant_noise):
        # Predict state covariance
        self.Phi = np.eye(6)
        self.Phi[0, 3] = delt
        self.Phi[1, 4] = delt
        self.Phi[2, 5] = delt
        self.Sp = np.dot(self.Phi, self.Sf)
        self.predicted_Time = self.Filtered_Time + delt

        T_3 = (delt * delt * delt) / 3.0
        T_2 = (delt * delt) / 2.0
        self.Q = np.zeros((6, 6))
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = delt
        self.Q[4, 4] = delt
        self.Q[5, 5] = delt
        self.Q = np.dot(self.Q, plant_noise)
        self.Pp = np.dot(np.dot(self.Phi, self.pf), self.Phi.T) + self.Q

    def Filter_state_covariance(self, H, Z):
        # Filter state covariance
        self.Prev_Sf = self.Sf.copy()
        self.Prev_Filtered_Time = self.Filtered_Time
        self.S = self.R + np.dot(np.dot(H, self.Pp), H.T)
        self.K = np.dot(np.dot(self.Pp, H.T), np.linalg.inv(self.S))
        self.Inn = Z - np.dot(H, self.Sp)
        self.Sf = self.Sp + np.dot(self.K, self.Inn)
        self.pf = np.dot((self.Inn - np.dot(self.K, H)), self.Pp)
        self.Filtered_Time = self.Meas_Time

# Example usage:
if __name__ == "__main__":
    dt = 1.0  # Time step
    measurement_var = 0.1  # Measurement noise variance

    # Initial conditions
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 1.0
    vy = 0.5
    vz = 0.2
    time = 0.0
    sig_r = 0.1
    sig_a = 0.1
    sig_e = 0.1

    # Initialize Extended Kalman filter
    ekf = ExtendedKalmanFilter(dt, measurement_var, x, y, z, vx, vy, vz, time, sig_r, sig_a, sig_e)

    # Example measurements and H matrix
    measurements = np.array([[10, 45], [9.8, 44.5], [9.5, 44]])  # Range and azimuth for t=0, t=1, t=2
    H = np.eye(6)[:2, :]  # Measurement matrix

    for i in range(len(measurements)):
        # Prediction step
        ekf.predict(vx, vy, vz)
        
        # Update step
        ekf.update(np.array(measurements[i]))

        print(f"Time step {i + 1}:")
        print("Predicted state:", ekf.x)
        print("Updated state:", ekf.x)
