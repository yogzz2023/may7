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

# Example usage:
if __name__ == "__main__":
    dt = 1.0  # Time step
    measurement_var = 0.1  # Measurement noise variance

    # Initialize Kalman filter
    kf = KalmanFilter(dt, measurement_var)

    # Example measurements
    measurements = [
        [10, 45],  # Range and azimuth for t=0
        [9.8, 44.5],  # Range and azimuth for t=1
        [9.5, 44],  # Range and azimuth for t=2
    ]

    # Example velocities (constant)
    vx = 1.0
    vy = 0.5
    vz = 0.2

    for i in range(len(measurements)):
        # Prediction step
        kf.predict(vx, vy, vz)
        
        # Update step
        kf.update(np.array(measurements[i]))

        print(f"Time step {i + 1}:")
        print("Predicted state:", kf.x)
        print("Updated state:", kf.x)
