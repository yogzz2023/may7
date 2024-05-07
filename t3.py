import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.measurement_noise_covariance = measurement_noise_covariance

    def predict(self, dt):
        # Prediction step
        # Assuming constant velocity model
        F = np.array([[1, dt, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]])

        self.Q = np.dot(self.process_noise_covariance, 20)  # Assuming plant noise is 20
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + self.Q

    def measurement_association(self, measurement):
        # Measurement association step
        # Assuming only one measurement for simplicity
        z = measurement
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

        self.S = self.measurement_noise_covariance + np.dot(np.dot(H, self.covariance), H.T)
        self.K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(self.S))
        self.Inn = z - np.dot(H, self.state)
        self.Sf = self.state + np.dot(self.K, self.Inn)
        self.covariance = self.covariance + np.dot(np.dot(self.K, H), self.covariance)

    def update_state(self):
        # Update state step
        self.state = self.Sf

# Convert the sample measurements into numpy arrays
measurements = np.array([
    [20665.41, 178.8938, 1.7606, 21795.857],
    [20666.14, 178.9428, 1.7239, 21796.389],
    [20666.49, 178.8373, 1.71, 21796.887],
    [20666.46, 178.9346, 1.776, 21797.367],
    [20667.39, 178.9166, 1.8053, 21797.852],
    [20679.63, 178.8026, 2.3944, 21798.961],
    [20668.63, 178.8364, 1.7196, 21799.494],
    [20679.73, 178.9656, 1.7248, 21799.996],
    [20679.9, 178.7023, 1.6897, 21800.549],
    [20681.38, 178.9606, 1.6158, 21801.08],
    [33632.25, 296.9022, 5.2176, 22252.645],
    [33713.09, 297.0009, 5.2583, 22253.18],
    [33779.16, 297.0367, 5.226, 22253.699],
    [33986.5, 297.2512, 5.1722, 22255.199],
    [34086.27, 297.2718, 4.9672, 22255.721],
    [34274.89, 297.5085, 5.0913, 22257.18],
    [34354.61, 297.5762, 4.9576, 22257.678],
    [34568.59, 297.8105, 4.8639, 22259.193]
])

# Initialize Kalman filter
initial_state = np.array([[0], [0], [0], [0], [0], [0]])  # Initial state: [x, x_dot, y, y_dot, azimuth, elevation]
initial_covariance = np.eye(6)  # Initial covariance matrix
process_noise_covariance = np.eye(6) * 0.01  # Process noise covariance
measurement_noise_covariance = np.eye(2) * 0.1  # Measurement noise covariance

kf = KalmanFilter(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance)

# Process each measurement
for measurement in measurements:
    range_, azimuth, elevation, time_ = measurement
    dt = time_ - kf.state[4] if kf.state[4] != 0 else 0  # Assuming dt is from measurement time
    kf.predict(dt)
    kf.measurement_association(np.array([[range_], [elevation]]))
    kf.update_state()

print("Final state:", kf.state)
