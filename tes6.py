import numpy as np
import matplotlib.pyplot as plt

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
        self.state = self.Sf  # Update state with the filtered state

def jpda_association(filters, measurements, association_threshold):
    num_filters = len(filters)
    num_measurements = len(measurements)
    associations = []

    # Perform association
    for j in range(num_measurements):
        max_likelihood = 0
        max_index = -1
        for i, kf in enumerate(filters):
            z = measurements[j]
            H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]])
            S = kf.measurement_noise_covariance + np.dot(np.dot(H, kf.covariance), H.T)
            innov = z - np.dot(H, kf.state)
            likelihood = 1 / np.sqrt(np.linalg.det(2 * np.pi * S)) * np.exp(-0.5 * np.dot(np.dot(innov.T, np.linalg.inv(S)), innov))
            if np.isscalar(likelihood) and likelihood > max_likelihood:
                max_likelihood = likelihood
                max_index = i
        if max_likelihood > association_threshold:
            associations.append(max_index)
        else:
            associations.append(None)

    return associations

# New inputs
inputs = """
20665.41 178.8938 1.7606 21795.857
20666.14 178.9428 1.7239 21796.389
20666.49 178.8373 1.71 21796.887
20666.46 178.9346 1.776 21797.367
20667.39 178.9166 1.8053 21797.852
20679.63 178.8026 2.3944 21798.961
... (the rest of the inputs)
"""

lines = inputs.strip().split('\n')
measurements = []
for i, line in enumerate(lines[1:], 1):
    values = line.split()
    try:
        measurements.append([float(value) for value in values])
    except ValueError as e:
        print(f"Error in line {i}: {e}")

measurements = np.array(measurements)

# Initialize Kalman filter
initial_state = np.array([[0], [0], [0], [0], [0], [0]])  # Initial state: [x, x_dot, y, y_dot, azimuth, elevation]
initial_covariance = np.eye(6)  # Initial covariance matrix
process_noise_covariance = np.eye(6) * 0.01  # Process noise covariance
measurement_noise_covariance = np.eye(2) * 0.1  # Measurement noise covariance

filters = [KalmanFilter(initial_state, initial_covariance, process_noise_covariance, measurement_noise_covariance) for _ in range(3)]

# JPDA association
association_threshold = 0.5
associations = jpda_association(filters, measurements, association_threshold)

# Plotting
true_states = measurements[:, :2]
estimated_states = []
estimated_velocities = []

for i, kf in enumerate(filters):
    time = measurements[i, 3]  # Get the time from the measurement
    dt = time - measurements[i-1, 3] if i > 0 else 0  # Calculate time difference
    kf.predict(dt)
    if associations[i] is not None:
        kf.measurement_association(np.array([[measurements[associations[i], 0]], [measurements[associations[i], 2]]]))
    estimated_states.append([time, kf.state[0][0], kf.state[2][0]])
    estimated_velocities.append([time, kf.state[1][0], kf.state[3][0]])

estimated_states = np.array(estimated_states)
estimated_velocities = np.array(estimated_velocities)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(true_states[:, 0], true_states[:, 1], 'r-', label='True State')
plt.plot(estimated_states[:, 1], estimated_states[:, 2], 'bo-', label='Estimated Track')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('True State vs Estimated Track')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(estimated_velocities[:, 0], estimated_velocities[:, 1], 'g-', label='X Velocity')
plt.plot(estimated_velocities[:, 0], estimated_velocities[:, 2], 'm-', label='Y Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Estimated Velocities')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()