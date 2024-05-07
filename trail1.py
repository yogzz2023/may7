import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, measurement_variance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.measurement_variance = measurement_variance

    def predict(self, dt, transition_matrix):
        # Predict state
        self.state = np.dot(transition_matrix, self.state)
        # Predict covariance
        self.covariance = np.dot(np.dot(transition_matrix, self.covariance), transition_matrix.T)

    def update(self, measurement, observation_matrix):
        # Kalman gain
        kalman_gain = np.dot(np.dot(self.covariance, observation_matrix.T), 
                             np.linalg.inv(np.dot(np.dot(observation_matrix, self.covariance), observation_matrix.T) + self.measurement_variance))
        # Update state
        self.state = self.state + np.dot(kalman_gain, (measurement - np.dot(observation_matrix, self.state)))
        # Update covariance
        self.covariance = np.dot((np.eye(self.covariance.shape[0]) - np.dot(kalman_gain, observation_matrix)), self.covariance)

def main():
    # Provided measurements
    measurements = [
        [20665.41, 178.8938, 1.7606, 21795.857],
        [20666.14, 178.9428, 1.7239, 21796.389],
        # Add more measurements here...
    ]

    # Convert measurements to numpy array
    measurements = np.array(measurements)

    # Initialize Kalman filter
    initial_state = np.array([measurements[0, 0], measurements[0, 1], measurements[0, 2]])
    initial_covariance = np.eye(3) * 1e3  # Initial covariance matrix
    measurement_variance = 1.0  # Measurement variance
    kf = KalmanFilter(initial_state, initial_covariance, measurement_variance)

    # Transition matrix (assuming constant velocity)
    dt = 1  # Time step
    transition_matrix = np.array([[1, dt, 0],
                                  [0, 1, dt],
                                  [0, 0, 1]])

    # Observation matrix
    observation_matrix = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])

    # Lists to store filtered values
    filtered_values = []

    # Perform Kalman filtering
    for measurement in measurements:
        kf.predict(dt, transition_matrix)
        kf.update(measurement[:3], observation_matrix)
        filtered_values.append(kf.state)

    # Convert filtered values to numpy array
    filtered_values = np.array(filtered_values)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(measurements[:, 3], label='Measurements', marker='o')
    plt.plot(filtered_values[:, 0], label='Filtered Range')
    plt.plot(filtered_values[:, 1], label='Filtered Azimuth')
    plt.plot(filtered_values[:, 2], label='Filtered Elevation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
