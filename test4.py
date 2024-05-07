import numpy as np

class KalmanFilter:
    def __init__(self):
        # Initialize necessary attributes
        self.tracks = []

    def initialize(self, x, y, z, vx, vy, vz, sig_r, sig_a, sig_e_sqr):
        # Initialize state vector, covariance matrix, measurement noise covariance
        self.state = np.array([[x], [y], [z], [vx], [vy], [vz]])  # State vector
        self.covariance = np.eye(6)  # Covariance matrix
        self.measurement_noise_cov = np.diag([sig_r**2, sig_a**2, sig_e_sqr])  # Measurement noise covariance

    def predict(self, dt, process_noise_cov):
        # Prediction step
        # State transition matrix
        F = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        # Predicted state and covariance
        self.state = np.dot(F, self.state)
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + process_noise_cov

    def associate_measurements(self, measurements):
        # Measurement association using JPDA
        # For simplicity, select the closest measurement to each predicted target
        associated_measurements = []
        for track in self.tracks:
            target = track['prediction']
            # Calculate distance to all measurements
            distances = [np.linalg.norm(target - measurement) for measurement in measurements]
            # Find the closest measurement
            closest_measurement_index = np.argmin(distances)
            associated_measurements.append(measurements[closest_measurement_index])
        return associated_measurements

    def update(self, measurements):
    # Update step
        for z in measurements:
        # Measurement matrix
            H = np.eye(6)[:, :3]  # Extract position components only

        # Extract position components from the measurement and reshape
            z_position = z[:3].reshape(-1, 1)

        # Residual
            y = z_position - np.dot(H, self.state[:3])

        # Kalman gain
            S = np.dot(np.dot(H, self.covariance), H.T) + self.measurement_noise_cov
            K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

        # Update state and covariance
            self.state[:3] += np.dot(K, y)
            self.covariance -= np.dot(np.dot(K, H), self.covariance)


    def track_management(self, measurements):
        # Remove tracks with no measurements associated
        self.tracks = [track for track in self.tracks if track['measurement'] is not None]

        # Associate measurements with tracks (simplest case: one measurement per track)
        for track in self.tracks:
            track['measurement'] = self.associate_measurements([track['prediction']])[0]

        # Initialize new tracks for unassociated measurements
        for measurement in measurements:
            # Check if measurement is already associated with a track
            associated = False
            for track in self.tracks:
                if np.array_equal(track['measurement'], measurement):
                    associated = True
                    break
            if not associated:
                # If not associated, initialize a new track
                new_track = {'prediction': measurement, 'measurement': None}
                self.tracks.append(new_track)

# Example usage:
kf = KalmanFilter()

# Step 1: Initialization
kf.initialize(x=20665.41, y=178.8938, z=1.7606, vx=21795.857, vy=0, vz=0, sig_r=0.1, sig_a=0.1, sig_e_sqr=0.1)
print("Initial state vector:", kf.state.ravel())
print("Initial covariance matrix:\n", kf.covariance)

# Step 2: Prediction
kf.predict(dt=1, process_noise_cov=np.eye(6))
print("\nPredicted state vector:", kf.state.ravel())
print("Predicted covariance matrix:\n", kf.covariance)

# Step 3: Measurement Association
measurements = np.array([[20666.14, 178.9428, 1.7239, 21796.389], 
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
                         [34568.59, 297.8105, 4.8639, 22259.193],
                         [34717.52, 297.9439, 4.789, 22260.213]])  # Example measurements
print("\nAssociated measurements:", measurements)

# Step 4: Update
kf.update(measurements)
print("\nUpdated state vector:", kf.state.ravel())
print("Updated covariance matrix:\n", kf.covariance)

# Step 5: Track Management
kf.track_management(measurements)
