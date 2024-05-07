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
            H = np.eye(6)[:3]  # Assuming only position measurements

            # Residual
            y = z - np.dot(H, self.state)

            # Kalman gain
            S = np.dot(np.dot(H, self.covariance), H.T) + self.measurement_noise_cov
            K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))

            # Update state and covariance
            self.state += np.dot(K, y)
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
kf.initialize(x=0, y=0, z=0, vx=0, vy=0, vz=0, sig_r=0.1, sig_a=0.1, sig_e_sqr=0.1)

# Step 2: Prediction
kf.predict(dt=1, process_noise_cov=np.eye(6))

# Step 3: Measurement Association
measurements = np.array([[1, 2, 3], [4, 5, 6]])  # Example measurements
associated_measurements = kf.associate_measurements(measurements)

# Step 4: Update
kf.update(associated_measurements)

# Step 5: Track Management
kf.track_management(measurements)