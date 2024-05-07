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
MR	MA	ME	MT
20665.41	178.8938	1.7606	21795.857
20666.14	178.9428	1.7239	21796.389
20666.49	178.8373	1.71	21796.887
20666.46	178.9346	1.776	21797.367
20667.39	178.9166	1.8053	21797.852
20679.63	178.8026	2.3944	21798.961
20668.63	178.8364	1.7196	21799.494
20679.73	178.9656	1.7248	21799.996
20679.9	178.7023	1.6897	21800.549
20681.38	178.9606	1.6158	21801.08
33632.25	296.9022	5.2176	22252.645
33713.09	297.0009	5.2583	22253.18
33779.16	297.0367	5.226	22253.699
33986.5	297.2512	5.1722	22255.199
34086.27	297.2718	4.9672	22255.721
34274.89	297.5085	5.0913	22257.18
34354.61	297.5762	4.9576	22257.678
34568.59	297.8105	4.8639	22259.193
34717.52	297.9439	4.789	22260.213
34943.71	298.0376	4.7376	22261.717
35140.06	298.2941	4.8053	22263.217
35357.11	298.4943	4.6953	22264.707
35598.12	298.7462	4.6313	22266.199
35806.11	298.8661	4.6102	22267.729
36025.82	299.0423	4.6156	22269.189
36239.5	299.282	4.5413	22270.691
36469.04	299.3902	4.5713	22272.172
36689.36	299.584	4.5748	22273.68
36911.89	299.7541	4.5876	22275.184
37141.31	299.9243	4.5718	22276.734
37369.89	300.2742	4.6584	22278.242
37587.8	300.2986	4.5271	22279.756
37826.4	300.4486	4.4945	22281.268
38067.38	300.544	4.4473	22282.76
38291.56	300.8192	4.3985	22284.32
38526.4	300.9783	4.4916	22285.836
38764.38	301.2371	4.4005	22287.371
38976.8	301.3102	4.3874	22288.873
39204.68	301.5079	4.5364	22290.336
39429.69	301.6073	4.4321	22291.795
39659.44	301.7577	4.516	22293.313
39880.14	301.9359	4.5347	22294.836
40103.77	302.1163	4.622	22296.344
40120.62	302.1084	4.5467	22297.844
40430.54	302.3206	4.7911	22298.412
40663.65	302.4815	4.8661	22299.953
40908.42	302.6934	4.8744	22301.436
41136.48	302.8007	5.1052	22302.979
41380.39	302.9622	5.1254	22304.545
41614.32	303.1239	5.0571	22306.049
41859.18	303.2371	4.8929	22307.645
42102.72	303.4228	4.8736	22309.191
42339.29	303.5496	4.6123	22310.713
42573.54	303.637	4.4669	22312.25
3584.91	195.0698	15.0631	23073.477
3419.02	232.7911	3.4832	23485.367
3415.15	233.0648	3.483	23485.889
3416.45	233.2521	3.4145	23486.424
3403.69	232.8549	3.4598	23486.947
3400.63	232.6775	3.1863	23487.451
3399.59	232.8685	3.691	23487.961
3164.86	178.7281	6.1788	23865.092
3162.15	179.2303	6.9371	23865.615
3158.44	179.3124	6.5594	23866.146
3146.75	179.0992	6.1864	23866.68
3146.7	179.0102	6.4748	23867.188
3146.72	179.3607	6.2801	23867.721
3135.17	179.2554	7.1696	23868.229
3133.07	179.1306	6.5405	23868.752
3131.21	179.6143	5.7612	23869.295
3128.93	179.3292	6.4467	23869.848
3116.78	179.4298	6.0387	23870.404
3116.76	179.1692	6.1037	23870.932
3116.76	179.6059	6.4751	23871.477
4032.69	203.9279	16.6675	24195.887
4031.87	203.9438	16.6021	24196.381
4030.82	203.4196	16.4505	24196.9
4030.18	203.3683	16.1477	24197.379
4029.49	203.1263	16.1294	24197.896
4031.4	202.8302	16.0643	24198.4
4029.28	202.8811	16.034	24198.9
4045.97	202.4432	16.3455	24199.492
4046.04	202.4003	15.754	24199.99
4046.07	202.6088	16.4879	24200.48
4034.85	201.8949	16.0261	24200.967
4035.76	202.168	16.186	24201.484
4035.63	202.1133	16.3273	24202.033
4034.23	202.264	16.2297	24202.523
3528.48	156.3082	10.033	24531.275
3526.63	155.661	9.7741	24531.811
3582.03	162.7401	18.2351	25040.08
3579.81	161.868	18.598	25040.582
3566.25	162.2727	18.1239	25041.152
3567.45	162.1454	18.3036	25041.652
3566.22	162.1023	18.4309	25042.266
3552.83	162.8264	18.518	25042.787
3552.42	162.7355	18.1144	25043.328
3551.27	162.8679	18.1587	25043.863
3536.35	162.7651	17.979	25044.363
3536.33	162.6045	18.4867	25044.904
3536.34	162.9391	18.7731	25045.424
3536.36	162.574	18.636	25045.943
3536.38	162.3654	18.7592	25046.492
3536.41	163.3342	18.1976	25047.02
3746.6	185.068	11.9169	25380.697
3778.4	183.9788	4.2555	25803.688
3794.22	184.1295	4.1076	25804.252
3794.64	184.2587	4.1412	25804.732
3800.31	184.3193	4.6122	25805.246
3805.29	184.2926	4.3479	25805.746
3790.41	184.6473	4.0674	25806.34
3789.24	184.7828	4.1672	25806.855
3669.97	185.2041	3.7385	25807.443
3670.93	185.4587	3.9674	25808.008
3673.54	185.3165	3.67	25808.535
3703.5	185.2395	4.2217	25809.072
3670	185.4056	4.1272	25809.568
3681.59	185.2206	3.7811	25810.053
3687.43	185.5261	3.9043	25810.561
3699.97	185.5703	3.9345	25811.059
3692.72	185.4568	3.9228	25811.545
3935.94	184.1736	13.6044	26347.418
3935.09	184.1967	13.6058	26348.061
3931.56	184.2927	13.6113	26348.652
3927.78	184.3962	13.6172	26349.289
3924.16	184.4957	13.6228	26349.9
32909	219.2239	3.237	26827.32
32919.64	219.5499	3.1253	26827.82
32926.82	220.5354	3.6029	26828.859
32933.03	220.2252	3.5654	26829.359
32941.14	220.3666	3.7736	26829.891
32950.97	220.5249	4.0066	26830.439
32960.63	220.6691	4.2187	26830.939
32973.07	220.842	4.4729	26831.539
5703.73	209.1257	11.4785	27421.572
5710.61	209.1486	11.6161	27422.189
5742.04	209.2512	12.2313	27422.721
20179.04	186.8639	1.7457	28225.592
20178.92	186.8343	1.74	28226.189
20169.92	187.4751	2.2553	28226.848
20166.95	187.0386	2.0066	28227.348
20164.85	186.9662	2.0225	28227.91
20169.58	187.1706	2.0303	28228.529
20169.84	186.9816	2.326	28229.076
20170.11	186.8992	1.9705	28229.668
20169.69	186.9339	2.1355	28230.168
20171.25	186.9137	2.1604	28230.789
20157.18	186.9271	2.0108	28231.451
20156.61	186.9684	1.788	28232.012
20152.41	186.7246	1.9664	28232.6
20152.97	186.5635	2.2207	28233.129
20152.12	186.799	1.7634	28233.592
20150.38	186.943	2.0593	28234.133
20139.46	186.9168	1.9338	28234.615
20139.29	186.9581	2.1633	28235.184
20139.2	186.8162	2.2771	28235.764

"""

lines = inputs.strip().split('\n')
measurements = []
for line in lines[1:]:
    values = line.split('\t')
    measurements.append([float(value) for value in values])

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

for i, kf in enumerate(filters):
    kf.predict(0)  # Initial prediction with dt=0
    if associations[i] is not None:
        kf.measurement_association(np.array([[measurements[associations[i], 0]], [measurements[associations[i], 2]]]))
    estimated_states.append(kf.state[:2])

estimated_states = np.array(estimated_states)

plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], true_states[:, 1], 'r-', label='True State')
plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'bo-', label='Estimated Track')
plt.xlabel('Azimuth')
plt.ylabel('Elevation')
plt.title('True State vs Estimated Track')
plt.legend()
plt.grid(True)
plt.show()
