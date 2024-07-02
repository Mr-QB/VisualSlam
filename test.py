import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gps_data = pd.read_pickle("droneTestData/GpsData.plk")
lat = gps_data["OSD.latitude"].tolist()  # latitude of the oxts-unit (deg)
lon = gps_data["OSD.longitude"].tolist()  # latitude of the oxts-unit (deg)
alt = gps_data["OSD.height [ft]"].tolist()  # altitude of the oxts-unit (m)
fig = plt.figure()
ax = fig.add_subplot(131, projection="3d")
ax.plot3D(lat, lon, alt, "red")


trajactory_predict = np.load(
    "droneTestData/25th-Apr-TrajectoryTracking/trajactory_predict.npy"
)
x = [pose[0] for pose in trajactory_predict]
y = [pose[1] for pose in trajactory_predict]
z = [pose[2] for pose in trajactory_predict]
ax = fig.add_subplot(132, projection="3d")
ax.plot3D(x, y, z, "green")

trajactory_predict_2 = np.load(
    "droneTestData/25th-Apr-TrajectoryTracking/trajactory_predict_2.npy"
)
x = [pose[0] for pose in trajactory_predict_2]
y = [pose[1] for pose in trajactory_predict_2]
z = [pose[2] for pose in trajactory_predict_2]
ax = fig.add_subplot(133, projection="3d")
ax.set_xlabel("X axis label")
ax.set_ylabel("Y axis label")
ax.set_zlabel("Z axis label")
ax.plot3D(x, y, z, "blue")
plt.show()
