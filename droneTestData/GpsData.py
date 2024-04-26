import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from RawData import RawData


class GpsData:
    def __init__(self):
        self.gps_data_path = "droneTestData/GpsData.plk"
        self.gps_data = pd.read_pickle(self.gps_data_path)

    def plotData(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("GPS Data Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        lat = self.gps_data["OSD.latitude"].tolist()  # latitude of the oxts-unit (deg)
        lon = self.gps_data["OSD.longitude"].tolist()  # latitude of the oxts-unit (deg)
        alt = self.gps_data["OSD.height [ft]"].tolist()  # altitude of the oxts-unit (m)

        (ln,) = ax.plot([], [], [], marker="o")

        min_lat_value, max_lat_value = float(1000), float(1)
        min_lon_value, max_lon_value = float(1000), float(1)
        min_alt_value, max_alt_value = float(1000), float(1)

        def update(frame):
            frame += 1

            nonlocal min_lat_value, max_lat_value, min_lon_value, max_lon_value, min_alt_value, max_alt_value

            ln.set_data(lon[:frame], lat[:frame])
            ln.set_3d_properties(alt[:frame])

            # ax.scatter(lon[:frame], lat[:frame], alt[:frame], "r-")

            min_lat_value = min(min_lat_value, lat[frame])
            max_lat_value = max(max_lat_value, lat[frame])
            min_lon_value = min(min_lon_value, lon[frame])
            max_lon_value = max(max_lon_value, lon[frame])
            min_alt_value = min(min_alt_value, alt[frame])
            max_alt_value = max(max_alt_value, alt[frame])

            ax.set_ylim([min_lat_value, max_lat_value])
            ax.set_xlim([min_lon_value, max_lon_value])
            ax.set_zlim([min_alt_value, max_alt_value])
            return (ln,)

        gps_plot = FuncAnimation(fig, update, frames=len(lat), blit=True)
        plt.show()

    def test(self):
        self.plotData()
        print(len(self.gps_data["OSD.latitude"]))


if __name__ == "__main__":
    raw_data = RawData()
    raw_data.dataProcessing()

    gps_data = GpsData()
    gps_data.test()
