import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class RawData:

    def __init__(
        self,
        raw_data_path="droneTestData/25th-Apr-TrajectoryTracking/DJIFlightRecord_2024-04-25_[13-58-21].csv",
    ):
        self.raw_data_path = raw_data_path

        self.readRawData()

    def readRawData(self):
        self.raw_data = pd.read_csv(self.raw_data_path)

    def exportImuData(self):
        colum_name = ["OSD.pitch", "OSD.roll", "OSD.yaw"]
        self.imu_data = self.raw_data[colum_name]
        self.imu_data.to_pickle("droneTestData/ImuData.plk")

    def exportGpsData(self):
        colum_name = ["OSD.latitude", "OSD.longitude", "OSD.height [ft]"]
        self.gps_data = self.raw_data[colum_name]
        self.gps_data.to_pickle("droneTestData/GpsData.plk")

    def dataProcessing(self):
        self.exportImuData()
        self.exportGpsData()


# if __name__ == "__main__":
#     raw_data = RawData()
#     raw_data.dataProcessing()
