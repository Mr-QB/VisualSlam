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
    
    def exportGimbalData(self):
        OSD_flyTime_deg = self.raw_data["OSD.flyTime [s]"]
        GIMBAL_pitch_deg = self.raw_data["GIMBAL.pitch"]
        GIMBAL_roll_deg = self.raw_data["GIMBAL.roll"]
        GIMBAL_yaw_deg = self.raw_data["GIMBAL.yaw"]

        deg_to_rad = np.pi / 180.0

        OSD_flyTime_rad = OSD_flyTime_deg * deg_to_rad
        GIMBAL_pitch_rad = GIMBAL_pitch_deg * deg_to_rad
        GIMBAL_roll_rad = GIMBAL_roll_deg * deg_to_rad
        GIMBAL_yaw_rad = GIMBAL_yaw_deg * deg_to_rad

        self.gimbal_data = pd.DataFrame({
            "OSD.flyTime [s]": OSD_flyTime_rad,
            "GIMBAL.pitch": GIMBAL_pitch_rad,
            "GIMBAL.roll": GIMBAL_roll_rad,
            "GIMBAL.yaw": GIMBAL_yaw_rad
        })
        self.gimbal_data.to_pickle("droneTestData/GimbalData.plk")

    def dataProcessing(self):
        self.exportImuData()
        self.exportGpsData()
        self.exportGimbalData()

if __name__ == "__main__":
    raw_data = RawData()
    raw_data.dataProcessing()
