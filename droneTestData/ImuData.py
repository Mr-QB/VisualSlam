import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DataRaw import RawData


class ImuData:
    def __init__(self):
        self.imu_data_path = "droneTestData/ImuData.plk"
        self.imu_data = pd.read_pickle(self.imu_data_path)

    def test(self):
        print(self.imu_data)


if __name__ == "__main__":
    raw_data = RawData()
    raw_data.dataProcessing()

    imu_data = ImuData()
    imu_data.test()
