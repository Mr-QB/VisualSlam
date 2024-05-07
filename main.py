from VO import *

if __name__ == "__main__":
    VO = VisualOdometry(
        image_path="droneTestData/2011_09_26/2011_09_26_drive_0001_sync/image_00/data",
        calib_camera_file="droneTestData/2011_09_26/calib_cam_to_cam.txt",
        all_oxits_data_path="droneTestData/2011_09_26/2011_09_26_drive_0001_sync/oxts/data",
    )
    # VO = VisualOdometry(
    #     # image_path="droneTestData/droneTest.mp4",
    #     image_path="droneTestData/25th-Apr-TrajectoryTracking/droneTest.mp4",
    #     calib_camera_file="droneTestData/droneCalib.txt",
    #     all_oxits_data_path=None,
    # )
    VO.exportResult()
