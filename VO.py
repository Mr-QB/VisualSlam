import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
from math import sin, cos
import time


class VisualOdometry:
    def __init__(
        self, image_path, calib_camera_file, all_oxits_data_path, gimbal_data_path
    ):
        self.imame_path = image_path
        self.calib_camera_file = calib_camera_file
        self.all_oxits_data_path = all_oxits_data_path
        self.trajactory_predict = []
        self.have_gps = True
        self.gimbal_data_path = gimbal_data_path
        self.getGimbalData()

        self.image_list = self.getImageList()

        self.getCameraMatrix()

        try:
            self.all_oxits_data = self.getAllOxitsData()
            self.pose_base = self.getPoseBase()
        except:
            # Initializes the camera translation vector t_f and rotation matrix R_f.
            t_f, R_f = np.zeros((3, 1)), np.eye(3)
            self.pose_base = self.form_transf(R_f, np.squeeze(t_f))
            self.have_gps = False

        self.getTrajactory()

    def getGimbalData(self):
        def rotationMatrixFromEuler(pitch, roll, yaw):
            pitch = np.radians(pitch)
            roll = np.radians(roll)
            yaw = np.radians(yaw)

            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)],
                ]
            )
            R_y = np.array(
                [
                    [np.cos(roll), 0, np.sin(roll)],
                    [0, 1, 0],
                    [-np.sin(roll), 0, np.cos(roll)],
                ]
            )
            R_z = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1],
                ]
            )

            R = np.dot(R_z, np.dot(R_y, R_x))
            return R

        self.gimbal_data = pd.read_pickle(self.gimbal_data_path)
        self.gimbal_rotation_matrix = {}
        for index in range(23, self.gimbal_data.shape[0]):
            pitch = (
                self.gimbal_data.loc[index]["GIMBAL.pitch"]
                - self.gimbal_data.loc[index - 1]["GIMBAL.pitch"]
            )
            roll = (
                self.gimbal_data.loc[index]["GIMBAL.roll"]
                - self.gimbal_data.loc[index - 1]["GIMBAL.roll"]
            )
            yaw = (
                self.gimbal_data.loc[index]["GIMBAL.yaw"]
                - self.gimbal_data.loc[index - 1]["GIMBAL.yaw"]
            )

            rotation_matrix = rotationMatrixFromEuler(pitch, roll, yaw)
            self.gimbal_rotation_matrix[
                (self.gimbal_data.loc[index]["OSD.flyTime [s]"])
            ] = rotation_matrix

    def getCameraMatrix(self):
        filedata = {}
        with open(self.calib_camera_file, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                if "S_" in key:
                    filedata[key] = np.array(list(map(float, value.split())))
                elif any(x in key for x in ["K_", "D_", "R_", "T_", "P_rect_"]):
                    filedata[key] = np.array(list(map(float, value.split())))
        self.projection_matrix = filedata["P_rect_00"].reshape((3, 4))
        self.camera_matrix = self.projection_matrix[0:3, 0:3]

    def getImageList(self):
        # Retrieves a list of images from a specified folder path.
        self.image_list = []
        try:
            for filename in sorted(os.listdir(self.imame_path)):
                if filename.endswith(".png"):
                    image_path = os.path.join(self.imame_path, filename)
                    self.image_list.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            return self.image_list
        except:
            video_capture = cv2.VideoCapture(self.imame_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            success, frame = video_capture.read()
            count = 0
            self.frame_times = []
            while success:
                if count % 3 == 0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.image_list.append(gray_frame)
                    current_time = count / fps
                    self.frame_times.append(current_time)
                    success, frame = video_capture.read()
                count += 1
            video_capture.release()
            return self.image_list

    def form_transf(self, R, t):
        # Makes a transformation matrix from the given rotation matrix and translation vector
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def featureMatches(self, pre_image, curr_image):
        # Matches feature points between two consecutive images using the Lucas-Kanade optical flow method.
        orb = cv2.ORB_create(4000)
        prev_features, prev_descriptors = orb.detectAndCompute(pre_image, None)
        prev_features_ = np.array(
            [prev_features[i].pt for i in range(len(prev_features))], dtype=np.float32
        )
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        matches, status, error = cv2.calcOpticalFlowPyrLK(
            pre_image, curr_image, prev_features_, None, **lk_params
        )
        status = status.flatten() == 1
        curr_features = matches[status.flatten() == 1]
        prev_features_ = prev_features_[status.flatten() == 1]
        return prev_features_, curr_features

    def get_pose(self, prev_features, curr_features, numFrame):
        # Calculates the transformation matrix
        E, _ = cv2.findEssentialMat(
            prev_features, curr_features, self.camera_matrix, threshold=1
        )  # Essential matrix

        # Decompose the Essential matrix into R and t
        # R, t = self.decomp_essential_mat(E, prev_features, curr_features)
        _, R, t, _ = cv2.recoverPose(
            E, curr_features, prev_features, cameraMatrix=self.camera_matrix
        )

        # Get transformation matrix
        transformation_matrix = self.form_transf(R, np.squeeze(t))
        return transformation_matrix

    def sumZCalRelativeCcale(self, R, t, prev_features, curr_features):
        T = self.form_transf(R, t)  # Get the transformation matrix
        # Make the projection matrix
        P = np.matmul(np.concatenate((self.camera_matrix, np.zeros((3, 1))), axis=1), T)
        hom_Q1 = cv2.triangulatePoints(
            self.projection_matrix, P, prev_features.T, curr_features.T
        )  # Triangulate the 3D points
        hom_Q2 = np.matmul(T, hom_Q1)  # Also seen from cam 2
        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)
        # Form point pairs and calculate the relative scale
        relative_scale = np.mean(
            np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
            / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
        )
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

    def decomp_essential_mat(self, E, prev_features, curr_features):

        R1, R2, t = cv2.decomposeEssentialMat(E)  # Decompose the essential matrix
        t = np.squeeze(t)
        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = self.sumZCalRelativeCcale(R, t, prev_features, curr_features)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmin(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    def getPoseBase(self):
        # all_oxits_data = self.getAllOxitsData(self.all_oxits_data_path)
        lat_0 = self.all_oxits_data[2][0]
        lon_0 = self.all_oxits_data[2][1]
        alt_0 = self.all_oxits_data[2][2]
        roll_0 = self.all_oxits_data[2][3]
        pitch_0 = self.all_oxits_data[2][3]
        yaw_0 = self.all_oxits_data[2][4]
        # Rotation matrix
        R = np.array(
            [
                [
                    cos(yaw_0) * cos(pitch_0),
                    -sin(yaw_0) * cos(roll_0) + cos(yaw_0) * sin(pitch_0) * sin(roll_0),
                    sin(yaw_0) * sin(roll_0) + cos(yaw_0) * sin(pitch_0) * cos(roll_0),
                ],
                [
                    sin(yaw_0) * cos(pitch_0),
                    cos(yaw_0) * cos(roll_0) + sin(yaw_0) * sin(pitch_0) * sin(roll_0),
                    -cos(yaw_0) * sin(roll_0) + sin(yaw_0) * sin(pitch_0) * cos(roll_0),
                ],
                [-sin(pitch_0), cos(pitch_0) * sin(roll_0), cos(pitch_0) * cos(roll_0)],
            ]
        )
        T = np.array([[lat_0], [lon_0], [alt_0]])  # Translation vector

        # Pose Matrix
        # pose_matrix_0 = np.vstack([np.hstack([R, T]), [0, 0, 0, 1]]
        pose_matrix_0 = self.form_transf(R, np.squeeze(T))
        return pose_matrix_0

    def getTrajactory(self):
        start_time = time.time()
        # Computes the trajectory of the camera motion based
        self.trajactory_predict = []
        cur_pose = self.pose_base
        self.trajactory_predict.append((cur_pose[0, 3], cur_pose[2, 3], cur_pose[3, 3]))
        pre_image = self.image_list[0]
        for numFrame in range(1, len(self.image_list)):
            curr_image = self.image_list[numFrame]
            prev_features, curr_features = self.featureMatches(pre_image, curr_image)
            transf = self.get_pose(prev_features, curr_features, numFrame)

            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            # print(cur_pose, cur_pose[0, 3], cur_pose[2, 3], cur_pose[3, 3])
            self.trajactory_predict.append(
                (cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3])
            )
            pre_image = curr_image
        # self.trajactory_predict.pop(0)
        # np.save("trajactory_predict", np.array(self.trajactory_predict))
        end_time = time.time()
        print("Trajectory has been predicted......\n")
        print("execution time: {} seconds\n".format(end_time - start_time))
        # return self.trajactory_predict

    def getAllOxitsData(self):
        all_oxits_data = []
        for filename in os.listdir(self.all_oxits_data_path):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(self.all_oxits_data_path, filename), "r"
                ) as file:
                    array = [float(x) for x in file.read().strip().split()]
                    all_oxits_data.append(array)

        return all_oxits_data

    def drawGPS(self):
        self.all_oxits_data = self.getAllOxitsData()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Ground Truth GPS")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        lat = []  # latitude of the oxts-unit (deg)
        lon = []  # latitude of the oxts-unit (deg)
        alt = []  # altitude of the oxts-unit (m)

        for i in range(len(self.all_oxits_data)):
            lat.append(self.all_oxits_data[i][0])
            lon.append(self.all_oxits_data[i][1])
            alt.append(self.all_oxits_data[i][2])

        (ln,) = ax.plot([], [], [], marker="o")

        min_lat_value, max_lat_value = float(1000), float(1)
        min_lon_value, max_lon_value = float(1000), float(1)
        min_alt_value, max_alt_value = float(1000), float(1)

        def update(frame):

            nonlocal min_lat_value, max_lat_value, min_lon_value, max_lon_value, min_alt_value, max_alt_value

            ln.set_data(lon[:frame], lat[:frame])
            ln.set_3d_properties(alt[:frame])

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

        return FuncAnimation(fig, update, frames=len(self.image_list), blit=True)

    def drawTrajactory(self):
        # Draws the trajectory of the visual odometry (VO) system using a list of points.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Trajactory Predict")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        (ln,) = ax.plot([], [], [], marker="o")

        min_x_value, max_x_value = float(100), float(1)
        min_y_value, max_y_value = float(1000), float(1)
        min_z_value, max_z_value = float(1000), float(1)

        x = [pose[0] for pose in self.trajactory_predict]
        y = [pose[1] for pose in self.trajactory_predict]
        z = [pose[2] for pose in self.trajactory_predict]

        def update(frame):

            nonlocal min_x_value, max_x_value, min_y_value, max_y_value, min_z_value, max_z_value
            cv2.imshow("Raw data", self.image_list[frame])

            ln.set_data(x[:frame], y[:frame])
            ln.set_3d_properties(z[:frame])

            min_x_value = min(min_x_value, x[frame])
            max_x_value = max(max_x_value, x[frame])
            min_y_value = min(min_y_value, y[frame])
            max_y_value = max(max_y_value, y[frame])
            min_z_value = min(min_z_value, z[frame])
            max_z_value = max(max_z_value, z[frame])
            ax.set_xlim([min_x_value, max_x_value])
            ax.set_ylim([min_y_value, max_y_value])
            ax.set_zlim([min_z_value, max_z_value])
            return (ln,)

        trajectory_pre = FuncAnimation(
            fig, update, frames=len(self.image_list), blit=True
        )
        return trajectory_pre

    def exportResult(self):
        if self.have_gps:
            gt_gps = self.drawGPS()
        trajectory_pre = self.drawTrajactory()
        plt.show()
