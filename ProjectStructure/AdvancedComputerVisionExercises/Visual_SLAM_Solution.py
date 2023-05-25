import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from List_Bundler3 import ListBundler
from Bundle_Adjustment import run_BA

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics
from mpl_toolkits.mplot3d import Axes3D
from itertools import groupby


class VisualOdometry():
    def __init__(self, data_dir, frame_limit):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'), frame_limit)
        self.images_l = self._load_images(os.path.join(data_dir, 'image_l'), frame_limit)
        self.images_r = self._load_images(os.path.join(data_dir, 'image_r'), frame_limit)

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()
        self.disparities2 = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        self.tp_1 = np.array([])
        self.tp_2 = np.array([])
        self.Q_1 = np.array([])

        self.tp_1_to_compare = []
        self.tp_2_to_compare = []
        self.Q_1_to_compare = []
        self.Q_2_to_compare = []

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath, frame_limit):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        poses = poses[:frame_limit]
        return poses

    @staticmethod
    def _load_images(filepath, frame_limit):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        image_paths = image_paths[:frame_limit]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """

        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Correct the coordinate for the point
            for keypt in keypoints:
                keypt.pt = (keypt.pt[0] + x, keypt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints

        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """

        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        # Combine the masks
        in_bounds = np.logical_and(mask1, mask2)

        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images

        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):

        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]  # for i = 1 then [0:2] i = 1, then keypoint frame, is 0.

        # Get teh tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # save the trackpoints
        self.tp_1 = np.reshape(np.append(self.tp_1, tp1_l), (-1, 2))
        self.tp_2 = np.reshape(np.append(self.tp_2, tp2_l), (-1, 2))
        # print(Q1)
        self.Q_1 = Q1  # np.reshape(np.append(self.Q_1,Q1),(-1,3))
        # self.Q_2 = np.reshape(np.append(self.Q_2,Q2),(-1,3))

        self.tp_1_to_compare.append(tp1_l)
        self.tp_2_to_compare.append(tp2_l)
        self.Q_1_to_compare.append(Q1)
        self.Q_2_to_compare.append(Q2)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix

    def project_built_in(Qs, cam_params):
        cam_params = np.array(cam_params)
        # Extract camera intrinsic parameters
        f_x = 7.188560000000e+02
        f_y = f_x
        c_x = 6.071928000000e+02
        c_y = 1.852157000000e+02

        camera_matrix = np.array([[f_x, 0., c_x], [0., f_y, c_y], [0., 0., 1.]])
        distortion_coeffs = np.array([0., 0., 0., 0.])
        rvec = cam_params[:, :3]
        tvec = cam_params[:, 3:6]

        # Reshape 3D points to match OpenCV's input format
        Qs_reshaped = Qs.reshape((-1, 1, 3))

        # Project 3D points to 2D image coordinates using OpenCV's function
        qs_list = []
        for k in range(len(Qs_reshaped)):
            qs_proj, _ = cv2.projectPoints(Qs_reshaped[k], rvec[k], tvec[k], camera_matrix, distortion_coeffs)
            qs_list.append(qs_proj)
        qs_list = np.array(qs_list)
        qs_proj_reshaped = np.reshape(qs_list, [-1, 2])

        return qs_proj_reshaped

    def final_new_pose(self, opt_params, stop_point):
        estimated_better_path = []
        for i in range(0, int(stop_point), 9):
            estimated_better_path.append((opt_params[i + 3], opt_params[i + 5]))
        return estimated_better_path

    def track_and_triangulate(self, i, keypoint):
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, keypoint)

        # Calculate the disparitie
        self.disparities2.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities2[i - 1],
                                                             self.disparities2[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        #transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return tp1_l, tp2_l, Q1, Q2

    def track_point_gen(self, from_frame,to_frame):
        start = from_frame
        lengh = to_frame-from_frame

        tp1list = []
        tp2list = []
        q1list = []
        q2list = []
        frameindex = []
        keypointindex = []
        output = []

        keypoints = self.get_tiled_keypoints(self.images_l[start], 20, 20)
        #Generate keypoints from frame 0
        for i in range(lengh): # Run for all frames. This function in essense, ceates keypoints in frame 0, then tries to track them though all frames
            if i == 0:
                print("i=0")
            else:
                tp1, tp2, Q1, Q2 = self.track_and_triangulate(i, keypoints) # returns trackable keypoints from tp1 to tp2
                #when run for a second time, then the "keypoints" are the trackable points from image 1
                #transliste.append(trans) #Saves the transformation from one frame to another only based on trackable points
                tp1list.append(tp1) # saves the trackable points from frame i-1
                tp2list.append(tp2)
                q1list.append(Q1)
                q2list.append(Q2)
                keypoints = cv2.KeyPoint_convert(tp2)

        formated_data = []
        for i in range(len(tp1list[0])): # save all the frame 0 trackable keypoints, and their frame 1 trackpoints, in seperate arrays
            formated_data.append([[tp1list[0][i][0],tp1list[0][i][1]],[tp2list[0][i][0],tp2list[0][i][1]]])

        for j in range(1,len(tp1list)):
            for i in range(len(formated_data)): # for each of our tracked points
                compare = formated_data[i][j]
                for k in range(len(tp1list[j])):
                    if tp1list[j][k][0] == compare[0] and tp1list[j][k][1] == compare[1]:
                        formated_data[i].append([tp2list[j][k][0],tp2list[j][k][1]])
                        break
                else:
                    formated_data[i].append([None,None])

        return(formated_data,q1[0])


def main():
    data_dir = 'data/00'  # Try KITTI sequence 00
    # data_dir = 'data/00'  # Try KITTI sequence 00
    # data_dir = 'data/07'  # Try KITTI sequence 07
    # data_dir = 'data/KITTI_sequence_1'  # Try KITTI_sequence_2
    frame_limit = 20  # we want to see if we can see a 90deg rotation from frame 100 to 140.
    vo = VisualOdometry(data_dir, frame_limit+1)
    lister = ListBundler()
    remove_duplicates = True
    ###listing = FeatureDetector()
    # play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    global_3d_points = []
    q1_frame_indx = np.array([])
    pose_list = []
    outlier_idx = []
    x_outliers = 0
    y_outliers = 0
    z_outliers = 0
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i == frame_limit + 1:
            break
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            for local3D_p in vo.Q_1:
                if abs(local3D_p[0]) > statistics.median(np.absolute(vo.Q_1[:, 0])) + 4:
                    outlier_idx.append(len(global_3d_points))
                    x_outliers += 1
                elif abs(local3D_p[1]) > statistics.median(np.absolute(vo.Q_1[:, 1])) + 4:
                    outlier_idx.append(len(global_3d_points))
                    y_outliers += 1
                elif local3D_p[2] > statistics.median(vo.Q_1[:, 2]):
                    outlier_idx.append(len(global_3d_points))
                    z_outliers += 1
                if i != 1:  # If we're working with frame 0 for our keypoint gen, then the points are already global, since f=0 is the world pose.
                    homogen_point = np.append([local3D_p[0], local3D_p[1], local3D_p[2]],
                                              1)  # It needs to be x,y,z, we're going to end up with neg y because up for the car is down for the cam.
                    global3D_p = np.matmul(cur_pose, homogen_point)
                    # Try and homogenize propper maybe
                    global3D_p = global3D_p[0:3] / global3D_p[3]  # This might work to make the 3d points propper?
                    global_3d_points.append(global3D_p[:3])
                else:
                    global_3d_points.append(local3D_p)
                q1_frame_indx = np.append(q1_frame_indx, i - 1)

            cur_pose = np.matmul(cur_pose,
                                 transf)  ## We use this function to add the our current place, it takes a 3d position and a transfer function.
        pose_list.append(cur_pose)
        # from here we have the current global pose for i.
        # Here we need a function that makes the current local 3d points globa
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  #
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    # rot100, _ = cv2.Rodrigues(pose_list[100][0:3, 0:3])
    # rot140, _ = cv2.Rodrigues(pose_list[140][0:3, 0:3])

    vo.track_point_gen(0,5)

    global_3d_points = np.array(global_3d_points)
    if remove_duplicates:
        print("Outliers in x: " + str(x_outliers))
        print("Outliers in y: " + str(y_outliers))
        print("Outliers in z: " + str(z_outliers))
        print("Removing Outliers...")
        with tqdm(total=len(outlier_idx) - 1, unit="Outliers") as pbar:
            for i in range(len(outlier_idx) - 1, 0, -1):
                global_3d_points = np.delete(global_3d_points, outlier_idx[i], 0)
                vo.tp_1 = np.delete(vo.tp_1, outlier_idx[i], 0)
                vo.tp_2 = np.delete(vo.tp_2, outlier_idx[i], 0)
                q1_frame_indx = np.delete(q1_frame_indx, outlier_idx[i])
                pbar.update(1)
        print("Outliers Removed")
    # print("Estimated path length, should be equal to number of poses",len(estimated_path))

    # print("q_1 frame index, should be equal to poses minus one",q1_frame_indx[-1])

    # print(global_3d_points)

    for i in range(len(vo.tp_1)):
        lister.append_keypoints(vo.tp_1[i], vo.tp_2[i], global_3d_points[i], q1_frame_indx[i])
    lister.list_sort()

    # new_transformation = vo.test_estimate_new_pose(global_3d_points, q1_frame_indx, pose_list)
    # We for some reason get less parameters than we have total 2D points, so maybe we only get the uniqe 3D points back or something?

    print("hello")

    '''
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i == frame_limit + 1:
            break
        if i < 1:
            cur_pose = gt_pose
            new_poses.append(cur_pose)
        else:
            transf = new_transformation[i-1]
            new_poses.append(transf)
            cur_pose = np.matmul(cur_pose, transf) ## We use this function to add the our current place, it takes a 3d position and a transfer function.
            # from here we have the current global pose for i.
            #Here we need a function that makes the current local 3d points global.
        estimated_better_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    '''
    # plt.plot(global_3d_points)

    temp = np.array([])
    '''
    oldframe = 5
    for i in range(len(q1_frame_indx)):
        if q1_frame_indx[i] != oldframe:
            oldframe = q1_frame_indx[i]
            temp = np.append(temp, global_3d_points[i])

    temp = np.reshape(temp, [-1, 3])
    '''
    oldframe = 5
    for i in range(len(lister.BA_list)):
        if lister.BA_list[i][0] != oldframe:
            oldframe = q1_frame_indx[i]
            temp = np.append(temp, lister.coord_3d_list[i])#global_3d_points[i])

    temp = np.reshape(temp, [-1, 3])

    v = temp
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 2], v[:, 1])
    plt.xlabel("x axies")
    plt.ylabel("y axies")
    plt.clabel("Z axies")
    plt.show()

    pose_list = np.array(pose_list)
    opt_params = run_BA(int(q1_frame_indx[-1] + 2), lister.BA_list, lister.coord_3d_list, pose_list.astype(float))
    # opt params should be in the form: all new 3D points, then cam params for poses, should be in rodreges oriengtation and xyz.

    estimated_better_path = vo.final_new_pose(opt_params, int(q1_frame_indx[-1] + 2) * 9)

    # for i, x in enumerate(vo.tp_1):
    #    if i > 5:
    #        break
    #    print(str(x) + '\t' + str(vo.tp_2[i]) + '\t' + str(q1_frame_indx[i]))
    # print(len(vo.Q_1))
    # print(len(global_3d_points))

    # print(vo.Q_1[len(vo.Q_1)-1])
    # print(global_3d_points[len(global_3d_points)-1])

    ###dist = listing.run_feature_detector(keypoints, descriptors, coords)
    ###if dist < listing.dist_limit:
    ###    run_BA(listing.curr_frame, listing.BA_list, listing.coord_3d_list)

    # plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
    #                         file_out=os.path.basename(data_dir) + ".html")
    '''
    stop_opt = int(q1_frame_indx[-1] + 2) * 9
    print("\n first 3:")
    for i in range(0, stop_opt, 9):
        print(opt_params[i:(i + 3)])
    print("\n Second 3:")
    for i in range(0, stop_opt, 9):
        print(opt_params[i + 3:(i + 6)])
    print("\n Third 3:")
    for i in range(0, stop_opt, 9):
        print(opt_params[i + 6:(i + 9)])
    print("\n Estimated Path vs Estimated Better Path")
    for i in range(len(estimated_path)):
        print(estimated_path[i], estimated_better_path[i])
    '''
    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")

    plotting.visualize_paths(gt_path, estimated_better_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
