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
from mpl_toolkits.mplot3d import Axes3D

class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images_l = self._load_images(os.path.join(data_dir, 'image_l'))
        self.images_r = self._load_images(os.path.join(data_dir, 'image_r'))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        self.tp_1 = np.array([])
        self.tp_2 = np.array([])
        self.Q_1 = np.array([])


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
    def _load_poses(filepath):
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
        return poses

    @staticmethod
    def _load_images(filepath):
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
        img1_l, img2_l = self.images_l[i - 1:i + 1] # for i = 1 then [0:2] i = 1, then keypoint frame, is 0.

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
        self.tp_1 = np.reshape(np.append(self.tp_1,tp1_l),(-1,2))
        self.tp_2 = np.reshape(np.append(self.tp_2,tp2_l),(-1,2))
        #print(Q1)
        self.Q_1 = Q1 #np.reshape(np.append(self.Q_1,Q1),(-1,3))
        #self.Q_2 = np.reshape(np.append(self.Q_2,Q2),(-1,3))
        
        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix
    
    def estimate_new_pose (self, opt_params, q1_frame_index, BA_list, coord_3d_list):
        #n_cams = int(q1_frame_index[-1] - 1)
        n_Qs = int(BA_list[-1][1] + 1) #Number of uniqe 3D points
        n_qs = len(BA_list) # number of 2D points

        frame_idxs = np.empty(n_qs, dtype=int) #Frame index
        Q_idxs = np.empty(n_qs, dtype=int) # What 3D point does belongs to this index
        qs = np.empty((n_qs, 2))

        #cam_idx = []
        #Q_idx = []
        #qs = []
        #Qs = []

        for i in range(len(BA_list)): #For each 2D point
            frame_idx, Q_idx, x, y = BA_list[i]  # Frame, uniqe point number, x and y
            frame_idxs[i] = int(frame_idx)
            Q_idxs[i] = int(Q_idx)
            qs[i] = [float(x), float(y)]

        Qs = np.empty(n_Qs * 3)
        for i in range(n_Qs):
            Qs[i * 3] = coord_3d_list[i][0]
            Qs[i * 3 + 2] = coord_3d_list[i][1]
            Qs[i * 3 + 1] = coord_3d_list[i][2]

        # cam_params = opt_params[:n_cams * 9]
        # cam_params = np.array(cam_params)
        # cam_params = cam_params.reshape((n_cams, -1))

        adjusted_transformations = []
        opt_3Dpoints = opt_params
        opt_3Dpoints = np.array(opt_3Dpoints)
        opt_3Dpoints = opt_3Dpoints.reshape((-1, 3)) #Make the new optimised points into a Q_n*3 matrix
    
        for i in range(int(q1_frame_index[-1])+1): # I think this is for each frame. So tp_1.len() or something. This was +1 for some reason
            tmp_q1 = []
            tmp_q2 = []
            tmp_Q1 = []
            tmp_Q2 = []
            for idx in range(len(q1_frame_index)): # For the number of 2d points.
                if q1_frame_index[idx] == i:  # is the frame'idx = to the i frame
                    tmp_q1.append(self.tp_1[idx])
                    tmp_Q1.append(opt_3Dpoints[Q_idxs[idx]]) # It seems like the 3D points are globally indexed, hopefully they match the order of the 2D points,
            
                if q1_frame_index[idx] == i + 1: # If inx = i plus 1, it logs temp q2
                    tmp_q2.append(self.tp_1[idx])
                    tmp_Q2.append(opt_3Dpoints[Q_idxs[idx]])

                if int(q1_frame_index[-1]) == i:
                    tmp_q2.append(self.tp_2[idx])
                    tmp_Q2.append(opt_3Dpoints[Q_idxs[idx-1]])
        
            if (len(tmp_q1) > len(tmp_q2)):
                tmp_q1 = tmp_q1[:len(tmp_q2)]
                tmp_Q1 = tmp_Q1[:len(tmp_q2)]
            else:
                tmp_q2 = tmp_q2[:len(tmp_q1)]
                tmp_Q2 = tmp_Q2[:len(tmp_q1)]
        
            tmp_q1 = np.array(tmp_q1)
            tmp_q2 = np.array(tmp_q2)
            tmp_Q1 = np.array(tmp_Q1)
            tmp_Q2 = np.array(tmp_Q2)
        
            if (len(tmp_q1) > 1):
                adjusted_transformations.append(self.estimate_pose(tmp_q1, tmp_q2, tmp_Q1, tmp_Q2))
    
        return np.array(adjusted_transformations)
    
    #def plot_transformations ()

    def test_estimate_new_pose(self, opt_params, q1_frame_index, pose_list):
        # n_cams = int(q1_frame_index[-1] - 1)
        n_Qs = len(opt_params)  # Number of uniqe 3D points
        n_qs = len(opt_params)  # number of 2D points


        # cam_params = opt_params[:n_cams * 9]
        # cam_params = np.array(cam_params)
        # cam_params = cam_params.reshape((n_cams, -1))
        adjusted_transformations = []

        for i in range(len(opt_params)):
            local_homo = np.matmul(-pose_list[int(q1_frame_index[i])], [opt_params[i][0],opt_params[i][1],opt_params[i][2],1])
            opt_params[i] = local_homo[:3]/local_homo[3]


        for i in range(int(q1_frame_index[-1]) + 1):  # for each frame, +1 because q1 is a frame short
            tmp_q1 = []
            tmp_q2 = []
            tmp_Q1 = []
            tmp_Q2 = []



            for idx in range(len(q1_frame_index)):  # For the number of 2d points.
                if q1_frame_index[idx] == i:  # is the frame'idx = to the i frame
                    tmp_q1.append(self.tp_1[idx])
                    tmp_Q1.append(opt_params[idx])

                if q1_frame_index[idx] == i + 1:  # If inx = i plus 1, it logs temp q2
                    tmp_q2.append(self.tp_1[idx])
                    tmp_Q2.append(opt_params[idx])

                if int(q1_frame_index[-1]) == i:
                    tmp_q2.append(self.tp_2[idx])
                    tmp_Q2.append(opt_params[idx])

            if (len(tmp_q1) > len(tmp_q2)):
                tmp_q1 = tmp_q1[:len(tmp_q2)]
                tmp_Q1 = tmp_Q1[:len(tmp_q2)]
            else:
                tmp_q2 = tmp_q2[:len(tmp_q1)]
                tmp_Q2 = tmp_Q2[:len(tmp_q1)]

            tmp_q1 = np.array(tmp_q1)
            tmp_q2 = np.array(tmp_q2)
            tmp_Q1 = np.array(tmp_Q1)
            tmp_Q2 = np.array(tmp_Q2)

            if (len(tmp_q1) > 1):
                adjusted_transformations.append(self.estimate_pose(tmp_q1, tmp_q2, tmp_Q1, tmp_Q2))

        return np.array(adjusted_transformations)

def main():
    data_dir = 'data/00_short'  # Try KITTI sequence 00
    # data_dir = 'data/00'  # Try KITTI sequence 00
    # data_dir = 'data/07'  # Try KITTI sequence 07
    # data_dir = 'data/KITTI_sequence_1'  # Try KITTI_sequence_2
    vo = VisualOdometry(data_dir)
    lister = ListBundler()
    frame_limit = 20 # we want to see if we can see a 90deg rotation from frame 100 to 140.
    debug_printer = False
    ###listing = FeatureDetector()
    #play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    estimated_better_path = []
    global_3d_points = []
    new_poses = []
    q1_frame_indx = np.array([])
    pose_list = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i == frame_limit + 1:
            break
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            for local3D_p in vo.Q_1:
                #print(local3D_p)
                homogen_point = np.append([local3D_p[0],local3D_p[2],local3D_p[1]], 1)
                gobal3D_p = np.matmul(cur_pose, homogen_point)
                # Try and homogenize propper maybe
                gobal3D_p = gobal3D_p[0:3]/gobal3D_p[3] # This might work to make the 3d points propper?
                global_3d_points.append(gobal3D_p[:3])
                q1_frame_indx = np.append(q1_frame_indx,i-1)
            cur_pose = np.matmul(cur_pose, transf) ## We use this function to add the our current place, it takes a 3d position and a transfer function.
        pose_list.append(cur_pose)
            # from here we have the current global pose for i.
            #Here we need a function that makes the current local 3d points globa
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3])) #
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))


    #rot100, _ = cv2.Rodrigues(pose_list[100][0:3, 0:3])
    #rot140, _ = cv2.Rodrigues(pose_list[140][0:3, 0:3])


    global_3d_points = np.array(global_3d_points)
    #print("Estimated path length, should be equal to number of poses",len(estimated_path))

    #print("q_1 frame index, should be equal to poses minus one",q1_frame_indx[-1])

    #print(global_3d_points)
    '''
    for i in range(len(vo.tp_1)):
        lister.append_keypoints(vo.tp_1[i], vo.tp_2[i], global_3d_points[i], q1_frame_indx[i])
    lister.list_sort()

    if debug_printer:
        print(len(vo.tp_1))
        print(len(vo.tp_2))
        print(len(global_3d_points))
        print("frames: " + str(len(q1_frame_indx)))
        print(len(lister.BA_list))
        print("BA_list is " + str(len(lister.BA_list)/len(vo.tp_1)) + " times longer than the amount of q's")
        print(len(lister.BA_list)/4)
        print(len(lister.coord_3d_list))
        print("coord_3d_list is " + str(len(lister.coord_3d_list)/len(global_3d_points)) + " times longer than the amount of Q's")
        print(len(lister.coord_3d_list)/2)
        for i, x in enumerate(lister.BA_list):
            if i > 50:
                break
            print(str(x) + '\t' + str(lister.coord_3d_list[i]))

    
    pose_list = np.array(pose_list)
    opt_params = run_BA(int(q1_frame_indx[-1] + 2), lister.BA_list, lister.coord_3d_list, pose_list.astype(float))
    
    new_transformation = vo.estimate_new_pose(opt_params, q1_frame_indx, lister.BA_list, lister.coord_3d_list)
    '''

    new_transformation = vo.test_estimate_new_pose(global_3d_points, q1_frame_indx, pose_list)

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

    #plt.plot(global_3d_points)

    temp = np.array([])

    oldframe = 5
    for i in range(len(q1_frame_indx)):
        if q1_frame_indx[i] != oldframe:
            oldframe = q1_frame_indx[i]
            temp = np.append(temp, global_3d_points[i])

    temp = np.reshape(temp, [-1, 3])

    v = temp
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 2], v[:, 1])
    plt.xlabel("x axies")
    plt.ylabel("y axies")
    plt.clabel("Z axies")
    plt.show()

    #for i, x in enumerate(vo.tp_1):
    #    if i > 5:
    #        break
    #    print(str(x) + '\t' + str(vo.tp_2[i]) + '\t' + str(q1_frame_indx[i]))
    #print(len(vo.Q_1))
    #print(len(global_3d_points))

    #print(vo.Q_1[len(vo.Q_1)-1])
    #print(global_3d_points[len(global_3d_points)-1])


        ###dist = listing.run_feature_detector(keypoints, descriptors, coords)
        ###if dist < listing.dist_limit:
        ###    run_BA(listing.curr_frame, listing.BA_list, listing.coord_3d_list)


    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")

    plotting.visualize_paths(gt_path, estimated_better_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")

if __name__ == "__main__":
    main()
