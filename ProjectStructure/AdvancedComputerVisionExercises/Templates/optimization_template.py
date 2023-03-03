import cv2
import lmfit
import numpy as np

from lib.visualization import camera


class Camera:
    def __init__(self, w, h, f, D, t, r, create_proj=True):
        self.width = w
        self.height = h
        self.focal_length = f
        self.K = np.array([[f, 0, w / 2.0 - 0.5],
                           [0, f, h / 2.0 - 0.5],
                           [0, 0, 1]], dtype=np.float64)
        self.t = np.array(t, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)
        self.D = np.array(D, dtype=np.float64)
        if create_proj:
            self.P = self.create_proj_mat(self.K, self.t, self.r)

    @staticmethod
    def create_proj_mat(K, t, r):
        KA = np.concatenate((K, np.zeros((3, 1))), axis=1)
        R, _ = cv2.Rodrigues(r)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return np.matmul(KA, T)


class StereoCameras:
    def __init__(self, w, h, f, D, t_r, r_r, t_l=(0, 0, 0), r_l=(0, 0, 0)):
        self.cam_l = Camera(w, h, f, D, t_l, r_l)
        self.cam_r = Camera(w, h, f, D, t_r, r_r)

    def plot_cams(self, ponts3d=None):
        camera.plot_cams([self.cam_l.t, self.cam_r.t], [self.cam_l.r, self.cam_r.r], ponts3d=ponts3d)


def objective_func(params, q_l, q_r, P_l, P_r):
    """
    Calculates the residuals

    Parameters
    ----------
    params (Parameter): The parameters given from lmfit
    q_l (ndarray): The left image point
    q_r (ndarray): The right image point
    P_l (ndarray): Projection matrix left
    P_r (ndarray): Projection matrix right

    Returns
    -------
    residual (list): The residuals
    """
    # Create the homogenized 3D point
    hom_Q = np.transpose(np.array([[params['x'].value, params['y'].value, params['z'].value, 1]], dtype=np.float64))

    # Project point
    hom_proj_q_l = np.matmul(P_l, hom_Q)
    uhom_proj_q_l = hom_proj_q_l[:2] / hom_proj_q_l[2]
    hom_proj_q_r = np.matmul(P_r, hom_Q)
    uhom_proj_q_r = hom_proj_q_r[:2] / hom_proj_q_r[2]

    # Calculate the residuals
    residual = []
    residual.append(q_l[0] - uhom_proj_q_l[0])
    residual.append(q_l[1] - uhom_proj_q_l[1])
    residual.append(q_r[0] - uhom_proj_q_r[0])
    residual.append(q_r[1] - uhom_proj_q_r[1])
    return residual


def define_stereo_cameras():
    """
    Defines the stereo cameras

    Returns
    -------
    cams (StereoCameras): The two cameras
    """
    # In this function you are going to describe the cameras
    # w, h, f = width, height, focal length
    # D = tuple of 5 distortion parameters
    # t_r, r_r = XYZ translatation and RPY rotation of right camera respective to left
    pass


def generate_points(x_range, y_range, z_range, n):
    """
    Generates 3D points in the given range

    Parameters
    ----------
    x_range (tuple): Tuple with the low and high border of the x range
    y_range (tuple): Tuple with the low and high border of the y range
    z_range (tuple): Tuple with the low and high border of the z range
    n (int): The number of random points

    Returns
    -------
    Qs (ndarray): Numpy array with the random points in the shape of (n, 3)
    """
    # In this function you have to generate some (n) random points within the given x, y and z range
    # The return should be a numpy array
    pass


def project_points_to_image_plane(Qs, cam):
    """
    Projects the 3D points to the image plane

    Parameters
    ----------
    Qs (ndarray): The points to project. Given in shape (n, 3)
    cam (Camera): Camera object there contains the camera parameters

    Returns
    -------
    qs (ndarray): The points in the image plane. In shape (2, n)
    """
    # In this function you have to project the points to the image plane
    # A good idea would be to use the cv2.projectPoints to project the 3D points to 2D
    # After this you also need to correct the shape of the output (squeeze and transpose)
    pass


def add_noise_to_image_points(qs, sigma):
    """
    Adds Gaussian noise to the image points

    Parameters
    ----------
    qs (ndarray): The points in the image plane. Given in shape (2, n)
    sigma (float): The standard deviation for the Gaussian noise

    Returns
    -------
    qs_noisy (ndarray): The points in the image plane with noise. Shape (2, n)
    """
    # In this function you are going to add some noise to the image points
    # The noise should be Gaussian noise
    pass


def triangulate_points(qs_l, qs_r, cams):
    """
    triangulates the 3D points form the feature points

    Parameters
    ----------
    qs_l (ndarray): The feature points for the left camera. Given in shape (2, n)
    qs_r (ndarray): The feature points for the right camera. Given in shape (2, n)
    cams (StereoCameras): The two cameras

    Returns
    -------
    uhom_Qs (ndarray): The un-homogenized 3D points. The shape should be (n, 3)
    """
    # Here you need to triangulate the points to find the 3D points.
    # A good idea would be to use the cv2.triangulatePoints
    # Remember to un-homogenize the points and correct the shape
    pass


def evaluate_reprojection_error(Qs, qs_l, qs_r, cams):
    """
    Calculates the reprojection error

    Parameters
    ----------
    Qs (ndarray): The triangulated 3D points. Given i shape (n, 3)
    qs_l (ndarray): The left image points (those there was used to triangulate the 3D points). Given in shape (2, n)
    qs_r (ndarray): The right image points (those there was used to triangulate the 3D points). Given in shape (2, n)
    cams (StereoCameras): The two cameras

    Returns
    -------
    rpe (float): The sum of the reprojection error for the left end right image plane
    """
    # In this function you need to project the points to 2D and calculate the rms
    # distance between the optimal and projected points
    pass


def optimize_points(Qs, qs_l, qs_r, cams):
    """
    Optimizes the points

    Parameters
    ----------
    Qs (ndarray): Triangulated 3D points. Given in shape (n, 3)
    qs_l (ndarray): The left image points (those there was used to triangulate the 3D points). Given in shape (2, n)
    qs_r (ndarray): The right image points (those there was used to triangulate the 3D points). Given in shape (2, n)
    cams (StereoCameras): The two cameras

    Returns
    -------
    Qs_opt (ndarray): Optimized 3D points. In shape (n, 3)
    """
    # In this function you are going to optimize the points
    # Use lmfit with the provided objective function to find a point minimizing the reprojection error
    Qs_opt = []
    for q_l, q_r, Q_init in zip(qs_l, qs_r, Qs):
        # Write your code here :)
        pass
    return np.array(Qs_opt, dtype=np.float64)


def main():
    # Create the cameras
    cams = define_stereo_cameras()
    # Generate the points
    orig_Qs = generate_points((0.05, 0.1), (0.18, 0.22), (1.98, 2.02), 5)
    # Plot the cameras with the points
    cams.plot_cams(orig_Qs)

    # Project the points to the image
    qs_l = project_points_to_image_plane(orig_Qs, cams.cam_l)
    qs_r = project_points_to_image_plane(orig_Qs, cams.cam_r)

    # Add noise to teh image points
    noisy_qs_l = add_noise_to_image_points(qs_l, 1)
    noisy_qs_r = add_noise_to_image_points(qs_r, 1)

    # Triangulate the points
    triangulated_Qs = triangulate_points(qs_l, qs_r, cams)
    noisy_triangulated_Qs = triangulate_points(noisy_qs_l, noisy_qs_r, cams)

    # Calculate the reprojection error
    noisy_rpe = evaluate_reprojection_error(noisy_triangulated_Qs, noisy_qs_l, noisy_qs_r, cams)

    # Run the optimization
    optimal_Qs = optimize_points(noisy_triangulated_Qs, np.transpose(noisy_qs_l), np.transpose(noisy_qs_r), cams)
    # Calculate the reprojection error for the optimized points
    optimal_rpe = evaluate_reprojection_error(optimal_Qs, noisy_qs_l, noisy_qs_r, cams)

    print('Triangulated noisy points reprojection error: {}'.format(noisy_rpe))
    print('Optimized noisy points reprojection error: {}'.format(optimal_rpe))
    print('Improvement: {}'.format(noisy_rpe - optimal_rpe))


if __name__ == "__main__":
    main()
