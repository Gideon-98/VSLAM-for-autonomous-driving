import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from lib.visualization.plotting import plot_residual_results, plot_sparsity

path = "ProjectStructure/AdvancedComputerVisionExercises/data/00_short/image_l"


def read_VSLAM_data(frame_count, BA_list, coord_3D_list, dof):
    n_cams = int(frame_count)
    n_Qs = int(BA_list[-1][1] + 1) #Why plus 1? oh because it is zero indexed
    n_qs = len(BA_list)

    cam_idxs = np.empty(n_qs, dtype=int)
    Q_idxs = np.empty(n_qs, dtype=int)
    qs = np.empty((n_qs, 2))

    for i in range(n_qs):
        cam_idx, Q_idx, x, y = BA_list[i]  # the number of cameras, points, and observations
        cam_idxs[i] = int(cam_idx)
        Q_idxs[i] = int(Q_idx)
        qs[i] = [float(x), float(y)]

    cam_params = np.empty(n_cams * 9)
    for i in range(0, (n_cams * 9), 9):
        rot, _ = cv2.Rodrigues(dof[int(i / 9)][0:3, 0:3])
        rot = rot.flatten()
        cam_params[i] = rot[0]
        cam_params[i + 1] = rot[1]
        cam_params[i + 2] = rot[2]
        cam_params[i + 3] = dof[int(i / 9)][0][3]
        cam_params[i + 4] = dof[int(i / 9)][1][3]
        cam_params[i + 5] = dof[int(i / 9)][2][3]
        cam_params[i + 6] = 718.856 #645.24
        cam_params[i + 7] = 0#607.1928
        cam_params[i + 8] = 0#185.2157
        #Why set number for 6,7 and 8?

    print("cam_params Length: {}".format(len(cam_params)))
    cam_params = cam_params.reshape((n_cams, -1))
    print("cam_params Length: {}".format(len(cam_params)))

    Qs = np.empty(n_Qs * 3)
    for i in range(n_Qs):
        Qs[i * 3] = coord_3D_list[i][0]
        Qs[i * 3 + 1] = coord_3D_list[i][1]
        Qs[i * 3 + 2] = coord_3D_list[i][2]
    Qs = Qs.reshape((n_Qs, -1))

    # print(cam_params)
    # print(Qs)
    print("Qs: " + str(len(Qs)))
    print("Q_idxs: " + str(len(Q_idxs)))
    print("cam_params: " + str(len(cam_params)))
    print("cam_idxs: " + str(len(cam_idxs)))
    print("n_cam: " + str(cam_idxs[-1]))

    return cam_params, Qs, cam_idxs, Q_idxs, qs


def reindex(idxs):
    keys = np.sort(np.unique(idxs))
    key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}

    return [key_dict[idx] for idx in idxs]


def shrink_problem(n, cam_params, Qs, cam_idxs, Q_idxs, qs):
    cam_idxs = cam_idxs[:n]
    Q_idxs = Q_idxs[:n]
    qs = qs[:n]
    cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
    Qs = Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]

    cam_idxs = reindex(cam_idxs)
    Q_idxs = reindex(Q_idxs)

    return cam_params, Qs, cam_idxs, Q_idxs, qs


def rotate(Qs, rot_vecs):

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]  # calculates the norm of each rotation vector and stores
    # it in `theta`.

    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
        #v[np.isnan(v)] = 0

    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    Qs_rot = cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v
    return Qs_rot


def project(Qs, cam_params):

    # Rotate the points using the camera rotation vectors `cam_params[:, :3]`
    qs_proj = rotate(Qs, cam_params[:, :3])
    # Translate the points by the camera translation vectors `cam_params[:, 3:6]`
    qs_proj += cam_params[:, 3:6]
    # Un-homogenized the points by dividing the first two coordinates by the third coordinate # repojected
    qs_proj = qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
    # Distortion applied to the un-homogenized points using the camera's focal
    f, k1, k2 = cam_params[:, 6:].T  # f=length ,k1,k2= dist. parameters
    n = np.sum(qs_proj ** 2, axis=1)  # x²+y^2
    r = 1 + k1 * n + k2 * n ** 2 # 97
    qs_proj *= (r * f)[:, np.newaxis]
    # We are missing the principle point in our reprojection
    #OpenCV2 docs for reprojection
    #Try use openCV project instead of this
    return qs_proj


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs):


    # Should return the residuals consisting of the difference between the observations qs and the reporjected points
    # Params is passed from bundle_adjustment() and contains the camera parameters and 3D points
    # project() expects an arrays of shape (len(qs), 3) indexed using Q_idxs and (len(qs), 9) indexed using cam_idxs
    # Copy the elements of the camera parameters and 3D points based on cam_idxs and Q_idxs

    # Extracts the camera parameters
    cam_params = params[:n_cams * 9].reshape((n_cams, 9))

    # Extracts the 3D points
    Qs = params[n_cams * 9:].reshape((n_Qs, 3))

    # Project the 3D points into the image planes
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])  # resulting projecting points


    #print("Qs Length: {}".format(len(Qs)))
    #print("Q_idxs Length: {}".format(len(Q_idxs)))
    #print("cam_params Length: {}".format(len(cam_params)))
    #print("cam_idxs Length: {}".format(len(cam_idxs)))

    # Calculate the residuals
    residuals = (qs_proj - qs).ravel()

    return residuals


def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs):

    # Use least_squares() from scipy.optimize to minimize the objective function
    # Stack cam_params and Qs after using ravel() on them to create a one dimensional array of the parameters
    # save the initial residuals by manually calling the objective function
    # residual_init = objective()
    # res = least_squares(.....)

    # Stack the camera parameters and the 3D points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))

    # Save the initial residuals
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

    # Perform the least_squares optimization
    res = least_squares(objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=50,
                        args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    return residual_init, residuals_solu, solu


def sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs):

    m = cam_idxs.size * 2  # number of residuals
    n = n_cams * 9 + n_Qs * 3  # number of parameters
    sparse_mat = lil_matrix((m, n), dtype=int)
    # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

    i = np.arange(cam_idxs.size)
    # Sparsity from camera parameters
    for s in range(9):
        sparse_mat[2 * i, cam_idxs * 9 + s] = 1
        sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1

    # Sparsity from 3D points
    for s in range(3):
        print(s)
        sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
        sparse_mat[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1

    return sparse_mat

def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
    # Stack the camera parameters and the 3D points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))

    # Save the initial residuals
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

    # Perform the least_squares optimization with sparsity
    res = least_squares(objective, params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        max_nfev=50, args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

    # Get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    return residual_init, residuals_solu, solu


def run_BA(frame_count, BA_list, coord_3D_list, dof):
    cam_params, Qs, cam_idxs, Q_idxs, qs = read_VSLAM_data(frame_count, BA_list, coord_3D_list, dof)
    #cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small = shrink_problem(1000, cam_params, Qs, cam_idxs,
    #                                                                                    Q_idxs, qs)
    '''
    n_cams_small = cam_params_small.shape[0]
    n_Qs_small = Qs_small.shape[0]
    print("n_cameras: {}".format(n_cams_small))
    print("n_points: {}".format(n_Qs_small))
    print("Total number of parameters: {}".format(9 * n_cams_small + 3 * n_Qs_small))
    print("Total number of residuals: {}".format(2 * qs_small.shape[0]))
    small_residual_init, small_residual_minimized, opt_params = bundle_adjustment(cam_params_small, Qs_small,
                                                                                  cam_idxs_small,
                        hallo                                                            Q_idxs_small, qs_small)
    '''
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))
    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    print("Sparse Mat Done")
    #plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
                                                                                    qs, sparse_mat)
    #We should get "about 1 pixel erroe in residual"
    print("BA Done")
    #plot_residual_results(qs_small, small_residual_init, small_residual_minimized, qs, residual_init, residual_minimized)

    #print("Hello")
    return opt_params


if __name__ == "__main__":
    main()
