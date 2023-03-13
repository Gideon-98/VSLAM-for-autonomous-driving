import bz2

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from lib.visualization.plotting import plot_residual_results, plot_sparsity

path = "ProjectStructure/AdvancedComputerVisionExercises/data/00_short/image_l"


def read_bal_data(path):
	"""
	The `read_bal_data()` function takes a file path as input and returns several arrays containing initial estimates
	of camera and point parameters for a 3D reconstruction problem.
	
	Loads the data

	Parameters
	----------
	file_name (str): The file path for the data

	Returns
	-------
	cam_params (ndarray):
		Shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
		First 3 components in each row form a rotation vector (
		https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector,
		then a focal distance and two distortion parameters.
	Qs (ndarray):
		Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
	cam_idxs (ndarray):
		Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
	Q_idxs (ndarray):
		Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
	qs (ndarray):
		Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
	"""
	with bz2.open(path, "rt") as file:
		n_cams, n_Qs, n_qs = map(int, file.readline().split())
		
		cam_idxs = np.empty(n_qs, dtype=int)
		Q_idxs = np.empty(n_qs, dtype=int)
		qs = np.empty((n_qs, 2))
		
		"""
		loop used to read each line of the file containing an observation. The camera index, point index,
		and 2D coordinates are parsed from the line and stored in the corresponding arrays
		"""
		for i in range(n_qs):
			cam_idx, Q_idx, x, y = file.readline().split()  # the number of cameras, points, and observations
			cam_idxs[i] = int(cam_idx)
			Q_idxs[i] = int(Q_idx)
			qs[i] = [float(x), float(y)]
		"""
		loop used to read the camera parameters from the file. There are nine parameters for each camera:
			three for rotation (in the form of a rotation vector)
			three for translation
			three for intrinsic camera parameters (focal length and two distortion parameters)
		"""
		cam_params = np.empty(n_cams * 9)
		for i in range(n_cams * 9):
			cam_params[i] = float(file.readline())
		cam_params = cam_params.reshape((n_cams, -1))
		"""
		loop used to read the point coordinates from the file. There are three coordinates (x, y, z) for each point in
		the world frame
		"""
		Qs = np.empty(n_Qs * 3)
		for i in range(n_Qs * 3):
			Qs[i] = float(file.readline())
		Qs = Qs.reshape((n_Qs, -1))
	
	"""
	function returns the camera parameters, point coordinates, camera indices, point indices, and measured 2D
	coordinates as NumPy arrays
	"""
	return cam_params, Qs, cam_idxs, Q_idxs, qs


def reindex(idxs):
	"""
	The function first finds the unique values in `idxs` and sorts them in ascending order using `np.sort()`.
	"""
	keys = np.sort(np.unique(idxs))
	"""
	It then creates a dictionary `key_dict` that maps each unique value in `keys` to its corresponding index in the
	sorted `keys` array (i.e., a consecutive integer starting from 0). This is done using a dictionary comprehension
	that iterates over `keys` and the corresponding index values returned by `range(keys.shape[0])`.
	"""
	key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
	"""
	Finally, the function uses a list comprehension to map each value in `idxs` to its corresponding index in the
	sorted `keys` array using `key_dict`, and returns the resulting list. This effectively reindexes the values in
	`idxs` to be consecutive integers starting from 0, with the order of the values preserved.
	"""
	return [key_dict[idx] for idx in idxs]


def shrink_problem(n, cam_params, Qs, cam_idxs, Q_idxs, qs):
	"""
	Shrinks the problem to be n points

	Parameters
	----------
	n (int): Number of points the shrink problem should be
	cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3
	components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
	next 3 components form a translation vector, then a focal distance and two distortion parameters.
	Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
	cam_idxs (ndarray): Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each
	observation.
	Q_idxs (ndarray): Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each
	observation.
	qs (ndarray): Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each
	observations.

	Returns
	-------
	cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3
	components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
	next 3 components form a translation vector, then a focal distance and two distortion parameters.
	Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
	cam_idxs (ndarray): Shape (n,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
	Q_idxs (ndarray): Shape (n,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
	qs (ndarray): Shape (n, 2) contains measured 2-D coordinates of points projected on images in each observations.
	"""
	"""
	This function takes as input initial estimates of camera parameters, point coordinates, and observations,
	and returns a shrunk version of the problem with a specified number of points. Specifically, it reduces the number
	of observations to n, keeps only the cameras and points involved in those observations, and reindexes the
	remaining observations and points.
	
	The `cam_idxs`, `Q_idxs`, and `qs` arrays are updated to contain only the indices relevant to the reduced problem,
	and the `cam_idxs` and `Q_idxs` arrays are reindexed so that they start from 0 and increase sequentially. The
	`cam_params` and `Qs` arrays are updated to include only the cameras and points involved in the reduced problem.
	The function assumes that `n` is less than or equal to the number of observations in the original problem.

	"""
	cam_idxs = cam_idxs[:n]
	Q_idxs = Q_idxs[:n]
	qs = qs[:n]
	cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
	Qs = Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]
	
	cam_idxs = reindex(cam_idxs)
	Q_idxs = reindex(Q_idxs)
	
	"""
	:returns
	-   cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3
	components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
	next 3 components form a translation vector, then a focal distance and two distortion parameters.
	-   Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
	-   cam_idxs (ndarray): Shape (n,) contains indices of cameras (from 0 to n_cameras - 1) involved in each
	observation.
	-   Q_idxs (ndarray): Shape (n,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
	-   qs (ndarray): Shape (n, 2) contains measured 2-D coordinates of points projected on images in each
	observations.
	"""
	return cam_params, Qs, cam_idxs, Q_idxs, qs

def rotate(Qs, rot_vecs):
	"""
	Rotate points by given rotation vectors.
	Rodrigues' rotation formula is used.

	Parameters
	----------
	Qs (ndarray): The 3D points
	rot_vecs (ndarray): The rotation vectors

	Returns
	-------
	Qs_rot (ndarray): The rotated 3D points
	"""
	"""
	-   `Qs`: a numpy array of shape `(n_points, 3)` representing the 3D points to be rotated.
	-   `rot_vecs`: a numpy array of shape `(n_points, 3)` representing the rotation vectors for each 3D point.
	"""
	
	theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]  # calculates the norm of each rotation vector and stores
	# it in `theta`.
	"""
	It then normalizes the rotation vectors by dividing them by `theta`, and replaces any NaN values with 0 using `np.nan_to_num`.
	"""
	with np.errstate(invalid='ignore'):
		v = rot_vecs / theta
		v = np.nan_to_num(v)
	"""
	Computes the dot product between each 3D point and its corresponding normalized rotation vector, and stores the result in `dot`. It also calculates the cosine and sine of `theta` using `np.cos` and `np.sin`, respectively.
	"""
	dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	"""
	Returns the rotated 3D points using Rodrigues' rotation formula.
	Specifically, it computes the rotated points by adding three terms:
	1.  `cos_theta * Qs`: a term that scales the original points by the cosine of `theta`.
	2.  `sin_theta * np.cross(v, Qs)`: a term that represents the cross product of the normalized rotation vectors and the original points, scaled by the sine of `theta`.
	3.  `dot * (1 - cos_theta) * v`: a term that scales the normalized rotation vectors by `(1 - cos_theta)` and then scales them again by `dot`, which represents the dot product between the original points and the normalized rotation vectors.
	
	The resulting numpy array `Qs_rot` has the same shape as the input `Qs` and contains the rotated 3D points.
	"""
	Qs_rot = cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v
	return Qs_rot


def project(Qs, cam_params):
	"""
	Convert 3-D points to 2-D by projecting onto images.

	Parameters
	----------
	Qs (ndarray): The 3D points
	cam_params (ndarray): Initial parameters for cameras

	Returns
	-------
	qs_proj (ndarray): The projectet 2D points
	"""
	# Rotate the points using the camera rotation vectors `cam_params[:, :3]`
	qs_proj = rotate(Qs, cam_params[:, :3])
	# Translate the points by the camera translation vectors `cam_params[:, 3:6]`
	qs_proj += cam_params[:, 3:6]
	# Un-homogenized the points by dividing the first two coordinates by the third coordinate
	qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
	# Distortion applied to the un-homogenized points using the camera's focal
	f, k1, k2 = cam_params[:, 6:].T #f=length ,k1,k2= dist. parameters
	n = np.sum(qs_proj ** 2, axis=1)
	r = 1 + k1 * n + k2 * n ** 2
	qs_proj *= (r * f)[:, np.newaxis]
	
	"""
	The `rotate` function takes in an array of rotation vectors `rot_vecs` and applies the rotation to an array of vectors `Qs`.
	"""
	return qs_proj


def objective(params, n_cams, n_Qs, cam_idxs, Q_idxs, qs):
	"""
	The objective function for the bundle adjustment

	Parameters
	----------
	params (ndarray): Camera parameters and 3-D coordinates.
	n_cams (int): Number of cameras
	n_Qs (int): Number of points
	cam_idxs (list): Indices of cameras for image points
	Q_idxs (list): Indices of 3D points for image points
	qs (ndarray): The image points

	Returns
	-------
	residuals (ndarray): The residuals between the observations and the reprojected points
	"""
	# Should return the residuals consisting of the difference between the observations qs and the reporjected points
	# Params is passed from bundle_adjustment() and contains the camera parameters and 3D points
	# project() expects an arrays of shape (len(qs), 3) indexed using Q_idxs and (len(qs), 9) indexed using cam_idxs
	# Copy the elements of the camera parameters and 3D points based on cam_idxs and Q_idxs
	
	# Extracts the camera parameters
	cam_params = params[:n_cams * 9].reshape((n_cams, 9))
	
	# Extracts the 3D points
	Qs = params[n_cams * 9:].reshape((n_Qs, 3))
	
	# Project the 3D points into the image planes
	qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs]) #resulting projecting points
	
	# Calculate the residuals
	residuals = (qs_proj - qs).ravel()
	"""
	taking the difference between the observations (`qs`) and the reprojected points (`qs_proj`), and the resulting array is flattened using `ravel()` to produce a 1D array.
	"""
	return residuals


def bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs):
	"""
	Preforms bundle adjustment stacking the input camera parameters and 3D points to create a one-dimensional array of parameters and uses least_squares from scipy.optimize to minimize the objective function. It returns the initial residuals, the residuals at the solution, and the solution.

	Parameters
	----------
	cam_params (ndarray): Initial parameters for cameras
	Qs (ndarray): The 3D points
	cam_idxs (list): Indices of cameras for image points
	Q_idxs (list): Indices of 3D points for image points
	qs (ndarray): The image points

	Returns
	-------
	residual_init (ndarray): Initial residuals
	residuals_solu (ndarray): Residuals at the solution
	solu (ndarray): Solution
	"""
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
	res = least_squares(objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))
	
	# Get the residuals at the solution and the solution
	residuals_solu = res.fun
	solu = res.x
	return residual_init, residuals_solu, solu


def sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs):
	"""
	Create the sparsity matrix

	Parameters
	----------
	n_cams (int): Number of cameras
	n_Qs (int): Number of 3d points
	cam_idxs (list): Indices of cameras for image points
	Q_idxs (list): Indices of 3D points for image points

	Returns
	-------
	sparse_mat (ndarray): The sparsity matrix
	"""
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
		sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
		sparse_mat[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1
	
	return sparse_mat


def bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
	"""
	Preforms bundle adjustment with sparsity

	Parameters
	----------
	cam_params (ndarray): Initial parameters for cameras
	Qs (ndarray): The 3D points
	cam_idxs (list): Indices of cameras for image points
	Q_idxs (list): Indices of 3D points for image points
	qs (ndarray): The image points
	sparse_mat (ndarray): The sparsity matrix

	Returns
	-------
	residual_init (ndarray): Initial residuals
	residuals_solu (ndarray): Residuals at the solution
	solu (ndarray): Solution
	"""
	
	# Stack the camera parameters and the 3D points
	params = np.hstack((cam_params.ravel(), Qs.ravel()))
	
	# Save the initial residuals
	residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)
	
	# Perform the least_squares optimization with sparsity
	res = least_squares(objective, params, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
	                    args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))
	
	# Get the residuals at the solution and the solution
	residuals_solu = res.fun
	solu = res.x
	return residual_init, residuals_solu, solu


def main():
	data_file = "data/problem-49-7776-pre/problem-49-7776-pre.txt.bz2"
	cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data(data_file)
	cam_params_small, Qs_small, cam_idxs_small, Q_idxs_small, qs_small = shrink_problem(1000, cam_params, Qs, cam_idxs,Q_idxs, qs)
	"""
	We need these parameters to perform bundle adjustement, how do we want to obtain them?
	cam_params: an array with the initial parameters for each camera.
	Qs: an array with the coordinates of the 3D points in the world.
	cam_idxs: a list with the indices of the cameras for each image point.
	Q_idxs: a list with the indices of the 3D points for each image point.
	qs: an array with the coordinates of the image points in the cameras.
	"""
	
	n_cams_small = cam_params_small.shape[0]
	n_Qs_small = Qs_small.shape[0]
	print("n_cameras: {}".format(n_cams_small))
	print("n_points: {}".format(n_Qs_small))
	print("Total number of parameters: {}".format(9 * n_cams_small + 3 * n_Qs_small))
	print("Total number of residuals: {}".format(2 * qs_small.shape[0]))
	
	small_residual_init, small_residual_minimized, opt_params = bundle_adjustment(cam_params_small, Qs_small,
	                                                                              cam_idxs_small,
	                                                                              Q_idxs_small, qs_small)
	
	n_cams = cam_params.shape[0]
	n_Qs = Qs.shape[0]
	print("n_cameras: {}".format(n_cams))
	print("n_points: {}".format(n_Qs))
	print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
	print("Total number of residuals: {}".format(2 * qs.shape[0]))
	
	# residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
	sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
	plot_sparsity(sparse_mat)
	residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs, cam_idxs, Q_idxs,
	                                                                                qs, sparse_mat)
	
	plot_residual_results(qs_small, small_residual_init, small_residual_minimized, qs, residual_init,
	                      residual_minimized)


if __name__ == "__main__":
	main()
