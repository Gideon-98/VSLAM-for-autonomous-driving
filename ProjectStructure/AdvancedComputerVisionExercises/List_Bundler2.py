import numpy as np

from Visual_SLAM_Solution import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# BoW code inspired by https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb


class ListBundler:

    def __init__(self, n_clusters=200, n_features=500, future_iterations=6):
        self.futuretrack = future_iterations
        self.detector = cv2.ORB_create()
        self.nclusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.nclusters, verbose=0)
        self.hist_list = []
        self.dist_limit = 0.1
        self.keypoint_list = []
        self.descriptor_list = []
        self.temp_3d_list = []
        self.BA_list = []
        self.coord_3d_list = []
        self.curr_frame = -1
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def trainer(self, img_list):
        n_imgs = len(img_list)
        descriptor_list = []
        for img_raw in img_list:
            img = cv2.imread(img_raw)
            _, descriptors = self.orb.detectAndCompute(img, None)
            if descriptors is not None:
                descriptor_list.append(img, descriptors)

        des = descriptor_list[0][1]
        for img, descriptors in descriptor_list:
            des = np.vstack(des, descriptors)

        des_float = des.astype(float)
        voc, variance = self.kmeans(des_float, self.nclusters, 1)
        image_features = np.zeros((n_imgs, self.nclusters), "float32")
        for i in range(n_imgs):
            words, distance = vq(descriptor_list[i][1], voc)

    def append_features(self, matched, xyz, descriptors):
        self.curr_frame += 1
        # Eliminate all keypoints that don't track for at least 1 frame
        tracked_matched = []
        tracked_xyz = []
        tracked_features = []
        for i in range(len(matched)):
            if len(matched[i]) > 1:
                tracked_matched.append(matched[i])
                tracked_xyz.append(xyz[i])
                tracked_features.append(descriptors[i])
        self.keypoint_list.append(tracked_matched)
        self.temp_3d_list.append(tracked_xyz)
        self.descriptor_list.append(tracked_features)

    def append_histogram(self, features):
        self.descriptor_list.append(features)
        if self.keypoint_list[1] is not None:
            histogram = KMeans.build_histogram(features, self.kmeans)
            self.hist_list.append(histogram)

    def find_nearest_neighbor(self, curr_histogram):
        neighbor = NearestNeighbors(n_neighbors=1)
        neighbor.fit(self.hist_list)
        dist, result = neighbor.kneighbors([curr_histogram])

        return dist, result

    def compare_keypoints_loop(self, target):
        # Find keypoints in target frame
        preindex = []
        for i in range(len(self.BA_list)):
            if self.BA_list[i][1] == target:
                preindex.append(i)

        # Search for keypoint label to save computing power
        labels = []
        for i in range(len(self.keypoint_list[target])):
            for j in range(len(preindex)):
                x_correct = (self.keypoint_list[target][i][0][0] == self.BA_list[preindex[j]][2])
                y_correct = (self.keypoint_list[target][i][0][0] == self.BA_list[preindex[j]][3])
                if x_correct and y_correct:
                    labels.append(self.BA_list[target][j][0])

        # match descriptors of image 1 and image 2 keypoints
        matches = self.bf.match([i[0] for i in self.descriptor_list[target]],
                                [i[0] for i in self.descriptor_list[self.curr_frame]])
        # create empty lists to store matched keypoints and labels
        matched_kp = []
        matched_labels = []
        matched_3d = []
        for match in matches:
            # extract index of matched keypoint in image 2
            img2_idx = match.trainIdx

            # extract coordinates of matched keypoints in image 1 and image 2
            kp = self.keypoint_list[self.curr_frame][match.queryIdx][0].pt
            coord = self.temp_3d_list[target][match.queryIdx]

            # append matched keypoints and labels to lists
            matched_kp.append(kp)
            matched_labels.append(labels[img2_idx])
            matched_3d.append(coord)

        # create a list to store unmatched keypoints
        unmatched_kp = []
        unmatched_3D = []

        # loop over all keypoints in image 2 and check if each one is in matched keypoints
        for i in self.descriptor_list[self.curr_frame]:
            if i not in [match.trainIdx for match in matches]:
                # if not matched, append the keypoint coordinates to the list of unmatched keypoints
                point = self.keypoint_list[self.curr_frame][0][i].pt
                matched_point = [self.BA_list[-1][0], self.curr_frame, point[0], point[1]]
                self.BA_list.append(matched_point)
                self.coord_3d_list.append(self.temp_3d_list[self.curr_frame][i])

        # return list of matched keypoints and labels
        return matched_kp, matched_labels, matched_3d, unmatched_kp, unmatched_3D

    def append_keypoints_loop(self, matched_kp, matched_labels, matched_3d):
        # Create each point in the form the BA likes
        matched_point_list = []
        for i in range(len(matched_kp)):
            matched_point_list[i] = [matched_labels[i], self.curr_frame, matched_kp[i][0], matched_kp[i][1]]
        # Sort by first index
        # matched_point_list = sorted(matched_point_list, key=lambda x: x[0])

        # Append matched points
        if self.BA_list:
            for i in range(len(matched_kp)):
                for j in range(len(self.BA_list)):
                    current_match_point = (self.BA_list[j][0] == matched_labels[i])
                    next_is_false = (self.BA_list[j + 1][0] != matched_labels[i])
                    if current_match_point and next_is_false:
                        self.BA_list.insert(j + 1, matched_point_list[i])
                        self.coord_3d_list.insert(j + 1, matched_3d[i])

    def append_keypoints_no_loop(self):
        lastpoint = self.BA_list[-1][0] + 1
        for i in range(len(self.keypoint_list[self.curr_frame])):
            for j in range(len(self.keypoint_list[self.curr_frame][i])):
                point = self.keypoint_list[self.curr_frame][i][j].pt
                to_add = [(lastpoint + i), (self.curr_frame + j), point[0], point[1]]
                self.BA_list.append(to_add)
                self.coord_3d_list.append(self.temp_3d_list[self.curr_frame][i])

    def run_feature_detector(self, matched, xyz, descriptors):
        self.append_features(matched, xyz, descriptors)
        self.append_histogram(descriptors)
        dist, target = self.find_nearest_neighbor(self.hist_list[-1])
        if dist < self.dist_limit and self.curr_frame > 1:
            self.append_keypoints_loop(self.compare_keypoints_loop(target))
        else:
            self.append_keypoints_tracked_no_loop()
        return dist
