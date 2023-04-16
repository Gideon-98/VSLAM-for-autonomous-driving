import numpy as np

from Visual_SLAM_Solution import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# BoW code inspired by https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb


class ListBundler:

    def __init__(self):
        self.detector = cv2.ORB_create()
        self.kmeans = KMeans(n_clusters=800)
        self.hist_list = []
        self.dist_limit = 0.1
        self.keypoint_list = []
        self.BA_list = []
        self.coord_3d_list = []
        self.curr_frame = -1
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create()

    def append_features(self, match, no_match):
        self.curr_frame += 1
        features = np.concatenate(match, no_match)
        self.keypoint_list.append([features[0], features[1], features[2]])

    def append_histogram(self):
        # keypoint, descriptor = self.features(image, self.detector)
        if self.keypoint_list[1] is not None:
            histogram = KMeans.build_histogram(self.keypoint_list[1], self.kmeans)
            self.hist_list.append(histogram)

    def find_nearest_neighbor(self, curr_histogram):
        neighbor = NearestNeighbors(n_neighbors=1)
        neighbor.fit(self.hist_list)
        dist, result = neighbor.kneighbors([curr_histogram])

        return dist, result

    def compare_keypoints(self, target):
        # Find labels of keypoints
        preindex = []
        last_stop = -1
        for i in range(len(self.keypoint_list[target][0])):
            for j in range(len(self.BA_list)):
                if self.BA_list[j][1] == target and j > last_stop:
                    preindex.append(j)
                    last_stop = j

        # Sort labels of keypoints
        labels = []
        for i in range(len(self.keypoint_list[target][0])):
            for j in range(len(preindex)):
                x_correct = (self.keypoint_list[target][0][0][i] == self.BA_list[preindex[j]][2])
                y_correct = (self.keypoint_list[target][0][1][i] == self.BA_list[preindex[j]][3])
                if x_correct and y_correct:
                    labels.append(self.BA_list[target][j][0])

        # match descriptors of image 1 and image 2 keypoints
        matches = self.bf.match(self.keypoint_list[target][1], self.keypoint_list[self.curr_frame][1])
        # create empty lists to store matched keypoints and labels
        matched_kp = []
        matched_labels = []
        for match in matches:
            # extract index of matched keypoint in image 2
            img2_idx = match.trainIdx

            # extract coordinates of matched keypoints in image 1 and image 2
            kp = self.keypoint_list[self.curr_frame][0][match.queryIdx].pt

            # append matched keypoints and labels to lists
            matched_kp.append(kp)
            matched_labels.append(labels[img2_idx])

        # create a list to store unmatched keypoints
        unmatched_kp = []
        unmatched_3D = []

        # loop over all keypoints in image 2 and check if each one is in matched keypoints
        for i in range(len(self.keypoint_list[self.curr_frame][1])):
            if i not in [match.trainIdx for match in matches]:
                # if not matched, append the keypoint coordinates to the list of unmatched keypoints
                unmatched_kp.append(self.keypoint_list[self.curr_frame][0][i].pt)
                unmatched_3D.append(self.keypoint_list[self.curr_frame][2][i])

        # return list of matched keypoints and labels
        return matched_kp, matched_labels, unmatched_kp, unmatched_3D

    def append_keypoints_match(self, matched_kp, matched_labels, unmatched_kp, unmatched_3D):
        # Create each point in the form the BA likes
        matched_point_list = []
        for i in range(len(matched_kp)):
            matched_point_list[i] = [matched_labels[i], self.curr_frame, matched_kp[i][0], matched_kp[i][1]]
        # Sort by first index
        # matched_point_list = sorted(matched_point_list, key=lambda x: x[0])

        # If BA_list is empty, just append
        if not self.BA_list:
            for i in range(len(matched_point_list)):
                self.BA_list.append(matched_point_list[i])
        # Append matched points
        if self.BA_list:
            for i in range(len(matched_kp)):
                for j in range(len(self.BA_list)):
                    current_match_point = (self.BA_list[j][0] == matched_labels[i])
                    next_is_false = (self.BA_list[j + 1][0] != matched_labels[i])
                    correct_point = current_match_point and next_is_false
                    if correct_point:
                        self.BA_list.insert(j + 1, matched_point_list[i])

    def residual_append(self, target, match, no_match):
        match_list = []
        no_match_list = []
        for i in range(len(match)):
            hit = False
            for j in range(len(self.BA_list)):
                match_frame = (self.BA_list[j][1] == target)
                match_x = (self.BA_list[j][2] == match[i][0][0])
                match_y = (self.BA_list[j][3] == match[i][0][1])
                if match_frame and match_x and match_y:
                    hit = True
            if not hit:
                match_list.append(match[i])

        for i in range(len(no_match)):
            hit = False
            for j in range(len(self.BA_list)):
                match_frame = (self.BA_list[j][1] == target)
                match_x = (self.BA_list[j][2] == no_match[i][0][0])
                match_y = (self.BA_list[j][3] == no_match[i][0][1])
                if match_frame and match_x and match_y:
                    hit = True
            if not hit:
                no_match_list.append(no_match[i])

        self.append_keypoints_tracked(match_list[0], match_list[2], match_list[3])
        self.append_keypoints_no_match(no_match_list[0], no_match_list[2])

    def append_keypoints_tracked(self, keypoints, coordinates, match):
        # Append unmatched list
        startpoint = self.BA_list[-1][0]
        for i in range(len(keypoints)):
            for j in range(len(self.BA_list)):
                match_x = (self.BA_list[j][1] == (self.curr_frame - 1) and self.BA_list[j][2] == match[i][0])
                match_y = (self.BA_list[j][1] == (self.curr_frame - 1) and self.BA_list[j][3] == match[i][1])
                tracked_frame = (match_x and match_y)
                if tracked_frame:
                    point = [self.BA_list[j][0], self.curr_frame, keypoints[i][0], keypoints[i][1]]
                    self.BA_list.insert((j+1), [self.BA_list[j][0], self.curr_frame, point[0], point[1]])
                    self.coord_3d_list.append(coordinates[i])

    def append_keypoints_no_match(self, keypoints, coordinates):
        # Append unmatched list
        startpoint = self.BA_list[-1][0]
        for i in range(len(keypoints)):
            point = [keypoints[i][0], keypoints[i][1]]
            self.BA_list.append([startpoint + i + 1, self.curr_frame, point[0], point[1]])
            self.coord_3d_list.append(coordinates[i])

    def run_feature_detector(self, match, no_match):
        self.append_features(match, no_match)
        self.append_histogram()
        dist, target = self.find_nearest_neighbor(self.hist_list[-1])
        if dist < self.dist_limit:
            self.append_keypoints_match(self.compare_keypoints(target))
            self.residual_append(target, match, no_match)
        else:
            self.append_keypoints_tracked(match[0], match[2], match[3])
            self.append_keypoints_no_match(no_match[0], no_match[2])

        return dist

# TODO: Make the main file sort the descriptors with the keypoints when doing the optical flow
# TODO: Once they are sorted along with each other, push them into self.keypoint_list
# TODO: Do this instead of running self.features()
# TODO: Then you can just run self.run_feature_detector(keypoints, descriptors), and do BA if dist is close/far(?) enough
