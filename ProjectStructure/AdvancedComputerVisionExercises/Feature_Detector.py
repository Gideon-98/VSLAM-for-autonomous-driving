from Visual_SLAM_Solution import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# BoW code inspired by https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb


class FeatureDetector:

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

    def append_features(self, keypoints, descriptors, coords):
        self.curr_frame += 1
        self.keypoint_list.append([keypoints, descriptors, coords])

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

        # Append unmatched list
        startpoint = self.BA_list[-1][0]
        for i in range(len(unmatched_kp)):
            self.BA_list.append([startpoint + i + 1, self.curr_frame, unmatched_kp[i][0], unmatched_kp[i][1]])
            self.coord_3d_list.append(unmatched_3D[i])

    def append_keypoints_no_match(self):
        # Append unmatched list
        startpoint = self.BA_list[-1][0]
        for i in range(len(self.keypoint_list[self.curr_frame])):
            point = [self.keypoint_list[self.curr_frame][i][0], self.keypoint_list[self.curr_frame][i][1]]
            self.BA_list.append([startpoint + i + 1, point[0], point[1]])
            self.coord_3d_list.append(self.keypoint_list[self.curr_frame][i][2])

    def run_feature_detector(self, keypoints, descriptors, coords):
        self.append_features(keypoints, descriptors, coords)
        self.append_histogram()
        dist, target = self.find_nearest_neighbor(self.hist_list[-1])
        if dist < self.dist_limit:
            self.append_keypoints_match(self.compare_keypoints(target))
        else:
            self.append_keypoints_no_match()

        return dist

# TODO: Make the main file sort the descriptors with the keypoints when doing the optical flow
# TODO: Once they are sorted along with each other, push them into self.keypoint_list
# TODO: Do this instead of running self.features()
# TODO: Then you can just run self.run_feature_detector(keypoints, descriptors), and do BA if dist is close/far(?) enough
