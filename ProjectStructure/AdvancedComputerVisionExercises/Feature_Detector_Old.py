from Visual_SLAM_Solution import cv2


class FeatureDetector:
    def __init__(self, detector_type='SIFT'):
        self.detector_type = detector_type
        self.detector = self._create_detector()
        self.curr_frame = 0
        self.keypoints_list = []

    def _create_detector(self):
        if self.detector_type == 'SIFT':
            return cv2.xfeatures2d.SIFT_create()
        elif self.detector_type == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        elif self.detector_type == 'ORB':
            return cv2.ORB_create()
        elif self.detector_type == 'BRISK':
            return cv2.BRISK_create()
        elif self.detector_type == 'KAZE':
            return cv2.KAZE_create()
        elif self.detector_type == 'FAST':
            return cv2.FastFeatureDetector_create()
        elif self.detector_type == 'HARRIS':
            return cv2.cornerHarris()

    def detect(self, gray):
        keypoints = self.detector.detect(gray, None)
        return keypoints

    def extract_descriptors(self, gray, keypoints):
        if self.detector_type in ['SIFT', 'SURF', 'KAZE']:
            descriptors = self.detector.compute(gray, keypoints)[1]
        elif self.detector_type == 'ORB':
            descriptors = cv2.ORB_create().compute(gray, keypoints)[1]
        elif self.detector_type == 'BRISK':
            descriptors = cv2.BRISK_create().compute(gray, keypoints)[1]
        elif self.detector_type in ['FAST', 'HARRIS']:
            descriptors = None
        else:
            keypoints, descriptors = self.detector.compute(gray, keypoints)
        return descriptors

    def update_keypoints_list(self, prev_keypoints, curr_keypoints):
        self.curr_frame += 1

        # Loop over the previous frame's keypoints
        for i in range(len(prev_keypoints)):
            x, y = prev_keypoints[i].pt

            # Search through the keypoints list in reverse order
            for j in range(len(self.keypoints_list) - 1, -1, -1):
                # If a match is found between the previous frame's keypoint and a keypoint in the list,
                # append the current frame's keypoint with the matching keypoint number
                if self.keypoints_list[j][2:] == (x, y) and self.keypoints_list[j][1] == self.curr_frame - 1:
                    x, y = curr_keypoints[j].pt
                    self.keypoints_list.append((self.keypoints_list[j][0], self.curr_frame, x, y))
                    break
            # If no match is found, append the current frame's keypoint with the index of the previous keypoint
            else:
                x, y = curr_keypoints[i].pt
                self.keypoints_list.append((i + 1, self.curr_frame, x, y))

        # Sort the keypoints list by frame number, then by keypoint number
        self.keypoints_list = sorted(self.keypoints_list, key=lambda x: (x[1], x[0]))