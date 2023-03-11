from Visual_SLAM_Solution import cv2


class FeatureDetector:
    def __init__(self, detector_type='SIFT'):
        self.detector_type = detector_type
        self.detector = self._create_detector()

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
