from Visual_SLAM_Solution import cv2, np


class HistogramGenerator:
    def __init__(self, num_clusters=50, threshold=0.5):
        self.num_clusters = num_clusters
        self.vocabulary = None
        self.matcher = None
        self.histograms = []
        self.threshold = threshold

    def generate_bow_histogram(self, keypoints, descriptors):
        if self.vocabulary is None:
            # Cluster descriptors to create visual vocabulary
            kmeans = cv2.KMeans(n_clusters=self.num_clusters)
            kmeans.fit(descriptors)
            self.vocabulary = kmeans.cluster_centers_

            # Create FLANN-based matcher
            flann_params = dict(algorithm=1, trees=5)
            self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        # Match descriptors to visual words
        matches = self.matcher.match(descriptors, self.vocabulary)

        # Count frequency of each visual word in the frame
        histogram = np.zeros(self.num_clusters)
        for match in matches:
            histogram[match.trainIdx] += 1

        return histogram

    def update_histograms(self, keypoints, descriptors):
        histogram = self.generate_bow_histogram(keypoints, descriptors)
        self.histograms.append(histogram)

    def detect_loop_closure(self):
        num_histograms = len(self.histograms)
        if num_histograms < 2:
            return -1

        current_histogram = self.histograms[-1]

        for i in range(num_histograms - 2, -1, -1):
            previous_histogram = self.histograms[i]
            distance = cv2.compareHist(current_histogram, previous_histogram, cv2.HISTCMP_CHISQR)
            if distance < self.threshold:
                return i

        return -1

"""
To use this class for Visual SLAM, you can create an instance of HistogramGenerator and call its 
update_histograms function with new frames as they become available. You can then call the 
detect_loop_closure function to check for loop closures:

histogram_generator = HistogramGenerator(num_clusters=50, threshold=0.5)
for i in range(num_frames):
    frame = get_next_frame()
    histogram_generator.update_histograms([frame])
    loop_closure_index = histogram_generator.detect_loop_closure()
    if loop_closure_index >= 0:
        # Perform loop closure
        # ...
"""