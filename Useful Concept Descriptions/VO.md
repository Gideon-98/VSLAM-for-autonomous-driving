# Fast Feature Algorithm from cv2
### Basic Idea 
The Fast feature detector algorithm is based on the idea of CORNER DETECTION. The algorithm checks each pixel in the image and COMPARES it with its neighboring pixels to determine WHETHER IT IS A CORNER OR NOT. 

A corner is a point in the image where the brightness changes rapidly in TWO OR MORE directions.

The Fast feature detector checks whether a pixel is brighter or darker than a threshold value, and if it is, it then checks whether there are a SUFFICIENT NUMBER OF CONTIGUOUS pixels in a circle around the pixel that are also brighter or darker. If there are enough contiguous pixels, the pixel is considered to be a keypoint.

The circle around the pixel has a radius of 3 pixels, and the algorithm checks the brightness of 16 pixels on the circle. The algorithm checks the brightness of these pixels in pairs, starting with the pixels on opposite sides of the circle. 
    
If the difference in brightness between the two pixels is greater than a threshold value, the pair of pixels is marked as a candidate for a corner. If there are enough contiguous pairs of candidate pixels, the center pixel is considered to be a corner.

The threshold value used to compare the brightness of pixels is a parameter of the algorithm, and it can be adjusted to make the detector more or less sensitive to changes in brightness. The number of contiguous pixels required to consider a pixel a keypoint is also a parameter that can be adjusted.

### IN SUMMARY, the Fast feature detector algorithm detects keypoints by COMPARING THE BRIGHTNESS of pixels in an image and checking whether there are ENOUGH CONTIGUOUS pixels around the pixel that are brighter or darker. The algorithm uses a threshold value to determine whether a pixel is brighter or darker, and it uses the number of contiguous pixels to determine whether the pixel is a keypoint.

# Optical Flow Algorithm

`cv2.calcOpticalFlowPyrLK` is a function in OpenCV that performs the Lucas-Kanade optical flow algorithm to estimate the motion of feature points between two images or video frames. It is used to track the movement of feature points in an image or video over time.

The function takes in two images, the previous image and the current image, and a set of feature points in the previous image. It then calculates THE DISPLACEMENT OF EACH FEATURE POINT IN THE CURRENT IMAGE by comparing it with the corresponding feature point in the previous image.

The algorithm works by creating a pyramid of images with increasingly lower resolutions. The feature points are tracked starting from the top of the pyramid and working down to the original image resolution. This helps to make the tracking more robust to small changes in the feature points' location and to handle large displacements.

The function uses the Lucas-Kanade algorithm to estimate the optical flow. This algorithm assumes that the displacement of the feature points between two images is small, and it computes the displacement that minimizes the difference between the two images in a neighborhood around each feature point.

The function returns the updated position of the feature points in the current image and a status flag that indicates whether the optical flow was successfully calculated for each feature point. It also returns an error vector that indicates the difference between the predicted and actual position of each feature point.

`cv2.calcOpticalFlowPyrLK` is a useful tool for a variety of computer vision tasks, such as object tracking, motion detection, and video stabilization.