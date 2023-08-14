import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load stereo images
left_image = cv2.imread('C:\\Users\\gianp\\PycharmProjects\\VSLAM-for-autonomous-driving\\ProjectStructure\\AdvancedComputerVisionExercises\\data\\KITTI_sequence_1\\image_l\\000007.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('C:\\Users\\gianp\\PycharmProjects\\VSLAM-for-autonomous-driving\\ProjectStructure\\AdvancedComputerVisionExercises\\data\\KITTI_sequence_1\\image_r\\000007.png', cv2.IMREAD_GRAYSCALE)

#plt.imshow(left_image, cmap='jet')
block = 11
P1 = block * block * 8
P2 = block * block * 32
# Initialize StereoSGBM object for disparity computation
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=block,
    P1=P1,
    P2=P2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)


# Compute disparity map
disparity_map = stereo.compute(left_image, right_image)

# Display the original left and right images
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(left_image, cmap='gray')
plt.title('Left Image')
plt.subplot(122)
plt.imshow(right_image, cmap='gray')
plt.title('Right Image')
plt.tight_layout()
plt.show()

# Display the grayscale disparity map
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(disparity_map, cmap='gray')
ax.set_title('Disparity Map (Grayscale)')

# Create a grayscale color map
cmap_gray = mcolors.ListedColormap(['black', 'white'])
norm_gray = mcolors.Normalize(vmin=disparity_map.min(), vmax=disparity_map.max())

# Add the horizontal grayscale color bar
cbar_gray = plt.colorbar(im, ax=ax, cmap=cmap_gray, norm=norm_gray, orientation='horizontal')
cbar_gray.set_label('Disparity')

# Display the colored disparity map with a color map
fig_colored, ax_colored = plt.subplots(figsize=(10, 5))
im_colored = ax_colored.imshow(disparity_map, cmap='jet')
ax_colored.set_title('Disparity Map (Colored)')

# Add the horizontal color bar
cbar_colored = plt.colorbar(im_colored, ax=ax_colored, orientation='horizontal')
cbar_colored.set_label('Disparity (Color)')

plt.show()