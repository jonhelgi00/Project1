import cv2
import numpy as np

#######
2.2
#######
# read input image
img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate SIFT object with default values
# sift = cv2.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 

# find the keypoints on image (grayscale)
kp = sift.detect(gray,None)
print(f"Number of keypoints is: {len(kp)}")

# draw keypoints in image
# img2 = cv2.drawKeypoints(gray, kp, None, flags=0)

#draw keypoints with size and orientation
img2_ = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image with keypoints drawn on it
cv2.imwrite('KTH_SIFT_size_orient_500kp.jpg', img2_)
cv2.imshow("Keypoints", img2_)
cv2.waitKey(0)
cv2.destroyAllWindows()