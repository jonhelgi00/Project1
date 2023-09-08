import cv2
import numpy as np

# read input image
img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate SIFT object with default values
# sift = cv2.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create() 

# find the keypoints on image (grayscale)
kp = sift.detect(gray,None)

# draw keypoints in image
img2 = cv2.drawKeypoints(gray, kp, None, flags=0)

# display the image with keypoints drawn on it
cv2.imwrite('KTH_SIFT1.jpg', img2)
cv2.imshow("Keypoints", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()