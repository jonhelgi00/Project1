import cv2

# read input image
img = cv2.imread('/Users/user/Desktop/KTH/Analysis and Search of Visual Data/project_1/data1/obj1_5.JPG')
#img = img.astype('uint8')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Initiate SURF object with default values
surf = cv2.xfeatures2d.SURF_create(5000) 
print( f"Threshold: {surf.getHessianThreshold()}" )

# Find keypoints and descriptors directly
kp_surf, des_surf = surf.detectAndCompute(gray,None)
print(f"Number of keypoints is: {len(kp_surf)}")

img2 = cv2.drawKeypoints(gray, kp_surf, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# resize image
resized_2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
# display the image with keypoints drawn on it
cv2.imwrite('KTH_SURF1.jpg', resized_2)
cv2.imshow("Keypoints", resized_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_scaled = cv2.resize(gray, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
# display the image with keypoints drawn on it
#cv2.imwrite('Scaled=1.2.jpg', img_scaled)
#cv2.imshow("Scaling factor = 1.2", img_scaled)
#cv2.waitKey(0)
#cv2.destroyAllWindows()