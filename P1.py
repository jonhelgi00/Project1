import cv2
import numpy as np

#######
2.2
#######
# a)
def part2_2a():
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

##
# b)

# Load the image
img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get the center of the image
height, width = img.shape[:2]
# center = (width // 2, height // 2) #floored
center = (width / 2, height / 2)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 
kp_og = sift.detect(gray,None)

# print(type(kp_og[0].pt[0]))
# cv2.circle(gray,(round(kp_og[200].pt[0]),round(kp_og[200].pt[1])), 30, (250,0,0), -1)
# cv2.imshow('Rotated Image', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Define the rotation angle (in degrees)
angle_list = np.arange(stop=360, step=15) # 0-345 degrees

for angle in angle_list:
    
  # Calculate the rotation matrix
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  # Apply the rotation to the image
  rotated_image = cv2.warpAffine(gray, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
  if angle == 0:
    og_coord = np.array([[kp_og[200].pt[0],
                         kp_og[200].pt[1]]])
    rotation_matrix = np.array(rotation_matrix) #transform to np array
    # print(rotation_matrix[:,:2].shape)
    # print(rotation_matrix[:,-1].shape)
    # print(og_coord.T.shape)
    # kp_new = rotation_matrix[:,:2] @ og_coord.T + rotation_matrix[:,-1]
    kp_new = rotation_matrix[:,:2] @ og_coord.T + np.reshape(rotation_matrix[:,-1],(2,1))
    kp_new = np.reshape(kp_new, (2,))
    print(kp_og[200].pt)
    print(og_coord)
    print(kp_new)
    print(round(kp_og[200].pt[1]))
    print(round(kp_new[1]))
    # print((rotation_matrix[:,:2] @ og_coord.T).shape)
    # center = np.array(center)
    x_rotated = (og_coord[0,0] - center[0]) * np.cos(angle * np.pi/180) - (og_coord[0,1] - center[1]) * np.sin(angle * np.pi/180) + center[0]
    y_rotated = (og_coord[0,0] - center[0]) * np.sin(angle * np.pi/180) + (og_coord[0,1] - center[1]) * np.cos(angle * np.pi/180) + center[1]

    # cv2.circle(rotated_image,((round(kp_new[0]),round(kp_new[1]))), 30, (250,0,0), -1)
    # cv2.circle(rotated_image,(round(kp_og[200].pt[0]),round(kp_og[200].pt[0])), 30, (250,0,0), -1)
    cv2.circle(rotated_image,(round(kp_og[200].pt[0]),round(kp_og[200].pt[1])), 30, (250,0,0), -1)
    # cv2.circle(rotated_image,(round(x_rotated), round(y_rotated)), 30, (250,0,0), -1)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  kp = sift.detect(rotated_image,None)
  img_kp = cv2.drawKeypoints(rotated_image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)





# # Display the rotated image
# cv2.imshow('Rotated Image', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


