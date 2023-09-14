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
  # cv2.imwrite('KTH_SIFT_size_orient_500kp.jpg', img2_)
  cv2.imshow("Keypoints", img2_)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

##
# b)

# Load the image
img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')

#add padding to avoid cut-off
old_height, old_width, channels = img.shape

new_width = int(np.ceil(np.sqrt(old_height**2 + old_width**2)))
color = (0,0,0)
result = np.full((new_width, new_width, channels), color, dtype=np.uint8)

x_center = (new_width - old_width) // 2
y_center = (new_width - old_height) // 2

# copy img image into center of result image
result[y_center:y_center+old_height, 
       x_center:x_center+old_width] = img

img_ = result

#grayscale
gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

# Get the center of the image
height, width = gray.shape[:2]
# center = (width // 2, height // 2) #floored
center = (width / 2, height / 2)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 
kp_og = sift.detect(gray,None)
N_kp = len(kp_og)
img2_ = cv2.drawKeypoints(gray, kp_og, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# print(type(kp_og[0].pt[0]))
# cv2.circle(gray,(round(kp_og[200].pt[0]),round(kp_og[200].pt[1])), 30, (250,0,0), -1)
# cv2.imwrite('KTH_SIFT_padding_528kp.jpg', img2_)
# cv2.imshow('Keypoints in OG image (padded)', img2_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Define the rotation angle (in degrees)
angle_list = np.arange(stop=360, step=15) # 0-345 degrees
match_counter = 0
repeat_list = []

for angle in angle_list:
  # Calculate the rotation matrix
  match_counter = 0
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  # Apply the rotation to the image
  rotated_image = cv2.warpAffine(gray, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
  kp_rotated = sift.detect(rotated_image,None) #detect keypoints
  kp_ideal = []
  for kp in kp_og:
    og_coord = np.array([[kp.pt[0],
                         kp.pt[1]]])
    rotation_matrix = np.array(rotation_matrix) #transform to np array
    # kp_new = rotation_matrix[:,:2] @ og_coord.T + rotation_matrix[:,-1]
    kp_new = rotation_matrix[:,:2] @ og_coord.T + np.reshape(rotation_matrix[:,-1],(2,1))
    kp_new = np.reshape(kp_new, (2,))
    kp_ideal.append(kp_new)
    found = False
    for kp_rot in kp_rotated:
      rot_coord = kp_rot.pt
      rot_coord = np.array(rot_coord)
      if np.linalg.norm(kp_new - rot_coord) < 2 and not found:
        match_counter += 1
        found = True
  repeatability = match_counter / N_kp
  repeat_list.append((angle, repeatability))  
  

print(repeat_list)

def scalebla():
  img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_scaled = cv2.resize(gray, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
  cv2.imshow('Scaled', img_scaled)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# scalebla()




# # Display the rotated image
# cv2.imshow('Rotated Image', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


