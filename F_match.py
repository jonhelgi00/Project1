import cv2
import numpy as np
import matplotlib.pyplot as plt

def fixed_threshold():
  img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
  img_database = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_t1.JPG')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_database = cv2.cvtColor(img_database, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 

  kp, des = sift.detectAndCompute(gray, None)
  kp_database, des_database = sift.detectAndCompute(gray_database, None)

  print(f"Number of keypoints in query image: {len(kp)}")
  print(f"Number of keypoints in database image: {len(kp_database)}")

  # img2_ = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # img2_database = cv2.drawKeypoints(gray_database, kp_database, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # cv2.imwrite('KTH_SIFT_database(t1)_kp.jpg', img2_database)
  # cv2.imshow("Keypoints in obj1_t1.JPG", img2_database)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  # Initialize the Brute-Force Matcher
  bf = cv2.BFMatcher()

  max_radius = 50
  # Match descriptors
  matches = bf.radiusMatch(des, des_database, maxDistance = max_radius)
  
    
  # good_matches = []
  
  # for m, n in matches:
  #   if m.distance < 0.98 * n.distance:
  #       good_matches.append([m])

  matched_image = cv2.drawMatchesKnn(gray, kp, gray_database, kp_database, matches, None,
          matchColor=(0, 255, 0), matchesMask=None,
          singlePointColor=(255, 0, 0), flags=0)
  
  # cv2.imwrite('KTH_SIFT_matching_r50_subopt.jpg', matched_image)
  cv2.imshow("matches", matched_image)
  cv2.waitKey(0)

def NN():
  img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
  img_database = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_t1.JPG')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_database = cv2.cvtColor(img_database, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 

  kp, des = sift.detectAndCompute(gray, None)
  kp_database, des_database = sift.detectAndCompute(gray_database, None)

  bf = cv2.BFMatcher()

  # Match descriptors
  matches = bf.knnMatch(des, des_database, k=1)
  
  matched_image = cv2.drawMatchesKnn(gray, kp, gray_database, kp_database, matches, None,
          matchColor=(0, 255, 0), matchesMask=None,
          singlePointColor=(255, 0, 0), flags=0)
  
  cv2.imwrite('KTH_SIFT_matching_NN.jpg', matched_image)
  cv2.imshow("matches", matched_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def NNDR():
  img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
  img_database = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_t1.JPG')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_database = cv2.cvtColor(img_database, cv2.COLOR_BGR2GRAY)

  sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100) 

  kp, des = sift.detectAndCompute(gray, None)
  kp_database, des_database = sift.detectAndCompute(gray_database, None)

  bf = cv2.BFMatcher()

  # Match descriptors
  matches = bf.knnMatch(des, des_database, k=2)

  good_matches = []

  ratio = 0.8
  
  for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append([m])
  
  matched_image = cv2.drawMatchesKnn(gray, kp, gray_database, kp_database, good_matches, None,
          matchColor=(0, 255, 0), matchesMask=None,
          singlePointColor=(255, 0, 0), flags=0)
  
  print(len(good_matches))
  # cv2.imwrite('KTH_SIFT_matching_NNDR_0.5.jpg', matched_image)
  # cv2.imshow("matches", matched_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()  


def NNDR_SURF():
  img = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_5.JPG')
  img_database = cv2.imread('/Users/jonhelgi/KTH_projects/Analysis_Search/Project1/obj1_t1.JPG')

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_database = cv2.cvtColor(img_database, cv2.COLOR_BGR2GRAY)

  # surf = cv2.xfeatures2d.SURF_create(5000) 
  surf = cv2.xfeatures2d.SURF_create(6000) 

  kp, des = surf.detectAndCompute(gray,None)
  kp_database, des_database = surf.detectAndCompute(gray_database,None)
  print(f"Number of keypoints in 1_5 is: {len(kp)}")
  print(f"Number of keypoints in t1 is: {len(kp_database)}")

  # kp, des = sift.detectAndCompute(gray, None)
  # kp_database, des_database = sift.detectAndCompute(gray_database, None)

  bf = cv2.BFMatcher()

  # Match descriptors
  matches = bf.knnMatch(des, des_database, k=2)

  good_matches = []

  ratio = 0.8
  
  for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append([m])
  
  matched_image = cv2.drawMatchesKnn(gray, kp, gray_database, kp_database, good_matches, None,
          matchColor=(0, 255, 0), matchesMask=None,
          singlePointColor=(255, 0, 0), flags=0)
  
  print(len(good_matches))
  
  # cv2.imwrite('KTH_SURF_matching_NNDR_0.8_new.jpg', matched_image)
  # cv2.imshow("matches", matched_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()  



NNDR()
