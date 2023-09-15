import cv2
import numpy as np
import matplotlib.pyplot as plt

# read input image
#img = cv2.imread('/Users/user/Desktop/KTH/Analysis and Search of Visual Data/project_1/data1/obj1_5.JPG')
# convert the image to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate SURF object with default values
#surf = cv2.xfeatures2d.SURF_create(6000)

# Find keypoints and descriptors directly
#kp_og, des_og = surf.detectAndCompute(gray,None)
#img2 = cv2.drawKeypoints(gray, kp_og, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Define the scaling factors
SF = [1.2] * 3 # create a list of nine 1.2
power = np.arange(3) # create list of power from 0 to 8
SF_list = np.power(SF, power)

def SURF_scaling():    
    # read input image
    img = cv2.imread('/Users/user/Desktop/KTH/Analysis and Search of Visual Data/project_1/data1/obj1_5.JPG')

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate SURF object with default values
    surf = cv2.xfeatures2d.SURF_create(6000) 
    print( f"SURF threshold: {surf.getHessianThreshold()}" )

    # Find keypoints and descriptors directly
    kp_og, des_og = surf.detectAndCompute(gray,None)
    N_kp = len(kp_og)
    print(f"Number of keypoints for SURF is: {len(kp_og)}")

    # Define the scaling factors
    SF = [1.2] * 9 # create a list of nine 1.2
    power = np.arange(9) # create list of power from 0 to 8
    SF_list = np.power(SF, power)
    match_counter = 0
    repeat_list = []
    match_counter_dyn = 0
    repeat_list_dyn = []

    for sf in SF_list:
        match_counter = 0
        match_counter_dyn = 0

        # Apply scaling to the image
        gray_scaled = cv2.resize(gray, None, fx = sf, fy = sf, interpolation = cv2.INTER_AREA)
        kp_scaled, des_scaled = surf.detectAndCompute(gray_scaled, None)
        kp_ideal = []
        for kp in kp_og:
            og_coord = np.array([[kp.pt[0],kp.pt[1]]])
            kp_new = og_coord * sf
            kp_ideal.append(kp_new)
            found = False
            found_dyn = False

            for kp_scal in kp_scaled:
                scaled_coord = kp_scal.pt
                scaled_coord = np.array(scaled_coord)
                if np.linalg.norm(kp_new - scaled_coord) < 2 and not found and not found_dyn:
                    match_counter += 1
                    match_counter_dyn += 1
                    found = True
                    found_dyn = True
                elif np.linalg.norm(kp_new - scaled_coord) < 2*sf and not found_dyn:
                    match_counter_dyn += 1
                    found_dyn = True

        repeatability = match_counter / N_kp
        repeat_list.append((sf, repeatability)) 
        repeatability_dyn = match_counter_dyn / N_kp
        repeat_list_dyn.append((sf, repeatability_dyn))

    # print(repeat_list)
    repeat_list = np.array(repeat_list)
    print(f"SURF repeatability with threshold = 2: {repeat_list[:,1]}")
    repeat_list_dyn = np.array(repeat_list_dyn)
    print(f"SURF repeatability with dynamic threshold: {repeat_list_dyn[:,1]}")

    return repeat_list, repeat_list_dyn

def SIFT_scaling():
    # read input image
    img = cv2.imread('/Users/user/Desktop/KTH/Analysis and Search of Visual Data/project_1/data1/obj1_5.JPG')

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT object 
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.18, edgeThreshold = 100)

    # Find keypoints and descriptors directly
    kp_og = sift.detect(gray,None)
    N_kp = len(kp_og)
    print(f"Number of keypoints for SIFT is: {len(kp_og)}")

    # Define the scaling factors
    SF = [1.2] * 3 # create a list of nine 1.2
    power = np.arange(3) # create list of power from 0 to 8
    SF_list = np.power(SF, power)
    match_counter = 0
    repeat_list = []
    match_counter_dyn = 0
    repeat_list_dyn = []

    for sf in SF_list:
        match_counter = 0
        match_counter_dyn = 0

        # Apply scaling to the image
        gray_scaled = cv2.resize(gray, None, fx = sf, fy = sf, interpolation = cv2.INTER_AREA)
        kp_scaled = sift.detect(gray_scaled, None)
        kp_ideal = []
        for kp in kp_og:
            og_coord = np.array([[kp.pt[0],kp.pt[1]]])
            kp_new = og_coord * sf
            kp_ideal.append(kp_new)
            found = False
            found_dyn = False

            for kp_scal in kp_scaled:
                scaled_coord = kp_scal.pt
                scaled_coord = np.array(scaled_coord)
                if np.linalg.norm(kp_new - scaled_coord) < 2 and not found and not found_dyn:
                    match_counter += 1
                    match_counter_dyn += 1
                    found = True
                    found_dyn = True
                elif np.linalg.norm(kp_new - scaled_coord) < 2*sf and not found_dyn:
                    match_counter_dyn += 1
                    found_dyn = True

        repeatability = match_counter / N_kp
        repeat_list.append((sf, repeatability)) 
        repeatability_dyn = match_counter_dyn / N_kp
        repeat_list_dyn.append((sf, repeatability_dyn))

    # print(repeat_list)
    repeat_list = np.array(repeat_list)
    print(f"SIFT repeatability with threshold = 2: {repeat_list[:,1]}")
    repeat_list_dyn = np.array(repeat_list_dyn)
    print(f"SIFT repeatability with dynamic threshold: {repeat_list_dyn[:,1]}")

    return repeat_list, repeat_list_dyn

repeat_list_SIFT, repeat_list_dyn_SIFT = SIFT_scaling()
#repeat_list_SURF, repeat_list_dyn_SURF = SURF_scaling()

#Plot repeatability
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed.
# Plot the data.
#plt.plot(repeat_list_SURF[:,0], repeat_list_SURF[:,1], label='SURF threshold = 2', color='blue', linestyle='-')
#plt.plot(repeat_list_dyn_SURF[:,0], repeat_list_dyn_SURF[:,1], label='SURF threshold = 2*sf', color='black', linestyle='-')
plt.plot(repeat_list_SIFT[:,0], repeat_list_SIFT[:,1], label='SIFT threshold = 2', color='red', linestyle='-')
plt.plot(repeat_list_dyn_SIFT[:,0], repeat_list_dyn_SIFT[:,1], label='SIFT threshold = 2*sf', color='brown', linestyle='-')
# Add a title.
plt.title('Repeatability vs Scaling for SIFT & SURF')
# Customize x and y labels.
plt.xlabel('Scaling factor')
plt.ylabel('Repeatability')
# Customize xticks (here, I'm setting custom ticks at specific positions).
custom_xticks = SF_list  # Replace with your custom xticks.
plt.xticks(custom_xticks)
# Add a legend if multiple lines are plotted.
plt.legend()
plt.ylim((0, 1))
# Show the plot.
plt.show()

#scale_percent = 30 # percent of original size
#width = int(img.shape[1] * scale_percent / 100)
#height = int(img.shape[0] * scale_percent / 100)
#dim = (width, height)

# resize image
#resized_2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
# display the image with keypoints drawn on it
#cv2.imwrite('KTH_SURF1.jpg', resized_2)
#cv2.imshow("Keypoints", resized_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

