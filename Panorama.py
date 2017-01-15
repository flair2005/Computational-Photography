

import numpy as np
import scipy as sp
import scipy.signal
import cv2

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB."
                                 % cv2.__version__)



def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    corners = np.array([[[0,0]],[[0,image.shape[0]]],[[image.shape[1],0]],[[image.shape[1],image.shape[0]]]], dtype=np.float32)
    return corners

def findMatchesBetweenImages(image_1, image_2, num_matches):

    matches = None
    image_1_kp = None
    image_1_desc = None
    image_2_kp = None
    image_2_desc = None

    #initialize ORB
    orb=cv2.ORB();

    #detect keypoint and descriptor
    image_1_kp, des1 = orb.detectAndCompute(image_1, None)
    image_2_kp, des2 = orb.detectAndCompute(image_2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x: x.distance)

    # We coded the return statement for you. You are free to modify it -- just
    # make sure the tests pass.
    return image_1_kp, image_2_kp, matches[:num_matches]

  # END OF FUNCTION.

def findHomography(image_1_kp, image_2_kp, matches):
    # Create two sequences of corresponding (matched) points
    image_1_points = []
    image_2_points = []
    for mat in matches:
        # Get the matching keypoints for each of the images
        image_1_points.append(image_1_kp[mat.queryIdx].pt)
        image_2_points.append(image_2_kp[mat.trainIdx].pt)
    image_1_points = np.asarray(image_1_points, dtype=np.float32)
    image_2_points = np.asarray(image_2_points, dtype=np.float32)

    # Compute homography (since our objects are planar) using RANSAC to reject outliers
    M_hom, inliers = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC) # transform img2 to img1's space

    # Replace this return statement with the homography.
    return M_hom

def blendImagePair(warped_image, image_2, point):
    output_image = np.copy(warped_image)

    corners1 = getImageCorners(image_1)
    corners2 = getImageCorners(image_2)
    corners1Trans = cv2.perspectiveTransform(corners1,homography)

    mins = np.amin(corners1Trans, axis=0)
    maxs = np.amax(corners1Trans, axis=0)
    Img1_x_min = 0
    Img1_x_max = maxs[0][0] - mins[0][0]
    Img1_y_min = 0
    Img1_y_max = maxs[0][1] - mins[0][1]
    #print Img1_x_min," ",Img1_x_max," ",Img1_y_min," ",Img1_y_max

    maxs = np.amax(corners2, axis=0)
    Img2_x_min = int(point[0])
    Img2_x_max = int(maxs[0][0] + point[0])
    Img2_y_min = int(point[1])
    Img2_y_max = int(maxs[0][1] + point[1])
    #print "\n",Img2_x_min," ",Img2_x_max," ",Img2_y_min," ",Img2_y_max

    Int_x_min = max(Img1_x_min,Img2_x_min) #Int means Intersection
    Int_y_min = max(Img1_y_min,Img2_y_min)
    Int_x_max = min(Img1_x_max,Img2_x_max)
    Int_y_max = min(Img1_y_max,Img2_y_max)
    #print "\n",Int_x_min," ",Int_x_max," ",Int_y_min," ",Int_y_max
    #Compute Image
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2
    #Handle Intersection Window
    Img1_x_centre = ( Img1_x_min + Img1_x_max ) / 2
    Img1_y_centre = ( Img1_y_min + Img1_y_max ) / 2
    max_distance1 =  int( ( Img1_x_max - Img1_x_centre ) +( Img1_y_max - Img1_y_centre ) )
    Img2_x_centre = ( Img2_x_min + Img2_x_max ) / 2
    Img2_y_centre = ( Img2_y_min + Img2_y_max ) / 2
    max_distance2 =  int( ( Img2_x_max - Img2_x_centre ) +( Img2_y_max - Img2_y_centre ) )

    for x in range(Int_x_min,Int_x_max):
        for y in range(Int_y_min,Int_y_max):
            distance2 = float( abs( x - Img2_x_centre) + abs( y - Img2_y_centre ) )
            alpha = distance2/max_distance2

            distance1 = float( abs( x - Img1_x_centre) + abs( y - Img1_y_centre ) )
            beta = distance1/max_distance1
            try:
                if not (warped_image[y,x].all() == 0 ):
                    if alpha > beta :
                        output_image[y,x] =   alpha * warped_image[y,x] +  ( 1-alpha ) * image_2[y-point[1],x-point[0]]
                    else :
                        output_image[y,x] =   (1-beta) * warped_image[y,x] +  beta * image_2[y-point[1],x-point[0]]
            except IndexError:
                continue

    return output_image

def warpImagePair(image_1, image_2, homography):

    corners1 = getImageCorners(image_1)
    corners2 = getImageCorners(image_2)

    #perspective transform
    corners1Trans = cv2.perspectiveTransform(corners1,homography)
    #corners1Trans = np.dot(homography,corners1)

    cornersAll=np.concatenate((corners1Trans,corners2))

    mins = np.amin(cornersAll, axis=0)
    maxs = np.amax(cornersAll, axis=0)
    x_min = mins[0][0]
    x_max = maxs[0][0]
    y_min = mins[0][1]
    y_max = maxs[0][1]

    translationM = [[1, 0, -1 * x_min],
                    [0, 1, -1 * y_min],
                    [0, 0, 1]]

    translatedHomography = np.dot(translationM, homography)

    warped_image = cv2.warpPerspective(image_1, translatedHomography, (x_max - x_min, y_max - y_min)) # apply transform to img2

    output_image = blendImagePair(warped_image, image_2,
                                  (-1 * x_min, -1 * y_min))
    return output_image

#Some simple testing.
image_1 = cv2.imread("images/source/panorama_3/4.jpg")
image_2 = cv2.imread("images/source/panorama_3/5.jpg")
image_1_kp, image_2_kp, matches = findMatchesBetweenImages(image_1, image_2, 400)
homography = findHomography(image_1_kp, image_2_kp, matches)
result = warpImagePair(image_1, image_2, homography)
cv2.imwrite("images/output/finfin.jpg", result)