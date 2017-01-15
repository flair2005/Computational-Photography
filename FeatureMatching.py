

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
      raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                      % cv2.__version__)


def findMatchesBetweenImages(image_1, image_2):
  matches = None
  image_1_kp = None
  image_1_desc = None
  image_2_kp = None
  image_2_desc = None

  #initialize ORB
  orb=cv2.ORB();

  #detect keypoint and descriptor
  kp1, des1 = orb.detectAndCompute(image_1, None)
  kp2, des2 = orb.detectAndCompute(image_2, None)
  image_1_kp = cv2.drawKeypoints(image_1, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  image_2_kp = cv2.drawKeypoints(image_2, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # Create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors
  matches = bf.match(des1, des2)
  print "{} matches found".format(len(matches))

  # Sort them in the order of their distance
  matches = sorted(matches, key = lambda x: x.distance)

  # Compute matches
  img_out = drawMatches(image_1,kp1,image_2,kp2,matches)

  return kp1, kp2, matches[:10]


