
import numpy as np
import scipy as sp
import scipy.signal
import cv2


def generatingKernel(parameter):
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def reduce(image):
  kernel = generatingKernel(0.4)
  conv_img = scipy.signal.convolve2d(image,kernel,'same')
  sub_sample = np.zeros(( np.ceil(image.shape[0]/2) , np.ceil(image.shape[1]/2) ))  
  sub_sample = conv_img[::2,::2]
  return sub_sample

def expand(image):

  #Upsample the image
  upsample = np.zeros((image.shape[0] * 2 , image.shape[1] * 2))
  upsample[::2,::2] = image
  #Do convolution 
  kernel = generatingKernel(0.4)
  conv_img = scipy.signal.convolve2d(upsample,kernel,'same')

  return conv_img * 4


def gaussPyramid(image, levels):
  output = [image]
  while(levels > 0):
    levels = levels - 1
    temp_image = reduce( output[len(output)-1])
    output.append(temp_image)
  return output

def laplPyramid(gaussPyr):
  output = []
  tmp = 0
  while( tmp < len(gaussPyr) - 1):
    expanded_img = expand( gaussPyr[tmp+1] )
    cropped = expanded_img[:gaussPyr[tmp].shape[0]: , :gaussPyr[tmp].shape[1]:]
    result = np.subtract( gaussPyr[tmp] , cropped )
    output.append( result )   
    tmp = tmp + 1
  output.append( gaussPyr[ len(gaussPyr) - 1 ] )
  return output

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):

  blended_pyr = []
  level = 0
  while( level < len(gaussPyrMask) ):
    output= np.zeros((gaussPyrMask[level].shape[0] , gaussPyrMask[level].shape[1]))
    for row in range(gaussPyrMask[level].shape[0]):
      for col in range(gaussPyrMask[level].shape[1]):
        output[row,col] = ( gaussPyrMask[level][row,col] * laplPyrWhite[level][row,col] ) + ( ( 1 - gaussPyrMask[level][row,col] ) * laplPyrBlack[level][row,col] )  
    blended_pyr.append(output)
    level = level +1
  return blended_pyr

def collapse(pyramid):

  level = len(pyramid) - 1
  while(level > 0):
    expanded_img = expand( pyramid[level] )
    cropped = expanded_img[:pyramid[level-1].shape[0]: , :pyramid[level-1].shape[1]:]
    pyramid[level-1] = np.add( pyramid[level-1] , cropped )
    level = level - 1  
  return pyramid[0]

