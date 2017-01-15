
import cv2
import numpy as np
import scipy as sp



def imageGradientX(image):
    new =np.zeros((image.shape[0],image.shape[1]-1))
    for row in range(image.shape[0]):
        for col in range(image.shape[1]-1):
		  new[row,col]=image[row,col+1]-image[row,col]
    new = np.absolute(new)
    return new

def imageGradientY(image):

    new =np.zeros((image.shape[0]-1,image.shape[1]))
    for row in range(image.shape[0]-1):
        for col in range(image.shape[1]):
            new[row,col]=image[row+1,col]-image[row,col]
    new = np.absolute(new)
    return new

def computeGradient(image, kernel):
    new = np.empty((image.shape[0]-2,image.shape[1]-2))
    for row in range(1,image.shape[0]-1):
        for col in range(1,image.shape[1]-1):
            new[row-1,col-1]=kernel[0,0]*image[row-1,col-1] + kernel[0,1]*image[row-1,col] + kernel[0,2]*image[row-1,col+1] + kernel[1,0]*image[row,col-1] + kernel[1,1]*image[row,col] + kernel[1,2]*image[row,col+1] + kernel[2,0]*image[row+1,col-1] + kernel[2,1]*image[row+1,col] + kernel[2,2]*image[row+1,col+1]  
    return new

