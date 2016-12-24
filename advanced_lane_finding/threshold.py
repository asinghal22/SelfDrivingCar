import numpy as np
import matplotlib.pyplot as plt
import cv2
#from PIL import Image
import matplotlib.image as mpimg
import glob 

def plot_images(images, nx, ny):

    fig, axes = plt.subplots(nx, ny)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
       # Plot image.
       if(i >= len(images)):
         ax.set_xticks([])
         ax.set_yticks([])
         continue
       else:
         ax.set_xticks([])
         ax.set_yticks([])
         ax.imshow(images[i],cmap='gray')

    plt.show()


def prep_calib (images, nx, ny):
   # Arrays to store object points and image points from all the images.
   objp = np.zeros((nx*ny,3), np.float32)
   objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

   objpoints = [] # 3d points in real world space
   imgpoints = [] # 2d points in image plane.

   # Step through the list and search for chessboard corners
   for idx, fname in enumerate(images):
       img = cv2.imread(fname)
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       # Find the chessboard corners
       ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

       # If found, add object points, image points
       if ret == True:
           objpoints.append(objp)
           imgpoints.append(corners)

           # Draw and display the corners
           #cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
           #cv2.imshow('img', img)
           #cv2.waitKey(1500)

   return objpoints, imgpoints


def cal_undistort(images, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    undist = []

    for idx, fname in enumerate (images):
      img  = cv2.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
      udist = cv2.undistort(img, mtx, dist, None, mtx)
      undist.append(udist)
      #plot_images ([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cv2.cvtColor(udist,cv2.COLOR_BGR2RGB)],2,1)

    return undist

def warp_imgs (images, M):

   wimgs = []

   for i in range(len(images)):
     # Warp the image using OpenCV warpPerspective()
     warped = cv2.warpPerspective(images[i], M, (images[i].shape[1],images[i].shape[0]))	
     wimgs.append(warped)

   return wimgs


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.

def abs_sobel(img, orient='x', sobel_kernel=3, thresh=(0,255)):


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value

    if orient == 'x':
        abssobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    if orient == 'y':
        abssobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abssobel/np.max(abssobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    binary_output[scaled_sobel >= thresh[0]]	= 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):


    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take both Sobel x and y gradients

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 

    # Create a binary image of ones where threshold is met, zeros otherwise

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def s_channel (img, thresh=(130,255)):
  
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2] 
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= thresh[0] ) & (s_channel <= thresh[1])] = 1

  return s_binary

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):


    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the x and y gradients

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    # Here I'm suppressing annoying error messages

    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


filename = 'test2.jpg'
image      = cv2.imread(filename)

def combine (gradx, grady, mag, dir, color):
   
   binary_output = np.zeros_like(dir)
   binary_output [ ((gradx == 1) | (grady == 1)) | ((mag == 1) & (dir == 1))] = 1 

   return binary_output
 

ksize = 15 
gradx = abs_sobel    (image, orient='x',sobel_kernel=ksize, thresh=(90,150) )
grady = abs_sobel    (image, orient='y',sobel_kernel=ksize, thresh=(90,150) )
mag   = mag_thresh   (image, sobel_kernel=ksize,            thresh=(90,150)  )
dir   = dir_threshold(image, sobel_kernel=ksize,               thresh=(0.7,1.2))
color = s_channel    (image)

plot_images ([gradx,grady,mag,dir,binary_output,color],3,2)
