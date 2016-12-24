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

def warp_imgs (image, M):

   # Warp the image using OpenCV warpPerspective()
   wimgs = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR)	

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

def s_channel (img, thresh=(105,255)):
  
  hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
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



def combine (gradx, grady, mag, dir, color):
   
   binary_output = np.zeros_like(dir)
   #binary_output [ ((gradx == 1) | (grady == 1)) | ((mag == 1) & (dir == 1)) | (color)] = 1 
   binary_output [ (gradx == 1)   | (color==1)] = 1 

   return binary_output
 


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


def cal_undistort(image, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    udist = cv2.undistort(image, mtx, dist, None, mtx)

    return udist


def get_histogram (img):
  histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
  plt.plot(histogram)
  plt.show()
 
  return histogram

# Make a list of calibration images
cal_images = glob.glob('camera_cal/*.jpg')
test_images = glob.glob('test_images/*.jpg')

objp, imgp = prep_calib(cal_images,nx=9,ny=6)

# Points below are based on experimentation with test2.jpg
# the source is a quadrilateral drawn on test22.jpg
# assumption is that it is roughly rectangle in the final image (top view)

s1 =   [310,  650]
s2 =   [1010, 650]
s3 =   [670,  440]
s4 =   [620,  440]

d1 =  [310, 650]
d2 =  [1010,650]
d3 =  [1010, 0]
d4 =  [310,  0]

#d3 =  [1010, 30]
#d4 =  [310,  30]

#s1 =  [310, 720]
#s2 =  [400, 625]
#s3 =  [1020,625]
#s4 =  [1200,720]

#d1 =  [310, 720]
#d2 =  [310, 625]
#d3 =  [1200,625]
#d4 =  [1200,720]

src = np.float32([s1,s2,s3,s4])
dst = np.float32([d1,d2,d3,d4])

M = cv2.getPerspectiveTransform(src, dst)


#for i in range(len(wi)):
#  cv2.imshow('img',wi[i])
#  cv2.waitKey(500)

ksize = 15 

filename = 'frame44.jpg'
image      = cv2.imread(filename)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

udistort    = cal_undistort(image, objp, imgp)
#gradx = abs_sobel    (udistort, orient='x',sobel_kernel=ksize, thresh=(70,150) )
gradx = abs_sobel    (udistort, orient='x',sobel_kernel=ksize, thresh=(30,100) )
grady = abs_sobel    (udistort, orient='y',sobel_kernel=ksize, thresh=(90,150) )
mag   = mag_thresh   (udistort, sobel_kernel=ksize,            thresh=(90,150)  )
dir   = dir_threshold(udistort, sobel_kernel=ksize,            thresh=(0.7,1.2))
color = s_channel    (udistort)

combined = combine    (gradx,grady,mag,dir,color)

warped    = warp_imgs (combined, M)

plot_images ([gray, gradx,color,combined,warped],2,3)
get_histogram(warped)
