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
           #cv2.waitKey(500)

   return objpoints, imgpoints


def cal_undistort(images, objpoints, imgpoints):
    # Use cv2.calibrateCamera and cv2.undistort()
    undist = []

    for idx, fname in enumerate (images):
      print (fname)
      img  = cv2.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
      udist = cv2.undistort(img, mtx, dist, None, mtx)
      undist.append(udist)
      #plot_images ([cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cv2.cvtColor(udist,cv2.COLOR_BGR2RGB)],2,1)

    return undist


s1 =  [310, 720]
s2 =  [400, 625]
s3 =  [1020,625]
s4 =  [1200,720]


d1 =  [310, 720]
d2 =  [310, 625]
d3 =  [1200,625]
d4 =  [1200,720]

src = np.float32([s1,s2,s3,s4])
dst = np.float32([d1,d2,d3,d4])

# Make a list of calibration images
cal_images = glob.glob('camera_cal/*.jpg')
test_images = glob.glob('test_images/*.jpg')

warp_image  = 'test_images/test22.jpg'
wimg        = cv2.imread(warp_image)
plt.imshow(wimg,interpolation='nearest')
plt.show()

M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(wimg, M, (wimg.shape[1],wimg.shape[0]))

plt.imshow(warped,interpolation='nearest')
plt.show()

#nobjs, imgp = prep_calib(cal_images,nx=9,ny=6)
#uimgs      = cal_undistort(test_images, objp, imgp)


