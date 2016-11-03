import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
# Note: in the previous example we were reading a .jpg 
# Here we read a .png and convert to 0,255 bytescale
#image = (mpimg.imread('exit_ramp.png')*255).astype('uint8')
image = (mpimg.imread('test_images/solidWhiteRight.jpg')).astype('uint8')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#plt.imshow(gray,cmap='gray')
#plt.show()

## Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
#plt.imshow(blur_gray,cmap='gray')
#plt.show()


#
## Define our parameters for Canny and apply
low_threshold = 90
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

## Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   

ignore_mask_color = 255   

## This time we are defining a four sided polygon to mask
imshape = image.shape

vertices = np.array([[(0,imshape[0]),((imshape[1]/2)-1, imshape[0]/2),((imshape[1]/2)+1,imshape[0]/2),(imshape[1],imshape[0])]], dtype=np.int32)


cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)


## Define the Hough transform parameters
## Make a blank the same size as our image to draw on

rho             = 1 		   # distance resolution in pixels of the Hough grid
theta           = np.pi/180 	   # angular resolution in radians of the Hough grid
threshold       = 8    	   	   # minimum number of votes (intersections in Hough grid cell)
min_line_length = 25 		   # minimum number of pixels making up a line
max_line_gap    = 10   		   # maximum gap in pixels between connectable line segments
line_image 	= np.copy(image)*0 # creating a blank to draw lines on

## Run Hough on edge detected image
## Output "lines" is an array containing endpoints of detected line segments

# Draw the lines on the edge image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

## Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

## Iterate over the output "lines" and draw lines on a blank image
left_avg  = 0
right_avg = 0

l = 0
r = 0
left_pts = []
right_pts= []

for line in lines: 
    for x1,y1,x2,y2 in line:
        slope = (y2 - y1)/(x2 - x1)

        if (abs(slope) < 0.4):
           continue

        #print   (x1, y1, x2, y2, slope)

        if slope < 0:
           left_avg = left_avg+slope
           left_pts.append((x1,y1))
           left_pts.append((x2,y2))
           l = l + 1
        else: 
           right_avg = right_avg+slope
           right_pts.append((x1,y1))
           right_pts.append((x2,y2))
           r = r + 1


sorted_left = sorted(left_pts,key=lambda tup: tup[1])
sorted_right = sorted(right_pts,key=lambda tup: tup[1])
left  = np.array(sorted_left)
right = np.array(sorted_right)

#plt.plot(left[:,0],left[:,1],'ro')
#plt.plot(right[:,0],right[:,1],'ro')
#plt.xlim(0,imshape[1])
#plt.ylim(0,imshape[0])
#plt.gca().invert_yaxis()
#plt.show()



lx = left[:,0]
ly = left[:,1]

lz = np.polyfit(lx,ly,1)
lf = np.poly1d(lz)

lx_new = np.linspace(lx[0],lx[-1],50)
ly_new = lf(lx_new)


	

rx = right[:,0]
ry = right[:,1]



rz = np.polyfit(rx,ry,1)
rf = np.poly1d(rz)

rx_new = np.linspace(rx[0],rx[-1],50)
ry_new = rf(rx_new)

left_lines = np.dstack((lx_new,ly_new))
right_lines = np.dstack((rx_new,ry_new))


for indx in range (1,len(left_lines)):
    cv2.line(line_image,left_lines[indx-1][0],left_lines[indx-1][1], left_lines[indx][0], left_lines[indx][1])
    cv2.line(line_image,right_lines[indx-1][0],right_lines[indx-1][1], right_lines[indx][0], right_lines[indx][1])

lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)
plt.show()

#plt.plot(lx,ly,'o',lx_new,ly_new)
#plt.plot(rx,ry,'o',rx_new,ry_new)
#plt.xlim(0,imshape[1])
#plt.ylim(0,imshape[0])
#plt.gca().invert_yaxis()
plt.show()


#left_line  = cv2.fitLine(left,cv2.DIST_L2,0,0.01,0.01)
#
#print (left_line)
#print (left_avg / l, right_avg / r)
