import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML




def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
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

            if slope < 0:
              left_avg = left_avg+slope
              left_pts.append((x1,y1))
              left_pts.append((x2,y2))
            else: 
              right_avg = right_avg+slope
              right_pts.append((x1,y1))
              right_pts.append((x2,y2))


    sorted_left = sorted(left_pts,key=lambda tup: tup[1])
    sorted_right    = sorted(right_pts,key=lambda tup: tup[1])

    left  = np.array(sorted_left)
    right = np.array(sorted_right)


    lx = left[:,0]
    ly = left[:,1]

    lz = np.polyfit(lx,ly,1)
    lf = np.poly1d(lz)

    lx_new = np.linspace(lx[0],lx[-1],50,dtype=int)
    ly_new = lf(lx_new).astype(int)

    rx = right[:,0]
    ry = right[:,1]

    rz = np.polyfit(rx,ry,1)
    rf = np.poly1d(rz)

    rx_new = np.linspace(rx[0],rx[-1],50,dtype=int)
    ry_new = rf(rx_new).astype(int)


    left_lines  = list(zip(lx_new,ly_new))
    right_lines = list(zip(rx_new,ry_new))


    for indx in range (1,50):
        cv2.line(img,left_lines[indx-1], left_lines[indx],color,thickness)
        cv2.line(img,right_lines[indx-1],right_lines[indx],color,thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    kernel_size       = 5
    low_threshold     = 90
    high_threshold    = 180
    rho               = 1 		   # distance resolution in pixels of the Hough grid
    theta             = np.pi/180 	   # angular resolution in radians of the Hough grid
    threshold         = 8    	   	   # minimum number of votes (intersections in Hough grid cell)
    min_line_length   = 10		   # minimum number of pixels making up a line
    max_line_gap      = 10   		   # maximum gap in pixels between connectable line segments
    ignore_mask_color = 255   

    #image = (mpimg.imread('test_images/whiteCarLaneSwitch.jpg')).astype('uint8')
    gray  = grayscale(image)

    blur_gray = gaussian_blur(gray, kernel_size)
    edges     = canny(blur_gray,low_threshold,high_threshold)
    imshape   = image.shape
    vertices  = np.array([[(0,imshape[0]),((imshape[1]/2)-1, imshape[0]/2),((imshape[1]/2)+1,imshape[0]/2),(imshape[1],imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    result     = weighted_img(line_image, image, 0.8, 1, 0)


    return result


################# main ##################

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
