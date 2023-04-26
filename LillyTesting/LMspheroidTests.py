import sys  # to access the system
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage
from PIL import Image, ImageChops
# import dlib

"""Open and display image in python using opencv
code adapted from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/"""
img = cv2.imread("spheroidImg.png", cv2.IMREAD_ANYCOLOR)
# resize image because it's too big
scale_percent = 0.75
scaled_dim = (int(img.shape[1] * scale_percent), int(img.shape[0] * scale_percent))

# define global variables:
image = cv2.resize(img, scaled_dim, interpolation=cv2.INTER_AREA)
previous = image.copy()
drawing = False  # true if mouse is pressed
ix, iy = -1, -1
centroid = 0, 0
radius = 1
color = (255, 100, 200)



def main():
	# Frame 1
	# Create a window
	cv2.namedWindow('Drag Circle Window')

	# bind the callback function to above defined window
	cv2.setMouseCallback('Drag Circle Window', draw_circle)

	# display the image
	while True:  # infinite loop, exited via explicit break
		cv2.imshow('Drag Circle Window', image)
		# 	cv2.waitKey(1)
		# 	break
		k = cv2.waitKey(0) & 0xFF  # 0xFF is a hexidecimal, helps with comparing pressed key
		if k == 27:  # hit Esc to close window (replace this with a button in gui)
			cv2.imwrite("circled_img.jpg", previous)
			break

	cv2.destroyAllWindows()
	image_processing()
	# identify_spheroid()

# define mouse callback function to draw circle
def draw_circle(event, x, y, flags, param):
	"""Drawing circle on image
	code adapted from:
	https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/
	https://www.life2coding.com/paint-opencv-images-save-image/
	https://www.tutorialspoint.com/opencv-python-how-to-draw-circles-using-mouse-events"""
	img_copy = image.copy()  # sets fresh image as canvas to clear the slate
	global ix, iy, drawing, previous, centroid, radius, color
	if event == cv2.EVENT_LBUTTONDOWN: # when left button on mouse is clicked...
		drawing = True
		# take note of where the mouse was located
		ix, iy = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		drawing = True
	elif event == cv2.EVENT_LBUTTONUP: # length dragged = diameter of circle
		radius = int((math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2))) / 2)
		center_x = int((ix - x) / 2) + x
		center_y = int((iy - y) / 2) + y
		centroid = center_x, center_y # can possibly use this for machine learning later
		cv2.circle(img_copy, centroid, radius, color, thickness=2)
		drawing = False
		previous = img_copy  # sets global variable to image with circle so it can be referenced outside of this method
		cv2.imshow('Drag Circle Window', img_copy)

def image_processing():
	"""Create a mask of spheroid (drawn circle) and blur out surroundings
	to allow for better tracking of cells in spheroid.
	Code adapted from:
	https://www.digitalocean.com/community/tutorials/arithmetic-bitwise-and-masking-python-opencv
	https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python
	blurring:
	https://stackoverflow.com/questions/73035362/how-to-draw-a-blurred-circle-in-opencv-python
	https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
	https://theailearner.com/2019/05/06/gaussian-blurring/
	https://theailearner.com/tag/cv2-medianblur/
	"""
	global image, centroid, radius
	# Make light blur around whole background, such that unselected cells are still identifiable:
	# (this is to catch parts of spheroid that got cut off by the drawn circle, and those in next frame)
	# create a white image that is the same size as the spheroid image
	mask = (np.ones(image.shape, dtype="uint8"))*255
	# create a black circle on the mask
	cv2.circle(mask, centroid, radius, 0, -1) # (image, (center_x, center_y), radius, color, thickness)
	# apply light gaussian blur to entire original image
	cv2.imshow("original", image)
	light_blur = cv2.GaussianBlur(image, (11,11), 1)
	# paste blurred image on white section of mask (background) and untouched image in black circle in mask (selected)
	blur1 = np.where(mask > 0, light_blur, image)
	cv2.imshow("first blur - background", blur1)
	# Create stronger blur in a halo around non-blurred region
	# make new mask with bigger circle
	mask2 = (np.ones(image.shape, dtype="uint8"))*255
	cv2.circle(mask2, centroid, int(radius * 1.5), 0, -1)
	# apply stronger median blur to white regions of mask2 (hide background contours)
	strong_blur = cv2.medianBlur(image, 21) # kernel size for medianBlur must be odd and >0
	# paste strong blur onto white region of mask2, fill black circle of mask2 with the first blurred image
	blur2 = np.where(mask2 > 0, strong_blur, blur1)
	blur2 *= 2 # multiplied by 2 as effective means of increasing contrast
	cv2.imshow("halo", blur2)
	# Merge cell shapes into one shape
	# median blur over processed image to create "shadowed" region where spheroid is
	blob = cv2.medianBlur(blur2, 21)
	cv2.imshow("lumped", blob)
	identify_spheroid(blob)
	# cv2.waitKey(0)



def identify_spheroid(processed):
	"""Detect outline of spheroid
	Code adapted from:
	https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	https://stackoverflow.com/questions/56754451/how-to-connect-the-ends-of-edges-in-order-to-close-the-holes-between-them
	https://towardsdatascience.com/edges-and-contours-basics-with-opencv-66d3263fd6d1"""
	global image, color
	# set canny edge detection parameter values
	t_lower = 15  # Lower Threshold
	t_upper = 30  # Upper threshold
	# apply the Canny Edge filter, convert to black and white image
	edges = cv2.Canny(processed, t_lower, t_upper)
	cv2.imshow("canny", edges)
	# connect lines from canny
	kernel = np.ones((12, 12), np.uint8)
	smooth = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)
	cv2.imshow("smoothed", smooth)
	# find contours in the binary image
	contours, hierarchy = cv2.findContours(smooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# select longest contour (spheroid outline)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)
	# use to locate centroid
	analysis(contours[0])



def analysis(outline):
	"""Run calculations and get data for each frame
	find approximate center of mass, centroid, radius, and area
	save/track center of mass and area
	pass along centroid and radius for creation of new blur circles on next frame
	Code adapted from:
	https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
	https://www.tutorialspoint.com/how-to-find-the-minimum-enclosing-circle-of-an-object-in-opencv-python"""
	global image, centroid, color, radius
	# Find approximate center of mass (CoM)
	M = cv2.moments(outline)
	center_x = int(M["m10"] / M["m00"])
	center_y = int(M["m01"] / M["m00"])
	CoM = center_x, center_y
	cv2.circle(image, CoM, 5, color, -1)
	cv2.putText(image, "center of mass", (center_x - 50, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.drawContours(image, outline, -1, color, 2)
	# Find area
	area = cv2.contourArea(outline)
	cv2.putText(image, "Area: " + str(area), (center_x - 50, center_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.imshow("result", image)
	# TODO: Save/track location of CoM and area
	# Update radius and centroid
	# draw a bounding circle around the spheroid
	(centroid_x, centroid_y), radius = cv2.minEnclosingCircle(outline)
	centroid = int(centroid_x), int(centroid_y)
	cv2.circle(image, centroid, int(radius), (0, 0, 0), thickness=2)
	cv2.circle(image, centroid, 5, (0, 0, 0), -1)
	cv2.putText(image, "centroid to pass to next frame", (center_x - 75, center_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	cv2.imshow("final", image)
	cv2.waitKey(0)

	# min enclosing circle, rectangle, eliptical opencv
	# cv2 threshhold to find contour, do min enclosing around largest




	# cv2.namedWindow('Contoured cells Window')
	# bw = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
	# ret, thresh = cv2.threshold(bw, 127, 255, 0)
	# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(image, contours, -1, (0,255,0), 3) # place on original image
	# cv2.imshow('Contoured cells Window', image)
	"""Track spheroid"""
	# draw circle around initial spheroid in first frame, detect cells as objects, outline and dilate to form single object
	# for each following frame, use previous circle and start with everything inside, apply same outline/dilation as before
	#  then generate a new circle to outline the new shape created, pass this to next frame


#main()
# performing a bitwise_and with the image and the mask, places image on white portion (background)
	# cv2.imshow("Mask", mask)
	# mask = cv2.GaussianBlur(mask, (17, 17), 7)
	# cv2.imshow("blurred mask", mask)
	# alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
	# cv2.imshow("alpha", alpha)
	# blended = cv2.convertScaleAbs(image * (1 - alpha) + image * alpha)
	# cv2.imshow("blended alpha", blended)
	# masked = cv2.bitwise_and(previous, previous, mask=mask)
	# cv2.imshow("Mask applied to Image", masked)

# TODO: Figure out new tracking method
# open source video
# objective 1: track the entire spheroid's position in each frame (highlight with a circle throughout to test) and return map of path tracked and stats about it (direction, velocity, etc.)
# objective 2: record/calculate stats on number of cells, area of spheroid, pass to existing functions?
# questions:
# track individual cells within mask (pick out individual cells) or track as one object (mush cells into one blob)? Both?
# what are the current methods for keeping track of an individual cell from one frame to next?
# ideas:
# identifying spheroid as one object: edge detection of masked image, dilation until cells all stick together as one object
# for tracking: use circle from previous frame?
# loop through each frame
# 	for first frame, apply mask, then identify cells within to track
#
"""read in video one frame at a time, first frame entered manually, circle drawn/edited before loop

cv.videocapture
initialize tracker
dicts to hold coordinates, use distance and area to find next object
dilate after canny processing
pass data thru tracker again

test without blur/mask, just passing centroid"""




#
# cv2.namedWindow('New Window')
# cv2.imshow('New Window', previous)
# cv2.waitKey(0)

# cv2.waitKey(0)
  # to exit from all the processes




"""Create new image with gaussian blur
https://leslietj.github.io/2020/08/05/Gradual-Gaussian-Blur-Using-OpenCV/"""
# def gauss():
# 	# cv2.namedWindow("blurred")
# 	circled = cv2.imread("circled_img.jpg")
# 	# cv2.imshow(circled)
# 	sigma = 3.0
#
# 	blurred = skimage.filters.gaussian(
# 	circled, sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
#
# 	#display blurred image
# 	fig, ax = plt.subplots()
# 	plt.imshow(blurred)

main()
