import sys  # to access the system
import cv2
import numpy as np
import math

"""Open and display image in python using opencv
code adapted from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/"""
img = cv2.imread("spheroidImg.png", cv2.IMREAD_ANYCOLOR)
# resize image because it's too big
scale_percent = 0.5
scaled_dim = (int(img.shape[1] * scale_percent), int(img.shape[0] * scale_percent))

# define global variables:
image = cv2.resize(img, scaled_dim, interpolation=cv2.INTER_AREA)
previous = image.copy()
drawing = False  # true if mouse is pressed
ix, iy = -1, -1
centroid = 0, 0
radius = 1


def main():
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

	mask()

# define mouse callback function to draw circle
def draw_circle(event, x, y, flags, param):
	"""Drawing circle on image
	code adapted from:
	https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/
	https://www.life2coding.com/paint-opencv-images-save-image/
	https://www.tutorialspoint.com/opencv-python-how-to-draw-circles-using-mouse-events"""
	img_copy = image.copy()  # sets fresh image as canvas to clear the slate
	global ix, iy, drawing, previous, centroid, radius
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
		color = (255, 100, 200)
		cv2.circle(img_copy, centroid, radius, color, thickness=1)
		drawing = False
		previous = img_copy  # sets global variable to image with circle so it can be referenced outside of this method
		cv2.imshow('Drag Circle Window', img_copy)


def mask():
	"""Create a mask of spheroid (drawn circle) and black out surroundings
	this will allow for better tracking of cells in spheroid.
	Code adapted from:
	https://www.digitalocean.com/community/tutorials/arithmetic-bitwise-and-masking-python-opencv
	https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python
	"""
	global previous, centroid, radius
	# create a black image that is the same size as the spheroid image
	mask = np.zeros(previous.shape[:2], dtype="uint8")

	# creating a white circle on the mask
	cv2.circle(mask, centroid, radius, 255, -1) # (image, (center_x, center_y), radius, color, thickness)


	# performing a bitwise_and with the image and the mask
	masked = cv2.bitwise_and(previous, previous, mask=mask)
	cv2.imshow("Mask applied to Image", masked)
	cv2.waitKey(0)



main()



#
# cv2.namedWindow('New Window')
# cv2.imshow('New Window', previous)
# cv2.waitKey(0)

# cv2.waitKey(0)
  # to exit from all the processes


# TODO: Figure out new tracking method

"""Create new image with gaussian blur"""
# def gauss():
#  	circled = cv2.imread("circled_img.jpg")
#  	cv2.imshow(circled)
# sigma = 3.0
#
# blurred = skimage.filters.gaussian(
#     image, sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
#
# # display blurred image
# # fig, ax = plt.subplots()
# # plt.imshow(blurred)