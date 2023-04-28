import sys  # to access the system
from collections import OrderedDict
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import skimage
from PIL import Image, ImageChops
# import dlib

"""Open and display image in python using opencv
code adapted from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/"""
# img = cv2.imread("spheroidImg.png", cv2.IMREAD_ANYCOLOR)
# # resize image because it's too big
# scale_percent = 0.75
# scaled_dim = (int(img.shape[1] * scale_percent), int(img.shape[0] * scale_percent))

# define global variables:
FRAME1 = np.zeros((1,1,1), np.uint8) # cv2.resize(img, scaled_dim, interpolation=cv2.INTER_AREA)
PREVIOUS = FRAME1.copy()
DRAWING = False  # true if mouse is pressed
IX, IY = -1, -1
START = (0, 0), 0 # centroid, radius
CIRCLE_COLOR = (200, 50, 100)
PATH_COLOR = (255, 255, 255)
START_COLOR = (255, 0, 0)
END_COLOR = (100,50,255)
SCALE = 0.2

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 0.9
VIDEO_WIDTH_MM = 1.2

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10



def main():
	"""Open video file and read each frame"""
	global FRAME1
	# Read in video
	video_file = "spheroidVideo.mp4"
	capture = cv2.VideoCapture(video_file)
	total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

	# Frame Dimensions
	(h, w) = (None, None)
	# mms each pixel takes up in real world space
	pixels_to_mm = None

	# Initialize Dictionary to store position and Area Data of tracked cell
	tracked_spheroid_data = {'Time': [], 'Centroid Coordinates (mm)': [], 'Area (mm^2)': []}
	# Keep Track of our tracked spheroid's coordinates in pixels
	# tracked_cell_coords = OrderedDict()
	frame_num = 1
	first_frame = None
	final_frame = None
	Xmin = None
	Ymin = None
	Xmax = 0
	Ymax = 0


	# Show Frame until space bar is pressed
	k = cv2.waitKey(0)
	# if k == 32:
	# open first frame
	valid, frame = capture.read()
	if not valid:
		raise Exception("Video cannot be read")

	# resize frame because it is huge
	frame = rescale_frame(frame)

	# Grab Frame's dimensions in order to convert pixels to mm
	(h, w) = frame.shape[:2]
	pixels_to_mm = ((VIDEO_HEIGHT_MM / h) + (VIDEO_WIDTH_MM / w)) / 2

	# User selects their spheroid
	FRAME1 = frame
	# update global variable START:
	select_spheroid()
	# initialize first centroid from user-drawn circle
	centroid, radius = START
	# Close First Frame
	# cv2.destroyAllWindows()

	# Loop through all frames of the video
	# loop must already have a center to start to do blurring, end with new centroid
	while frame_num < total_frames:
		# get centroid location and area
		processed = image_processing(frame, centroid, radius) # returns blob
		outline = identify_spheroid(processed) # returns largest contour
		position, area = data(frame, outline) # returns approximate center of mass and area in pixels

		# get time from start
		time = (frame_num - 1) * TIME_BETWEEN_FRAMES

		# Convert results to mm
		area = float(area * (pixels_to_mm ** 2))
		x_mm = float(position[0] * pixels_to_mm)
		y_mm = float(position[1] * pixels_to_mm)
		position_mm = (x_mm, y_mm)

		# save data
		tracked_spheroid_data['Centroid Coordinates (mm)'].append(position_mm)
		tracked_spheroid_data['Area (mm^2)'].append(area)
		tracked_spheroid_data['Time'].append(time)

		# Draw marker at cell's initial position
		cv2.circle(frame, START[0], 5, START_COLOR, -1) # circle: (image, (center_x, center_y), radius, color, thickness)
		cv2.putText(frame, "start", (int(START[0][0]) - 25, int(START[0][1]) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, START_COLOR, 2)
		# print(tracked_spheroid_data['Centroid Coordinates (mm)'])  #, tracked_spheroid_data['Area (mm^2)'], tracked_spheroid_data['Time'])

		# Draw an arrow for every frame of movement going from its last position to its next position
		if frame_num > 1:
			for i in range(1, len(tracked_spheroid_data['Centroid Coordinates (mm)'])):
				a = int(tracked_spheroid_data['Centroid Coordinates (mm)'][i - 1][0] / pixels_to_mm), int(tracked_spheroid_data['Centroid Coordinates (mm)'][i - 1][1] / pixels_to_mm)
				b = int(tracked_spheroid_data['Centroid Coordinates (mm)'][i][0] / pixels_to_mm), int(tracked_spheroid_data['Centroid Coordinates (mm)'][i][1] / pixels_to_mm)
				cv2.arrowedLine(frame, a, b, PATH_COLOR, 2, cv2.LINE_AA, 0, 0.1)

		# Display edited photo
		# cv2.imshow("Cell Tracking", frame)

		# update circle (centroid and radius of next circle to focus on spheroid)
		centroid, radius = next_circle_position(frame, outline)

		# Update frame count, move to next frame
		frame_num += 1

		# Keep track of previous frame
		final_frame = frame

		# if not the last frame display it for only a short amount of time
		if frame_num < total_frames:
			# Adjust waitKey to change time each frame is displayed
			# Press q to exit out of opencv early
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			# if on the last frame display it until the q key is pressed
			k = cv2.waitKey(0)
			if k == ord('q'):
				break

		valid, frame = capture.read()
		# if next frame is not found exit program
		if not valid:
			break

		# resize frame because it is huge
		frame = rescale_frame(frame)

	# Create Color Image containing the path the tracked cell took
	# Scale image to match
	final_photo = final_frame

	# Draw dot at Cell's starting position
	cv2.circle(final_photo, START[0], 5, START_COLOR, -1)
	first_area = tracked_spheroid_data['Area (mm^2)'][0]
	cv2.putText(final_photo, f'initial area: {first_area:.3f} mm^2', (int(START[0][0]) - 25, int(START[0][1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, START_COLOR, 2)

	# Draw a line for every frame of movement going from its last position to its next position
	for i in range(1, len(tracked_spheroid_data['Centroid Coordinates (mm)'])):
		a = int(tracked_spheroid_data['Centroid Coordinates (mm)'][i - 1][0] / pixels_to_mm), int(tracked_spheroid_data['Centroid Coordinates (mm)'][i - 1][1] / pixels_to_mm)
		b = int(tracked_spheroid_data['Centroid Coordinates (mm)'][i][0] / pixels_to_mm), int(tracked_spheroid_data['Centroid Coordinates (mm)'][i][1] / pixels_to_mm)
		cv2.arrowedLine(final_photo, a, b, PATH_COLOR, 2, cv2.LINE_AA, 0, 0.1)

	# Draw dot at final centroid
	last_position = (int(tracked_spheroid_data['Centroid Coordinates (mm)'][-1][0] / pixels_to_mm), int(tracked_spheroid_data['Centroid Coordinates (mm)'][-1][1] / pixels_to_mm))
	last_area = tracked_spheroid_data['Area (mm^2)'][-1]
	cv2.circle(final_photo, last_position, 5, END_COLOR, cv2.FILLED)
	cv2.putText(final_photo, "end", (last_position[0] - 25, last_position[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, END_COLOR, 2)
	cv2.putText(final_photo, f'final area: {last_area:.3f} mm^2', (last_position[0] - 25, last_position[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, END_COLOR, 2)

	# Print out data
	print(tracked_spheroid_data['Centroid Coordinates (mm)'], tracked_spheroid_data['Area (mm^2)'], tracked_spheroid_data['Time'])

	cv2.imshow("final", final_photo)
	cv2.waitKey(0)


# define mouse callback function to draw circle
def select_spheroid():
	# Frame 1
	# Create a window
	cv2.namedWindow('Drag Circle Window')

	# bind the callback function to the above defined window
	cv2.setMouseCallback('Drag Circle Window', draw_circle)

	# display the image
	while True:  # infinite loop, exited via explicit break
		cv2.imshow('Drag Circle Window', FRAME1)
		# 	cv2.waitKey(1)
		# 	break
		k = cv2.waitKey(0) & 0xFF  # 0xFF is a hexidecimal, helps with comparing pressed key
		if k == 27:  # hit Esc to close window (replace this with a button in gui)
			cv2.imwrite("circled_img.jpg", PREVIOUS)
			break

	cv2.destroyAllWindows()


def draw_circle(event, x, y, flags, param):
	"""Drawing circle on image based on mouse movements
	code adapted from:
	https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/
	https://www.life2coding.com/paint-opencv-images-save-image/
	https://www.tutorialspoint.com/opencv-python-how-to-draw-circles-using-mouse-events"""
	img_copy = FRAME1.copy()  # sets fresh image as canvas to clear the slate
	global IX, IY, DRAWING, PREVIOUS, START, CIRCLE_COLOR
	if event == cv2.EVENT_LBUTTONDOWN: # when left button on mouse is clicked...
		DRAWING = True
		# take note of where the mouse was located
		IX, IY = x, y
	elif event == cv2.EVENT_MOUSEMOVE:
		DRAWING = True
	elif event == cv2.EVENT_LBUTTONUP: # length dragged = diameter of circle
		radius = int((math.sqrt(((IX - x) ** 2) + ((IY - y) ** 2))) / 2)
		center_x = int((IX - x) / 2) + x
		center_y = int((IY - y) / 2) + y
		START = (center_x, center_y), radius
		cv2.circle(img_copy, START[0], START[1], CIRCLE_COLOR, thickness=2) # circle: (image, (center_x, center_y), radius, color, thickness)
		DRAWING = False
		PREVIOUS = img_copy  # sets global variable to image with circle so it can be referenced outside of this method
		cv2.imshow('Drag Circle Window', img_copy)


def image_processing(image, centroid, radius):
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
	# declare local constants
	gaus_kernel = (9,9) # must be odd and positive
	med_kernel_size = 21 # must be odd and positive, always a square so only one value
	intensity = 1
	halo_multiplier = 1.5

	# Make light blur around whole background, such that unselected cells are still identifiable:
	# (this is to catch parts of spheroid that got cut off by the drawn circle, and those in next frame)
	# create a white image that is the same size as the spheroid image
	mask = (np.ones(image.shape, dtype="uint8"))*255
	# create a filled black circle on the mask
	cv2.circle(mask, centroid, radius, 0, -1) # (image, (center_x, center_y), radius, color, thickness)
	# apply light gaussian blur to entire original image
	# cv2.imshow("original", image)
	light_blur = cv2.GaussianBlur(image, gaus_kernel, 1)
	# paste blurred image on white section of mask (background) and untouched image in black circle in mask (selected)
	blur1 = np.where(mask > 0, light_blur, image)
	# cv2.imshow("first blur - background", blur1)

	# Create stronger blur in a halo around non-blurred region
	# make new mask with bigger circle
	mask2 = (np.ones(image.shape, dtype="uint8"))*255
	cv2.circle(mask2, centroid, int(radius * halo_multiplier), 0, -1)
	# apply stronger median blur to white regions of mask2 (hide background contours)
	strong_blur = cv2.medianBlur(image, med_kernel_size)
	# paste strong blur onto white region of mask2, fill black circle of mask2 with the first blurred image
	blur2 = np.where(mask2 > 0, strong_blur, blur1)
	blur2 *= intensity # multiplied by an int as effective means of increasing contrast
	# cv2.imshow("halo", blur2)

	# Merge cell shapes into one shape, return this image
	# median blur over processed image to create "shadowed" region where spheroid is
	blob = cv2.medianBlur(blur2, med_kernel_size)
	# cv2.imshow("lumped", blob)
	return blob


def identify_spheroid(processed: np.ndarray) -> np.ndarray:
	"""Detect outline of spheroid
	Code adapted from:
	https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
	https://stackoverflow.com/questions/56754451/how-to-connect-the-ends-of-edges-in-order-to-close-the-holes-between-them
	https://towardsdatascience.com/edges-and-contours-basics-with-opencv-66d3263fd6d1"""
	# declare local constants
	t_lower = 8  # canny edge detection Lower Threshold parameter, higher number -> less lines show up
	t_upper = 12  # canny edge detection Upper threshold parameter
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # np.ones((12, 12), np.uint8)

	# apply the Canny Edge filter, convert to black and white image
	edges = cv2.Canny(processed, t_lower, t_upper)
	# cv2.imshow("canny", edges)

	# connect lines from canny
	smooth = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	s2 = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel)
	# cv2.imshow("smooth", s2)

	# find contours in the binary image
	contours, hierarchy = cv2.findContours(smooth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# select longest contour (spheroid outline)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	# use to locate centroid
	return contours[0]



def data(image, outline):
	"""Run calculations and get data for each frame
	find approximate center of mass, centroid, radius, and area
	save/track center of mass and area
	pass along centroid and radius for creation of new blur circles on next frame
	Code adapted from:
	https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
	https://www.tutorialspoint.com/how-to-find-the-minimum-enclosing-circle-of-an-object-in-opencv-python"""
	global CIRCLE_COLOR

	# Find approximate center of mass (CoM)
	M = cv2.moments(outline)
	CoM_x = int(M["m10"] / M["m00"])
	CoM_y = int(M["m01"] / M["m00"])
	CoM = (CoM_x, CoM_y)

	# display results for testing purposes:
	cv2.circle(image, CoM, 8, CIRCLE_COLOR, -1)
	cv2.putText(image, "center of mass", (CoM_x - 50, CoM_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CIRCLE_COLOR, 2)
	cv2.drawContours(image, outline, -1, CIRCLE_COLOR, 2)

	# Find area
	area = cv2.contourArea(outline)

	# display results for testing purposes:
	# cv2.putText(image, "Area: " + str(area), (CoM_x - 50, CoM_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CIRCLE_COLOR, 2)
	# cv2.imshow("result", image)
	return CoM, area

def next_circle_position(image, outline):
	# Update radius and centroid
	# draw a bounding circle around the spheroid
	(centroid_x, centroid_y), radius = cv2.minEnclosingCircle(outline)
	centroid = int(centroid_x), int(centroid_y)
	radius = int(radius)

	# display results for testing purposes:
	cv2.circle(image, centroid, radius, (0, 0, 0), thickness=2)
	cv2.circle(image, centroid, 5, (0, 0, 0), -1)
	# cv2.putText(image, "centroid to pass to next frame", (int(centroid_x) - 75, int(centroid_y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	cv2.imshow("final", image)
	# cv2.waitKey(0)

	return centroid, radius


def rescale_frame(frame):
	"""Resizes the given video frame or image to supplied scale
		@:param frame image or singular frame of video
		@:param scale (0.0,  inf) with 1.0 leaving the scale as is
		@:return resized frame"""
	global SCALE
	width = int(frame.shape[1] * SCALE)
	height = int(frame.shape[0] * SCALE)
	dimensions = (width, height)

	return cv2.resize(frame, dimensions, cv2.INTER_AREA)


#
"""read in video one frame at a time, first frame entered manually, circle drawn/edited before loop

cv.videocapture
initialize tracker
dicts to hold coordinates, use distance and area to find next object
dilate after canny processing
pass data thru tracker again

test without blur/mask, just passing centroid"""


main()
