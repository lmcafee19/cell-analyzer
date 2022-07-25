# Library File containing functions used by cell analysis/tracking scripts
# Author: zheath19@georgefox.edu

import math
import cv2 as cv
import numpy as np
from enum import Enum

# Edge Detection Algorithm Enum to store valid options
class Algorithm(Enum):
    CANNY = "CANNY"
    LAPLACIAN = "LAPLACIAN"
    SOBEL = "SOBEL"


# Define Constants for cell size. These indicate the size range in which we should detect and track cells
MIN_CELL_SIZE = 10
MAX_CELL_SIZE = 600

# Indicates how close circles must be to the perfect height/width ratio
# Should be float between 0-1 with higher percentages meaning more leniency and therefore more shapes being declared circles
CIRCLE_RATIO_RANGE = .30

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 150
VIDEO_WIDTH_MM = 195.9

'''
    Seperates given video or multi frame file into array of frames
'''
def read_video(file):
    # Determine file extension
    # if .tif must use imreadmulti
    if file.endswith(".tif"):
        ret, capture = cv.imreadmulti(file, [], cv.IMREAD_ANYCOLOR)
    else:
        capture = cv.VideoCapture(file)

    return capture


'''
    Adjusts image to better view individual cells
    This involves converting the image to grayscale, increasing contrast,
    applying filters, and applying the canny edge detector to bring out the edges of each cell
    @:param img Image to adjust
    @:param edge_alg Option from Algorithm enum
    @:param scale:   (0.0,  inf) with 1.0 leaving the scale as is
    @:param contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    @:param brightness: [-255, 255] with 0 leaving the brightness as is
    @:param blur: [0, 250] Blur Intensity
    @:return adjusted frame/image
'''
def process_image(img, edge_alg, scale:float=1.0, contrast:float=1.0, brightness:int=0, blur:int=0):
    # Scale image
    processed = rescale_frame(img, scale)
    # Convert to grayscale
    processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    # Increase contrast
    processed = adjust_contrast_brightness(processed, contrast, brightness)
    # Apply Bilateral Filter to Blur and reduce noise
    processed = cv.bilateralFilter(processed, 5, blur, blur)
    # Use Contrast Limited Adaptive histogram equalization
    clahe = cv.createCLAHE(2.0, (8, 8))
    processed = clahe.apply(processed)
    # Use Edge Detection Algorithm
    processed = detect_edges(processed, edge_alg)
    return processed


'''
    Adjusts image to better view individual cells
    This involves increasing contrast, applying filters, 
    and applying an edge detector to bring out the edges of each cell. 
    This does not convert it to grayscale
    @:param img Image to adjust
    @:param edge_alg Option from Algorithm enum
    @:param scale:   (0.0,  inf) with 1.0 leaving the scale as is
    @:param contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    @:param brightness: [-255, 255] with 0 leaving the brightness as is
    @:param blur: [0, 250] Blur Intensity
    @:return adjusted frame/image
'''
def process_color_image(img, edge_alg, scale:float=1.0, contrast:float=1.0, brightness:int=0, blur:int=0):
    # Scale image
    processed = rescale_frame(img, scale)
    # Increase contrast
    processed = adjust_contrast_brightness(processed, contrast, brightness)
    # Apply Bilateral Filter to Blur and reduce noise
    processed = cv.bilateralFilter(processed, 5, blur, blur)
    # Use Edge Detection Algorithm
    processed = color_canny(processed)
    return processed


'''
    Resizes the given video frame or image to supplied scale
    @:param frame image or singular frame of video
    @:param scale (0.0,  inf) with 1.0 leaving the scale as is
    @:return resized frame
'''
def rescale_frame(frame, scale:float=1.0):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, cv.INTER_AREA)


'''
    Adjusts contrast and brightness of an uint8 image
    @:param contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    @:param brightness: [-255, 255] with 0 leaving the brightness as is
    @:return adjusted frame/image
'''
def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    brightness += int(round(255*(1-contrast)/2))
    return cv.addWeighted(img, contrast, img, 0, brightness)


'''
    Uses a specified edge detection algorithm to display only edges found in the image
    @:param img: image to detect edges in
    @:param edge_alg Option from Algorithm enum
    @:return edited frame/image
'''
def detect_edges(img, edge_alg):
    processed = img
    # Type checking
    if not isinstance(edge_alg, Algorithm):
        raise TypeError('Given Edge detection algorithm must be defined by Algorithm Enum')

    # Canny Edge Detection
    if edge_alg.value == "CANNY":
        processed = cv.Canny(img, 100, 150)
    # Laplacian Derivative
    elif edge_alg.value == "LAPLACIAN":
        processed = cv.Laplacian(img, cv.CV_8UC3)
        processed = np.uint8(np.absolute(processed))
    # Sobel XY
    elif edge_alg.value == "SOBEL":
        sobel = cv.Sobel(img, cv.CV_64F, 1, 1, 5)
        processed = cv.convertScaleAbs(sobel)

    # Dilate frame to thicken and define edges
    processed = cv.dilate(processed, (7, 7), iterations=1)

    # Erode. Can be used on dilated image to sharpen lines
    # processes = cv.erode(processed, (7, 7), 1)

    return processed


'''
    Uses canny edge detection algorithm to display only edges found in the color image
    @:param img: image to detect edges in
    @:return edited frame/image
'''
def color_canny(img):
    # Use canny edge detection on each color channel seperately
    (B, G, R) = cv.split(img)
    B_canny = cv.Canny(B, 50, 200)
    #cv.imshow("B", B_canny)
    G_canny = cv.Canny(G, 50, 200)
    #cv.imshow("G", G_canny)
    R_canny = cv.Canny(R, 50, 200)
    #cv.imshow("R", R_canny)

    # Recombine seperate color channels
    processed = cv.merge([B_canny, G_canny, R_canny])

    # Dilate frame to thicken and define edges
    processed = cv.dilate(processed, (7, 7), iterations=1)

    return processed


'''
    Detects objects within the image and determines if they are circles or rectangles
    @:param img: image to detect edges in
    @:return edited image with shape labels
'''
def detect_shape(img):
    photo = img.copy()
    # Epsilon value determines how exact the contour needs to follow specifications
    EPSILON = 3
    #threshold_val, thrash = cv.threshold(photo, 240, 255, cv.THRESH_BINARY)
    ret, thrash = cv.threshold(photo, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    for contour, tree in zip(contours, hierarchy):
        # Grab contour following shape. The True value indicates that it must be a closed shape
        approx = cv.approxPolyDP(contour, EPSILON*cv.arcLength(contour, True), True)
        # Throw out all contours under specified size (Helps reduce tracking of noise)
        if MIN_CELL_SIZE < cv.contourArea(contour) < MAX_CELL_SIZE:
            # Do not display inner-most contours. This will avoid tracking organelles or multiple lines on the same cell
            if not tree[2] < 0:
                # First number = index of contour
                # Draw contour in white
                # Last num = thickness of line
                cv.drawContours(photo, [approx], 0, (255, 255, 255), 10)
                # Grab coordinates of contour
                x = approx.ravel()[0]
                y = approx.ravel()[1]

                # Use circumference of contour to calculate its area
                circumference = cv.arcLength(contour, True)
                true_area = cv.contourArea(contour)

                # Radius = C/2pi
                radius = circumference/(2 * math.pi)
                # A = pi r^2
                calc_area = math.pi * (radius**2)

                # Find smallest circle that encompasses each Cell
                (small_x, small_y), radius = cv.minEnclosingCircle(contour)
                center = (int(small_x), int(small_y))
                smallest_r = int(radius)
                cv.circle(photo, center, smallest_r, (255, 255, 255), 2)
                min_circle_area = math.pi * (smallest_r**2)

                # If a contour's calculated minimum spanning circle's area is within a specified percentage of the calculated
                # area then assume its a circle
                if (calc_area * .80) < min_circle_area < (calc_area * 1.20):
                    #print("Circle")
                    cv.putText(photo, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
                # Otherwise use rectangle
                else:
                    #print("Rectangle")
                    cv.putText(photo, "Rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

        # cv.HoughCircles()?
        # Or find center and radius
        # Write Cell label onto photo in white font
        #cv.putText(photo, "Cell", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))



    return photo


'''
    Places rectangles around all objects within the image that are larger MIN_CELL_SIZE
    @:param img: image to detect edges in
    @:returns edited image with drawn on rectangle boundries and text, and an array of all rectangles drawn(cell boundries)
'''
def detect_cell_rectangles(img):
    rectangles = []

    # Create copy of img as to not edit the original
    photo = img.copy()

    threshold_val, thrash = cv.threshold(photo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Grab all contours not surrounded by another contour
    contours, hierarchy = cv.findContours(thrash, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(photo.shape[:2], dtype=np.uint8)
    # Draw each contour found onto a black mask if they are large enough to be considered cells
    for contour in contours:
        if cv.contourArea(contour) > MIN_CELL_SIZE:
            # Straight Rectangle. Approximation of contour and does not take into account angle
            # x, y, w, h = cv.boundingRect(contour)
            # cv.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)

            # Minimum rectangle needed to cover contour, will be angled
            rect = cv.minAreaRect(contour)
            # Box returns list of four tuples containing coordinates of the vertices of the rectangle
            # First value in each tuple is x, and second is y
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # Draw rectangle found onto image in white
            cv.drawContours(photo, [box], 0, (255, 255, 255), 2)

            # Find smallest circle that encompasses each Cell
            # (x, y), radius = cv.minEnclosingCircle(contour)
            # center = (int(x), int(y))
            # radius = int(radius)
            # cv.circle(photo, center, radius, (255, 255, 255), 2)

            # Write Cell label onto each boundry
            x = contour.ravel()[0]
            y = contour.ravel()[1]
            #cv.putText(photo, "Cell", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            # Record all rectangles found into array
            # Convert box (list of vertices) to list of starting x coordinate, starting y, ending x, and ending y
            # to be compatible with the update function of centroid tracker
            rec_coordinates = [box[0][0], box[1][1], box[2][0], box[3][1]]
            rectangles.append(rec_coordinates)

    return photo, rectangles


'''
    Places circles around all cells within the image that are larger MIN_CELL_SIZE
    Uses Hough circles algorithm to find and draw them
    @:param img: preprocessed image with edges clearly extracted using a method such as canny
    @:returns edited image with drawn on circles and text, and an array of all circles drawn (containing radius)
'''
def detect_cell_circles(img):
    # Create copy of img as to not edit the original
    photo = img.copy()

    # Might need to apply gaussian blur
    #photo = cv.GaussianBlur(photo, (7, 7), 1.5)

    # Detect Circles
    # Circles contains arrays for each circle detected with x coordintate of center, y coordinate of center, and radius length
    circles = cv.HoughCircles(photo, cv.HOUGH_GRADIENT, 1, 40, param1=50, param2=10, minRadius=4, maxRadius=20)
    #circles = cv.HoughCircles(photo, cv.HOUGH_GRADIENT_ALT, 1, 40, param1=300, param2=.85 , minRadius=0, maxRadius=0)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image
            cv.circle(photo, (x, y), r, (255, 255, 255), 2)
            # then draw a rectangle corresponding to the center of the circle
            cv.rectangle(photo, (x - 2, y - 2), (x + 2, y + 2), (255, 255, 255), -1)
            # Label Each Cell
            cv.putText(photo, "Cell", (x + r, y + r), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return photo, circles


'''
    Detects objects within the image and determines if they are circles or rectangles
    @:param img: image to detect edges in
    @:returns edited image with shape labels, and dictionary between all centroids found and their encompassing shapes centroid
'''
def detect_shape_v2(img):
    # Create Dictionary Mapping detected centroids to their area
    centroids = {}

    # Create copy of img as to not edit the original
    photo = img.copy()

    threshold_val, thrash = cv.threshold(photo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Grab all contours not surrounded by another contour
    contours, hierarchy = cv.findContours(thrash, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop through each contour if they are large enough to be considered cells
    for contour in contours:
        # Filter out all contours not in specified range
        if MIN_CELL_SIZE < cv.contourArea(contour) < MAX_CELL_SIZE:

            # Minimum rectangle needed to cover contour, will be angled
            # Rect format: (center(x, y), (width, height), angle of rotation)
            rect = cv.minAreaRect(contour)

            # Box returns list of four tuples containing coordinates of the vertices of the rectangle
            # First value in each tuple is x, and second is y
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # Use the height/width ratio to figure out if the cell is closer to a rectangle or circle
            ratio = rect[1][1]/rect[1][0]
            # A perfect circle would have a ratio of 1, so we accept values around it
            if 1 - CIRCLE_RATIO_RANGE < ratio < 1 + CIRCLE_RATIO_RANGE:
                # Detected contour is a circle, measure it with the smallest enclosing circle
                (x, y), radius = cv.minEnclosingCircle(contour)
                centroid = (int(x), int(y))
                radius = float(radius)

                # Grab Area
                area = calc_area_circle(radius)
                # Grab Centroid
                centroids[centroid] = area

                # Draw circle and label
                cv.circle(photo, centroid, int(radius), (255, 255, 255), 2)
                cv.putText(photo, "Circle", (int(x + radius), int(y + radius)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            else:
                # Detected contour is rectangular
                # Convert box (list of vertices) to list of starting x coordinate, starting y, ending x, and ending y
                # to be compatible with the update function of centroid tracker
                rec_coordinates = [box[0][0], box[1][1], box[2][0], box[3][1]]

                # Get Centroid from rectangle
                centroid = (int(rect[0][0]), int(rect[0][1]))

                # Grab Area and Centroid
                area = calc_rect_area(rec_coordinates)
                centroids[centroid] = area

                # Draw rectangle and label found onto image in white
                cv.drawContours(photo, [box], 0, (255, 255, 255), 2)
                cv.putText(photo, "Rectangle", centroid, cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return photo, centroids


'''
Uses coordinates found from the centroid tracker to label each cell with its unique id
@param img Photo to add labels onto
@param cell_coors Ordered dict containing a mapping between each cell id and its coordinates
'''
def label_cells(img, cell_coords):
    # Create copy of img as to not edit the original
    photo = img.copy()

    # Unpack Dictionary to grab coordinates
    # Record initial position / data needed to redraw outline
    for cell_id, coordinates in cell_coords.items():
        coord = tuple(coordinates)

        cv.putText(photo, str(cell_id), (coord[0] + 10, coord[1] + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return photo


'''
    Calculates the area of the given rectangle
    @param rectangle: List Containing starting x coordinate, starting y, ending x, and ending y in that order
    @return The area of the given rectangle
'''
def calc_rect_area(rectangle):
    # Grab length and Width
    length = rectangle[3] - rectangle[1]
    width = rectangle[2] - rectangle[0]
    # A = l * w
    area = length * width
    return area


'''
    Calculates the area of the given circle
    @param radius radius of the circle
    @return The area of the circle
'''
def calc_area_circle(radius):
    # A = 2 * pi * r^2
    area = 2 * math.pi * (radius ** 2)
    return area


def get_circular_kernel(diameter):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel

