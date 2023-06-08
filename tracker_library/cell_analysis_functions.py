'''
@brief Library File containing functions used by cell analysis/tracking scripts
    These functions are used to edit the given images/video into a beneficial format for ml or edge detection and also
    to actually detect the edges of each cell boundary
@author zheath19@georgefox.edu
'''

import math
import cv2 as cv
import numpy as np
from enum import Enum

# Edge Detection Algorithm Enum to store valid options
class Algorithm(Enum):
    CANNY = "CANNY"
    LAPLACIAN = "LAPLACIAN"
    SOBEL = "SOBEL"

# Indicates how close circles must be to the perfect height/width ratio
# Should be float between 0-1 with higher percentages meaning more leniency and therefore more shapes being declared circles
CIRCLE_RATIO_RANGE = .30


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
    @:returns edited image with shape labels, and dictionary between all centroids found and their encompassing shapes centroid
'''
def detect_shape_v2(img, min_size, max_size):
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
        if min_size < cv.contourArea(contour) < max_size:
            # Area = cv.contourArea(contour)

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

                # Calc Area
                # If this is not closed contour (closed shape detected) then record contour's area,
                # if not cv.isContourConvex(contour):
                #     area = cv.contourArea(contour)
                # else:
                # otherwise record our min created shape's area
                area = calc_area_circle(radius)

                # Record centroid and area
                centroids[centroid] = area
                # print(f"Min Circle Area: {area}")
                # print(f"This is a closed contour: {cv.isContourConvex(contour)}")
                # print(f"Contour Area: {cv.contourArea(contour)}")


                # Draw circle and label
                cv.circle(photo, centroid, int(radius), (255, 255, 255), 2)
                #cv.putText(photo, "Circle", (int(x + radius), int(y + radius)), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            else:
                # Detected contour is rectangular
                # Convert box (list of vertices) to list of starting x coordinate, starting y, ending x, and ending y
                # to be compatible with the update function of centroid tracker
                rec_coordinates = [box[0][0], box[1][1], box[2][0], box[3][1]]

                # Get Centroid from rectangle
                centroid = (int(rect[0][0]), int(rect[0][1]))

                # Calc Area and Centroid
                # If this is not closed contour (closed shape detected) then record contour's area,
                # if not cv.isContourConvex(contour):
                #     area = cv.contourArea(contour)
                # else:
                # otherwise record our min created shape's area
                #area = calc_area_rect(rec_coordinates)
                area = calc_area_eliptical(rec_coordinates)

                # Record Centroid and its area
                centroids[centroid] = area
                # print(f"Min Rect Area: {area}")
                # print(f"THis is a closed contour: {cv.isContourConvex(contour)}")
                # print(f"Contour Area: {cv.contourArea(contour)}")

                # Draw rectangle and label found onto image in white
                cv.drawContours(photo, [box], 0, (255, 255, 255), 2)
                #cv.putText(photo, "Rectangle", centroid, cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return photo, centroids


# TODO Create Function which accepts a starting centroid, min, max size and finds the cluster most closely associated with it to track
'''
    Accepts a starting centroid, min, max size and finds the cluster most closely associated with it to track
    Intended to be used in tandem with a user drawn circle on the photo, this should find the starting object that most closely resembles that circle
    Replacement of detect_shape_v2 for the first frame of the video only
    @param img
    @param centroid Tuple of (x,y) coordinates that represents the center of the user drawn circle
    @param radius Radius of user drawn circle in pixels
    @return dictionary with the key being the detected centroid and the value being the area
'''
def detect_cluster_in_circle(img, centroid, radius):
    # Create Dictionary Mapping between  detected centroid and the area of the cell
    cell_info = {}
    return cell_info


'''
Finds the initial boundary of a cell from the given coordinates(centroid) and then draws that shape onto the given image
@param first_frame First frame of the video used to determine which cell to track, edited the same way as when it had the shapes intially detected
@param point Tuple of (x,y) coordinates
@param color RGB value used to draw cell boundary. Default = white
@return Image with drawn on cell boundary
'''
def draw_initial_cell_boundary(first_frame, point:tuple, img, color=(255, 255, 255)):
    # Create copy of image as to not edit the original
    photo = img.copy()

    threshold_val, thrash = cv.threshold(first_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Grab all contours not surrounded by another contour
    contours, hierarchy = cv.findContours(thrash, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Determine which contour is closest to given point
    closest_contour = None
    for contour in contours:
        # Determines if point lies inside(1), outside(-1), or on the boundary(0) of the contour
        dist = cv.pointPolygonTest(contour, point, False)

        # If point lies on or inside contour save it and break loop
        if dist == 1 or dist == 0:
            closest_contour = contour
            break

    # If no contour is found by the given coordinate do not draw anything
    if closest_contour is not None:
        # Minimum rectangle needed to cover contour, will be angled
        # Rect format: (center(x, y), (width, height), angle of rotation)
        rect = cv.minAreaRect(closest_contour)

        # Box returns list of four tuples containing coordinates of the vertices of the rectangle
        # First value in each tuple is x, and second is y
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # Use the height/width ratio to figure out if the cell is closer to a rectangle or circle
        ratio = rect[1][1] / rect[1][0]
        # A perfect circle would have a ratio of 1, so we accept values around it
        if 1 - CIRCLE_RATIO_RANGE < ratio < 1 + CIRCLE_RATIO_RANGE:
            # Detected contour is a circle, measure it with the smallest enclosing circle
            (x, y), radius = cv.minEnclosingCircle(closest_contour)
            centroid = (int(x), int(y))
            radius = float(radius)

            # Draw circle and label
            cv.circle(photo, centroid, int(radius), color, 2)

        else:
            # Detected contour is rectangular
            # Draw rectangle found onto image in white
            cv.drawContours(photo, [box], 0, color, 2)

    return photo


'''
Uses coordinates found from the centroid tracker to label each cell with its unique id
@param img Photo to add labels onto
@param cell_coords Ordered dict containing a mapping between each cell id and its coordinates
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
Uses coordinates to outline the cell and hide everything else
@param img Photo to add labels onto
@param cell_coord a tuple of X,Y coordinates which describes the centroid of the cell
@param radius radius of the circle around which the cell resides
'''
def outline_cell(img, cell_id, cell_coord, radius):
    # Create copy of img as to not edit the original
    photo = img.copy()

    # Label the cell with its id
    cv.putText(photo, str(cell_id), (cell_coord[0] + 10, cell_coord[1] + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    try:
        # draw filled circle in white on black background as mask
        mask = np.zeros_like(photo)
        mask = cv.circle(mask, (cell_coord[0], cell_coord[1]), radius, (255, 255, 255), -1)

        # apply mask to image to hide everything outside the circle
        result = cv.bitwise_and(photo, mask)
    except:
        print("Radius too large")

    return result


'''
    Calculates the area of the given rectangle
    @param rectangle: List Containing starting x coordinate, starting y, ending x, and ending y in that order
    @return The area of the given rectangle
'''
def calc_area_rect(rectangle):
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


'''
    Calculates the area of the given Eliptical Using the formula: A = a * b * pi
    @param box: List Containing starting x coordinate, starting y, ending x, and ending y in that order
    @return The area of the given eliptical
'''
def calc_area_eliptical(box):
    # Grab length and Width
    length = box[3] - box[1]
    width = box[2] - box[0]
    # a = 1/2 of the total height
    # b = 1/2 of the total width
    area = (0.5 * length) * (0.5 * width) * math.pi
    return area


def get_circular_kernel(diameter):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel

