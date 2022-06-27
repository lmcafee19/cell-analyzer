# Test File Integrating the centroid tracker class to keep track of cells as they move between frames
# Author: zheath19@georgefox.edu
import cv2 as cv
import numpy as np
import os
from enum import Enum
import centroid_tracker as ct
from collections import OrderedDict
from itertools import chain
import export_data

# Edge Detection Algorithm Enum to store valid options
class Algorithm(Enum):
    CANNY = "CANNY"
    LAPLACIAN = "LAPLACIAN"
    SOBEL = "SOBEL"

# Define Constants
PATH = '../videos/'
VIDEO = 'sample_cell_video.mp4'
SCALE = 0.25
CONTRAST = 4.0
BRIGHTNESS = .5
BLUR_INTENSITY = 75
MIN_CELL_SIZE = 50

# Elliptical Kernel
ELIPTICAL_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

# TODO Determine approx shape of Cell
# TODO Determine Direction Cells are moving
# TODO Record Data About Location and size Separately
def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{PATH}{VIDEO}'
    # Check if video exists
    if not os.path.exists(videoFile):
        raise Exception("File cannot be found")
    else:
        # Read in video frame by frame
        capture = cv.VideoCapture(videoFile)
        total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        # Initialize Centroid tracker
        tracker = ct.CentroidTracker()
        # Frame Dimensions
        (h, w) = (None, None)

        # Initialize Objects to store data on cell size and location
        # Indexed by cell ID given by centroid tracker and set to size = num frames
        cell_positions = OrderedDict()
        cell_sizes = OrderedDict()

        while True:
            valid, frame = capture.read()

            # If next frame is not found exit program
            if not valid:
                break

            processed_laplacian = process_image(frame, Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)

            # Display Proccessed Video
            cv.imshow("Laplacian", processed_laplacian)

            # Detect minimum cell boundaries and display edited photo
            cont, rectangles = detect_cell_boundries(processed_laplacian)
            cv.imshow("Contours-External", cont)

            # Update Centroid tracker with list of rectangles
            #print(f"num rectangles: {len(rectangles)}")
            cell_locations, cell_areas = tracker.update(rectangles)
            #print(f"Tracked Cells: {cells}")

            # Record Data about Cell position, and cell size
            # Record positonal data given by tracker
            for cell_id, coordinates in cell_locations.items():
                # If no entry exist for that cell create it
                if not (cell_id in cell_positions):
                    cell_positions[cell_id] = list()

                cell_positions[cell_id].append(list(coordinates))

            # Record Area
            for cell_id, area in cell_areas.items():
                # If no entry exist for that cell create it
                if not (cell_id in cell_sizes):
                    cell_sizes[cell_id] = list()

                cell_sizes[cell_id].append(area)

            print(cell_positions)
            print(cell_sizes)

            #print(cell_positions)
            #
            # # loop over the tracked cells and reformat data to array of args we want
            # for (objectID, centroid) in cells.items():
            #     # draw both the ID of the object and the centroid of the
            #     # object on the output frame
            #     text = "ID {}".format(objectID)
            #     cv.putText(processed_laplacian, text, (centroid[0] - 10, centroid[1] - 10),
            #                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     cv.circle(processed_laplacian, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            #
            # # show the output frame
            # cv.imshow("Tracked", processed_laplacian)
            #
            # # If frame diemensions aren't set then grab them
            # if w is None or h is None:
            #     (h, w) = processed_laplacian.shape[:2]

            # Adjust waitKey to change time each frame is displayed
            # Press q to exit out of opencv early
            if cv.waitKey(500) & 0xFF == ord('q'):
                break

        # Close opencv
        capture.release()
        cv.destroyAllWindows()


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
    Uses a specified edge detection algorithm to display only edges found in the image
    @:param img: image to detect edges in
    @:param edge_alg Option from Algorithm enum
    @:return edited frame/image with drawn on contours and text
'''
def detect_shape(img):
    # Epsilon value determines how exact the contour needs to follow specifications
    EPSILON = 3
    threshold_val, thrash = cv.threshold(img, 240, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        # Grab contour following shape. The True value indicates that it must be a closed shape
        approx = cv.approxPolyDP(contour, EPSILON*cv.arcLength(contour, True), True)
        # First number = index of contour
        # Draw contour in white
        # Last num = thickness of line
        cv.drawContours(img, [approx], 0, (255, 0, 0), 5)
        # Grab coordinates of contour
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        # TODO add logic to determine if shape matches that of cell
        # cv.HoughCircles()?
        # Or find center and radius
        # Write Cell label onto photo in white font
        cv.putText(img, "Cell", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return img


'''
    Places rectangles around all objects within the image that are larger MIN_CELL_SIZE
    @:param img: image to detect edges in
    @:returns edited image with drawn on rectangle boundries and text, and an array of all rectangles drawn(cell boundries)
'''
def detect_cell_boundries(img):
    rectangles = []

    # Create copy of img as to not edit the original
    photo = img.copy()

    threshold_val, thrash = cv.threshold(photo, 240, 255, cv.THRESH_BINARY)
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
            cv.putText(photo, "Cell", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            # Record all rectangles found into array
            # Convert box (list of vertices) to list of starting x coordinate, starting y, ending x, and ending y
            # to be compatible with the update function of centroid tracker
            rec_coordinates = [box[0][0], box[1][1], box[2][0], box[3][1]]
            rectangles.append(rec_coordinates)

    return photo, rectangles


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


def get_circular_kernel(diameter):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)

    return kernel


main()
