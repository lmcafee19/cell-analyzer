# Test File Integrating the centroid tracker class to keep track of cells as they move between frames
# Author: zheath19@georgefox.edu

import math
import numpy as np
import cv2 as cv
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from collections import OrderedDict

# Define Constants
VIDEO = '../videos/Circular_high_contrast.avi'
EXPORT_FILE = "../data/culture_data.xlsx"
SCALE = 0.25
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10
MIN_CELL_SIZE = 10
MAX_CELL_SIZE = 600

# Define Constants for cell size. These indicate the size range in which we should detect and track cells
MIN_CELL_SIZE = 10
MAX_CELL_SIZE = 600

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 5
VIDEO_WIDTH_MM = 5.5

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10

# Elliptical Kernel
#ELIPTICAL_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))


def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{VIDEO}'
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
        # mms each pixel takes up in real world space
        pixels_to_mm = None

        # Initialize Objects to store data on cell size and location
        # Indexed by cell ID given by centroid tracker and set to size = num frames
        # Each Key contains a dictionary with data about the cells x position, y position, and area)
        cell_positions_mm = OrderedDict()
        cell_sizes_mm = OrderedDict()
        positional_headers = ["Cell ID", "Initial X Position (mm)", "Initial Y Position (mm)"]
        size_headers = ["Cell ID", "Initial Size (mm^2)"]
        frame_num = 0

        while True:
            valid, frame = capture.read()

            # If next frame is not found exit program
            if not valid:
                break

            processed_canny = analysis.process_image(frame, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)

            # Display Proccessed Video
            #cv.imshow("Canny", processed_canny)

            # Detect if cell is a circle or square and grab each objects centroid and area
            shapes_img, shapes = analysis.detect_shape_v2(processed_canny, MIN_CELL_SIZE, MAX_CELL_SIZE)
            cv.imshow("SHAPES", shapes_img)

            # Detect minimum cell boundaries and display edited photo
            #cont, rectangles = analysis.detect_cell_rectangles(processed_canny)
            #cv.imshow("Contours-External", cont)

            # Use Hough Circles to find all circles within image
            #cir, circles = analysis.detect_cell_circles(processed_canny)
            #cv.imshow("Circles", cir)

            # Grab Frame's dimensions in order to convert pixels to mm
            if w is None or h is None:
                (h, w) = shapes_img.shape[:2]
                pixels_to_mm = ((VIDEO_HEIGHT_MM/h) + (VIDEO_WIDTH_MM/w))/2

            # Update Centroid tracker with list of rectangles
            #print(f"num rectangles: {len(rectangles)}")
            cell_locations, cell_areas = tracker.update(shapes)
            #print(f"Tracked Cells: {cell_locations}\nAreas: {cell_areas}")


            # Record Data about Cell position, and cell size
            # Record positional data given by tracker
            for cell_id, coordinates in cell_locations.items():
                # If no entry exist for that cell create it
                if not (cell_id in cell_positions_mm):
                    cell_positions_mm[cell_id] = list()

                # Convert coordinates to mm
                # Coordinates correspond to centroids distance from the left and top of the image
                coordinates_mm = list(coordinates)
                coordinates_mm[0] = float(coordinates_mm[0] * pixels_to_mm)
                coordinates_mm[1] = float(coordinates_mm[1] * pixels_to_mm)

                cell_positions_mm[cell_id].append(coordinates_mm)

            # Record Area
            for cell_id, area in cell_areas.items():
                # If no entry exist for that cell create it
                if not (cell_id in cell_sizes_mm):
                    cell_sizes_mm[cell_id] = list()

                # Convert area to mm^2
                area_mm = area * (pixels_to_mm**2)
                cell_sizes_mm[cell_id].append(area_mm)

            # Increment Frame Counter
            frame_num += 1

            #print(f"Locations: {cell_positions_mm}")
            #print(f"Areas: {cell_sizes_mm}")

            # Adjust waitKey to change time each frame is displayed
            # Press q to exit out of opencv early
            if cv.waitKey(150) & 0xFF == ord('q'):
                break

        # Close opencv
        capture.release()
        cv.destroyAllWindows()

        # Generate Headers
        for i in range(1, frame_num):
            size_headers.append(f"Frame {i} Size")
            positional_headers.append(f"Frame {i} X Position")
            positional_headers.append(f"Frame {i} Y Position")

        # Add Final Columns for calculations
        # positional_headers.append("Distance between Initial Position and Final Position")
        size_headers.append("Final Growth")
        size_headers.append("Largest Growth in one interval")

        # Export Data to excel sheet
        export.culture_to_excel_file(EXPORT_FILE, cell_positions_mm, cell_sizes_mm, TIME_BETWEEN_FRAMES,
                                    (VIDEO_HEIGHT_MM * VIDEO_WIDTH_MM), positional_headers, size_headers)

        # Export Graphs using Average Area and


main()
