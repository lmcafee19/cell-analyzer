# Script for Tracking an individual Cell throughout a video of cell growth
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
PATH = '../videos/'
VIDEO = 'Sample_cell_culture_0.mp4'
EXPORT_FILE = "../data/Sample_cell_culture_data.xlsx"
SCALE = 0.25
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 150
VIDEO_WIDTH_MM = 195.9

def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{PATH}{VIDEO}'
    # Check if video exists
    if not os.path.exists(videoFile):
        raise Exception("File cannot be found")
    else:
        # Read in video
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
        cell_positions_mm = OrderedDict()
        cell_sizes_mm = OrderedDict()
        # Keep Track of our tracked cell's coordinates in pixels
        tracked_cell_coords = OrderedDict()
        positional_headers = ["Cell ID", "Initial Position"]
        size_headers = ["Cell ID", "Initial Size (mm^2)"]
        frame_num = 0

        # Open First Frame of Video and Detect all cells within it, making sure to label them
        valid, frame = capture.read()
        if not valid:
            raise Exception("Video cannot be read")

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)

        # Detect minimum cell boundaries and display edited photo
        cont, rectangles = analysis.detect_cell_rectangles(processed)

        # Use Tracker to label and record coordinates of all cells
        cell_locations, cell_areas = tracker.update(rectangles)

        # Label all cells with cell id
        labeled_img = label_cells(processed, cell_locations)

        # Display edited photo
        cv.imshow("First Frame", labeled_img)

        # Grab Frame's dimensions in order to convert pixels to mm
        (h, w) = labeled_img.shape[:2]
        pixels_to_mm = VIDEO_HEIGHT_MM / h

        # Show Frame until space bar is pressed
        k = cv.waitKey(0)
        if k == 32:
            # Ensure that user selects a valid cell id
            tracked_cell_id = -1
            while not (0 <= int(tracked_cell_id) < len(cell_locations)):
                # Allow User to select which cell they want to track
                tracked_cell_id = int(input("Select Cell to Track: "))

            # Close First Frame
            cv.destroyAllWindows()

            # Record data about tracked cell
            tracked_cell_coords[tracked_cell_id] = list()
            tracked_cell_coords[tracked_cell_id].append(list(cell_locations[tracked_cell_id]))

            # Convert area to mm^2
            cell_sizes_mm[tracked_cell_id] = list()
            area_mm = cell_areas[tracked_cell_id] * (pixels_to_mm ** 2)
            cell_sizes_mm[tracked_cell_id].append(area_mm)

            # Convert Coordinates to mm
            cell_positions_mm[tracked_cell_id] = list()
            coordinates_mm = list(cell_locations[tracked_cell_id])
            coordinates_mm[0] = float(coordinates_mm[0] * pixels_to_mm)
            coordinates_mm[1] = float(coordinates_mm[1] * pixels_to_mm)
            cell_positions_mm[tracked_cell_id].append(coordinates_mm)

            # Increment Frame Counter
            frame_num += 1

            # Loop through all frames of the video
            while True:
                valid, frame = capture.read()

                # If next frame is not found exit program
                if not valid:
                    break

                # Process Image to better detect cells
                processed = analysis.process_image(frame, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS,
                                                   BLUR_INTENSITY)

                # Detect minimum cell boundaries and display edited photo
                cont, rectangles = analysis.detect_cell_rectangles(processed)

                # Use Tracker to label and record coordinates of all cells
                cell_locations, cell_areas = tracker.update(rectangles)

                # Update Tracking information
                # Record data about tracked cell
                tracked_cell_coords[tracked_cell_id].append(list(cell_locations[tracked_cell_id]))

                # Convert area to mm^2
                area_mm = cell_areas[tracked_cell_id] * (pixels_to_mm ** 2)
                cell_sizes_mm[tracked_cell_id].append(area_mm)

                # Convert Coordinates to mm
                coordinates_mm = list(cell_locations[tracked_cell_id])
                coordinates_mm[0] = float(coordinates_mm[0] * pixels_to_mm)
                coordinates_mm[1] = float(coordinates_mm[1] * pixels_to_mm)
                cell_positions_mm[tracked_cell_id].append(coordinates_mm)

                # Draw marker at cell's initial position
                # print(tracked_cell_coords)
                # print(f"coords: {tracked_cell_coords[tracked_cell_id]}")
                cv.circle(processed, tracked_cell_coords[tracked_cell_id][0], 2, (255, 255, 255), 3)

                # Draw an arrow for every frame of movement going from its last position to its next position
                for i in range(1, len(tracked_cell_coords[tracked_cell_id])):
                    cv.arrowedLine(processed, tracked_cell_coords[tracked_cell_id][i-1], tracked_cell_coords[tracked_cell_id][i], (255, 255, 255), 3, cv.LINE_AA, 0, 0.1)

                # Display edited photo
                cv.imshow("First Frame", processed)

                # Increment Frame Counter
                frame_num += 1

                # Adjust waitKey to change time each frame is displayed
                # Press q to exit out of opencv early
                if cv.waitKey(50) & 0xFF == ord('q'):
                    break

            # Export that data to excel


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

        cv.putText(photo, str(cell_id), (coord[0], coord[1]), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return photo


main()