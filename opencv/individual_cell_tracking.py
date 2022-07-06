# Script for Tracking an individual Cell throughout a video of cell growth
# Author: zheath19@georgefox.edu

import cv2 as cv
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing
from collections import OrderedDict

# Define Constants
PATH = '../videos/'
VIDEO = 'Sample_cell_culture_0.mp4'
EXCEL_FILE = "../data/Individual_cell_data.xlsx"
PDF_FILE = "../data/"
SCALE = 0.25
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 150
VIDEO_WIDTH_MM = 195.9

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10

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

        # Initialize Dictionary to store position and Area Data of tracked cell
        tracked_cell_data = {'Time': [0], 'X Position (mm)': [], 'Y Position (mm)': [], 'Area (mm^2)': []}
        # Keep Track of our tracked cell's coordinates in pixels
        tracked_cell_coords = OrderedDict()
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
            area_mm = cell_areas[tracked_cell_id] * (pixels_to_mm ** 2)
            tracked_cell_data['Area (mm^2)'].append(area_mm)

            # Convert Coordinates to mm
            coordinates_mm = list(cell_locations[tracked_cell_id])
            coordinates_mm[0] = float(coordinates_mm[0] * pixels_to_mm)
            coordinates_mm[1] = float(coordinates_mm[1] * pixels_to_mm)
            tracked_cell_data['X Position (mm)'].append(coordinates_mm[0])
            tracked_cell_data['Y Position (mm)'].append(coordinates_mm[1])

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
                tracked_cell_data['Area (mm^2)'].append(area_mm)

                # Convert Coordinates to mm
                coordinates_mm = list(cell_locations[tracked_cell_id])
                coordinates_mm[0] = float(coordinates_mm[0] * pixels_to_mm)
                coordinates_mm[1] = float(coordinates_mm[1] * pixels_to_mm)
                tracked_cell_data['X Position (mm)'].append(coordinates_mm[0])
                tracked_cell_data['Y Position (mm)'].append(coordinates_mm[1])

                # Record Time from start
                tracked_cell_data['Time'].append(frame_num * TIME_BETWEEN_FRAMES)

                # Draw marker at cell's initial position
                cv.circle(processed, tracked_cell_coords[tracked_cell_id][0], 2, (255, 255, 255), 3)

                # Draw an arrow for every frame of movement going from its last position to its next position
                for i in range(1, len(tracked_cell_coords[tracked_cell_id])):
                    cv.arrowedLine(processed, tracked_cell_coords[tracked_cell_id][i-1], tracked_cell_coords[tracked_cell_id][i], (255, 255, 255), 2, cv.LINE_AA, 0, 0.1)

                # Display edited photo
                cv.imshow("Cell Tracking", processed)

                # Increment Frame Counter
                frame_num += 1

                # if not the last frame display it for only a short amount of time
                if frame_num < total_frames - 1:
                    # Adjust waitKey to change time each frame is displayed
                    # Press q to exit out of opencv early
                    if cv.waitKey(50) & 0xFF == ord('q'):
                        break
                else:
                    # if on the last frame display it until the q key is pressed
                    k = cv.waitKey(0)
                    if k == ord('q'):
                        break

            # Export data to excel
            export.individual_to_excel_file(EXCEL_FILE, tracked_cell_data, TIME_BETWEEN_FRAMES, f"Cell {tracked_cell_id}")
            # Draw Graph charting cell's size
            matplotlib_graphing.export_individual_cell_area(f"{PDF_FILE}Cell{tracked_cell_id}_Area_Graph.pdf", tracked_cell_data, "Time", "Area (mm^2)", f"Cell {tracked_cell_id}: Area vs Time")



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