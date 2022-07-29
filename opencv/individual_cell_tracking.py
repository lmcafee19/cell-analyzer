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
VIDEO = '../videos/Sample_cell_culture_4.mp4'
EXCEL_FILE = "../data/cell41_data.xlsx"
PDF_FILE = "../data/"
IMAGE_FILE = "../data/"
SCALE = .25
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10
PATH_COLOR = (255, 255, 255)
START_COLOR = (255, 0, 0)
END_COLOR = (0,0,255)

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 150
VIDEO_WIDTH_MM = 195.9

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10

def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{VIDEO}'
    # Check if video exists
    if not os.path.exists(videoFile):
        raise Exception("File cannot be found")
    else:
        # Read in video
        capture = cv.VideoCapture(videoFile)
        total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        # Grab last frame for processing later TOO SLOW!!
        # Fast forward to last frame
        # capture.set(cv.CAP_PROP_POS_FRAMES, total_frames - 1)
        #
        # valid, final_frame = capture.read()
        #
        # # Rewind Video to first frame
        # capture.set(cv.CAP_PROP_POS_FRAMES, 0)

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
        frame_num = 1
        first_frame = None
        final_frame = None
        Xmin = None
        Ymin = None
        Xmax = 0
        Ymax = 0

        # Open First Frame of Video and Detect all cells within it, making sure to label them
        valid, frame = capture.read()
        if not valid:
            raise Exception("Video cannot be read")

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)

        # Save Reference to edited first frame for later
        first_frame = processed

        # Detect minimum cell boundaries and display edited photo
        cont, shapes = analysis.detect_shape_v2(processed)

        # Use Tracker to label and record coordinates of all cells
        cell_locations, cell_areas = tracker.update(shapes)

        # Label all cells with cell id
        labeled_img = analysis.label_cells(processed, cell_locations)

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
                cont, shapes = analysis.detect_shape_v2(processed)

                # Use Tracker to label and record coordinates of all cells
                cell_locations, cell_areas = tracker.update(shapes)

                # Update Tracking information
                # Record data about tracked cell
                tracked_cell_coords[tracked_cell_id].append(list(cell_locations[tracked_cell_id]))

                # Keep Track of min/max x/y value the tracked cell is at
                if Xmin is None or cell_locations[tracked_cell_id][0] < Xmin:
                    Xmin = cell_locations[tracked_cell_id][0]

                if cell_locations[tracked_cell_id][0] > Xmax:
                    Xmax = cell_locations[tracked_cell_id][0]

                if Ymin is None or cell_locations[tracked_cell_id][1] < Ymin:
                    Ymin = cell_locations[tracked_cell_id][1]

                if cell_locations[tracked_cell_id][1] > Ymax:
                    Ymax = cell_locations[tracked_cell_id][1]

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

                # Keep track of previous frame
                final_frame = frame

                # if not the last frame display it for only a short amount of time
                if frame_num < total_frames:
                    # Adjust waitKey to change time each frame is displayed
                    # Press q to exit out of opencv early
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    # if on the last frame display it until the q key is pressed
                    k = cv.waitKey(0)
                    if k == ord('q'):
                        break

            # Create Color Image containing the path the tracked cell took
            # Scale image to match
            final_photo = analysis.rescale_frame(final_frame, SCALE)

            # Draw Boundary for Cell's starting position
            final_photo = analysis.draw_initial_cell_boundary(first_frame, tracked_cell_coords[tracked_cell_id][0],
                                                              final_photo, START_COLOR)
            # Draw a line for every frame of movement going from its last position to its next position
            for i in range(1, len(tracked_cell_coords[tracked_cell_id])):
                cv.line(final_photo, tracked_cell_coords[tracked_cell_id][i - 1], tracked_cell_coords[tracked_cell_id][i],
                        PATH_COLOR, 2)

            # Draw dot at final centroid
            cv.circle(final_photo, tracked_cell_coords[tracked_cell_id][len(tracked_cell_coords[tracked_cell_id]) - 1], 4, END_COLOR, cv.FILLED)


            # Draw an arrow for every frame of movement going from its last position to its next position
            # for i in range(1, len(tracked_cell_coords[tracked_cell_id])):
            #     cv.arrowedLine(final_photo, tracked_cell_coords[tracked_cell_id][i - 1], tracked_cell_coords[tracked_cell_id][i],
            #                    (255, 0, 0), 2, cv.LINE_AA, 0, 0.2)


            # Crop Image to have path take up majority of photo
            # TODO FIX Cropping and Upscaling of image to be sharper
            # w = final_photo.shape[0]
            # h = final_photo.shape[1]
            # final_dimensions = (1920, 1080)
            #
            # # Calculate border around cell's path using specified percentage
            # border_percent = .25
            # border_pixels_height = int(border_percent * h)
            # border_pixels_width = int(border_percent * w)
            #
            # print(f"minx: {Xmin}, maxx: {Xmax}, miny: {Ymin}, maxy: {Ymax}")
            # print(final_photo.shape)
            #
            # # Image Cropping [ymin: ymax, xmin:xmax]
            # # Crop Image to center around path of cell
            # crop = final_photo[Ymin - border_pixels_height:h, Xmin - border_pixels_width:Xmax + border_pixels_width]
            #
            # # Display Image
            # cv.imshow("Cropped Path", crop)

            # # Resize Image to desired final dimensions and then redraw the path ontop
            # resized = cv.resize(crop, final_dimensions, interpolation=cv.INTER_AREA)
            #
            # # Calculate Ratio between old pixel dimensions and final
            # dimension_ratio_x = final_dimensions[0]/final_photo.shape[0]
            # dimension_ratio_y = final_dimensions[1]/final_photo.shape[1]
            #
            # # Draw an arrow for every frame of movement going from its last position to its next position
            # for i in range(1, len(tracked_cell_coords[tracked_cell_id])):
            #     # Determine new coordinates for resized images
            #     Xprev = int(tracked_cell_coords[tracked_cell_id][i][0] * dimension_ratio_x)
            #     Yprev = int(tracked_cell_coords[tracked_cell_id][i][1] * dimension_ratio_y)
            #     Xcur = int(tracked_cell_coords[tracked_cell_id][i][0] * dimension_ratio_x)
            #     Ycur = int(tracked_cell_coords[tracked_cell_id][i][1] * dimension_ratio_y)
            #     # Draw Arrow
            #     cv.arrowedLine(resized, (Xprev, Yprev),
            #                    (Xcur, Ycur),
            #                    (255, 0, 0), 2, cv.LINE_AA, 1, .5)
            #
            # cv.imshow("Resized", resized)

            cv.imshow("OG", final_photo)
            cv.waitKey(0)

            # Save Image
            #cv.imwrite(f"{IMAGE_FILE}Cell{tracked_cell_id}_Path.png", final_photo)

            # Export data to excel
            export.individual_to_excel_file(EXCEL_FILE, tracked_cell_data, TIME_BETWEEN_FRAMES, f"Cell {tracked_cell_id}")
            # Draw Graph charting cell's size
            # matplotlib_graphing.export_individual_cell_data(
            #                                                 tracked_cell_data, "Time", "Area (mm^2)",
            #                                                 title=f"Cell {tracked_cell_id}: Area vs Time", filename=f"{PDF_FILE}Cell{tracked_cell_id}_Area_Graph.pdf")
            # # Draw Graph charting cell's movement
            # matplotlib_graphing.export_individual_cell_data(tracked_cell_data, "X Position (mm)", "Y Position (mm)",
            #                                                 filename=f"{PDF_FILE}Cell{tracked_cell_id}_Movement_Graph.pdf",
            #                                                 labels=tracked_cell_data["Time"], title=f"Cell {tracked_cell_id}: Movement")
            #
            # matplotlib_graphing.export_individual_cell_data(tracked_cell_data, "X Position (mm)", "Y Position (mm)",
            #
            #                                                 labels=tracked_cell_data["Time"],
            #                                                 title=f"Cell {tracked_cell_id}: Movement")
            #
            # # Draw Simplified version of graph charting cell's movement
            # matplotlib_graphing.export_simplified_individual_cell_data(f"{PDF_FILE}Cell{tracked_cell_id}_Simple_Movement_Graph.pdf",
            #                                                 tracked_cell_data, "X Position (mm)", "Y Position (mm)", 15,
            #                                                 labels=tracked_cell_data["Time"],
            #                                                 title=f"Cell {tracked_cell_id}: Movement")

main()