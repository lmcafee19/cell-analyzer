'''
@brief Contains Classes to provide functionality to the two options from the GUI
    Individual Tracker: Contains functions to track a single cell within the image or video
    Culture Tracker: Contains functions to track all cells within an image or video
'''
import math
import cv2
import numpy as np
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing
from collections import OrderedDict
from datetime import datetime

# Define global variables for spheroid class
FRAME1 = np.zeros((1, 1, 1), np.uint8)
PREVIOUS = FRAME1.copy()
DRAWING = False  # true if mouse is pressed
IX, IY = -1, -1
START = (0, 0), 0  # centroid, radius


"""
Defines class that manages the tracking of a specified individual cell within a video
"""
# create spheroid tracking class
# make button event
# find event that matches up with event under run
# hide settings page
class IndividualTracker:
    def __init__(self, source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):

        self.source = source
        # Open the source
        if is_image(source):
            # If given file is an image, use imread
            self.vid = cv2.imread(source)
            self.frames = 1
        else:
            # if video use VideoCapture
            self.vid = cv2.VideoCapture(source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", source)
            # Get source frames, if source is an image default to 1
            self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)


        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1


        self.height_mm = float(height_mm)
        self.width_mm = float(width_mm)
        # Real world time in minutes that pass between each image being taken
        self.time_between_frames = time_between_frames

        # Max/Min Size of Objects to detect as cells within video
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

        # Define Constants for video editing
        self.scale = scale
        self.contrast = contrast
        self.brightness = brightness
        self.blur_intensity = blur_intensity

        # Keep track of cell id
        self.tracker = ct.CentroidTracker()
        self.tracked_cell_id = -1
        self.tracked_cell_data = {'Time (mins)': [], f'X Position ({self.units})': [], f'Y Position ({self.units})': [], f'Area ({self.units}^2)': []}
        self.tracked_cell_coords = OrderedDict()

        # Keep track of last played frame's tracker data
        self.cell_locations = None
        self.cell_areas = None

        # Keep Track of Photo Data for exports later
        self.first_frame = None
        self.final_frame = None
        self.Xmin = None
        self.Ymin = None
        self.Xmax = 0
        self.Ymax = 0

    '''
    Updates the currently tracked cell to that of the given ID.
    @param tracked_cell_id Positive integer pertaining to the ID of one of the tracked cells
    '''
    def set_tracked_cell(self, tracked_cell_id:int):
        self.tracked_cell_id = tracked_cell_id


    '''
    Updates the min_cell_size field
    @param min_size Positive integer pertaining to smallest size cell to track
    '''
    def set_min_size(self, min_size:int):
        self.min_cell_size = min_size


    '''
    Updates the max_cell_size field
    @param min_size Positive integer pertaining to largest size cell to track
    '''
    def set_max_size(self, max_size:int):
        self.max_cell_size = max_size


    '''
    Updates the contrast field
    '''
    def set_contrast(self, contrast):
        self.contrast = contrast


    '''
    Updates the brightness field
    '''
    def set_brightness(self, brightness):
        self.brightness = brightness


    '''
    Updates the blur_intensity field
    '''
    def set_blur_intensity(self, blur_intensity):
        self.blur_intensity = blur_intensity


    '''
    Retrieves the next frame of the current video and records data about the tracked cell. If no frame is found returns None instead
    @:returns unedited frame, edited frame
    '''
    def next_frame(self):
        valid, frame = self.vid.read()

        # If next frame is not found return None?
        if not valid:
            return None, None

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast,
                                            self.brightness, self.blur_intensity)

        # Detect minimum cell boundaries and their centroids for tracker
        cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)

        # Draw a line for every frame of movement going from its last position to its next position
        for i in range(1, len(self.tracked_cell_coords[self.tracked_cell_id])):
            cv2.line(processed, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    (255, 255, 255), 2)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Update Tracking information
        self.update_tracker_data()

        # Increment Frame Counter
        self.frame_num += 1

        # If this is the final frame of the video record it for export later
        if self.frame_num >= self.frames:
            self.final_frame = frame

        return frame, processed


    '''
    Updates recorded data about the currently tracked cell based on data collected in the last frame 
    '''
    def update_tracker_data(self):
        # Record data about tracked cell from previous frame
        self.tracked_cell_coords[self.tracked_cell_id].append(list(self.cell_locations[self.tracked_cell_id]))

        # Keep Track of min/max x/y value the tracked cell is at
        if self.Xmin is None or self.cell_locations[self.tracked_cell_id][0] < self.Xmin:
            self.Xmin = self.cell_locations[self.tracked_cell_id][0]

        if self.cell_locations[self.tracked_cell_id][0] > self.Xmax:
            self.Xmax = self.cell_locations[self.tracked_cell_id][0]

        if self.Ymin is None or self.cell_locations[self.tracked_cell_id][1] < self.Ymin:
            self.Ymin = self.cell_locations[self.tracked_cell_id][1]

        if self.cell_locations[self.tracked_cell_id][1] > self.Ymax:
            self.Ymax = self.cell_locations[self.tracked_cell_id][1]

        # Convert area to mm^2
        area_mm = self.cell_areas[self.tracked_cell_id]
        print("cell_areas[self.tracked_cell_id]:", area_mm)
        area_mm = area_mm * (self.pixels_to_mm**2)
        print("area_mm:", area_mm)
        self.tracked_cell_data[f'Area ({self.units}^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data[f'X Position ({self.units})'].append(coordinates_mm[0])
        self.tracked_cell_data[f'Y Position ({self.units})'].append(coordinates_mm[1])

        # Record Time from start
        self.tracked_cell_data['Time (mins)'].append((self.frame_num - 1) * self.time_between_frames)


    '''
    Proceeds to the first frame of the video and initializes the centroid tracker 
    @returns Unedited first frame and Edited first frame of the video with each cell detected and given an ID for the user to choose
    '''
    def get_first_frame(self):
        # If source is an image return the already read image type, otherwise grab the first frame
        if is_image(self.source):
            valid = True
            frame = self.vid
        else:
            # Open First Frame of Video and Detect all cells within it, making sure to label them
            valid, frame = self.vid.read()

        if not valid:
            raise Exception("Source cannot be read")

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)

        # Save Reference to edited first frame for export of cell's path
        self.first_frame = processed

        # Detect minimum cell boundaries and display edited photo
        cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Label all cells with cell id only if there is a small amount of cells
        if len(self.cell_locations) <= 25:
            labeled_img = analysis.label_cells(processed, self.cell_locations)
        else:
            labeled_img = processed

        if self.pixels_to_mm is None or self.pixels_to_mm == 0:
            # Grab Frame's dimensions in order to convert pixels to mm
            (h, w) = labeled_img.shape[:2]
            # If instead the dimensions of the image were given then calculate the pixel conversion using those
            # If selected units were micro meters (µm)
            if self.units == "µm":
                self.pixels_to_mm = (((self.height_mm / h) + (self.width_mm / w)) / 2) * 1000
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = ((self.height_mm / h) + (self.width_mm / w)) / 2
        else:
            # If pixels to mm were already given, then convert it to our new scale for the image
            if self.units == "µm":
                # If µm was selected as the units convert between mm and µm (mm * 1000)
                self.pixels_to_mm = (self.pixels_to_mm * self.scale) * 1000
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = self.pixels_to_mm * self.scale


        # Increment Frame Counter
        self.frame_num += 1

        # Return unedited first frame and the img with cell ids labeled
        return frame, labeled_img


    '''
    Returns the number of cells found within the first frame
    '''
    def get_num_cells_found(self):
        return len(self.cell_locations)

    '''
    Masks the first frame of the image to only show the area around the selected cell
    @param cell_id Which Cell to highlight
    @return first_frame of the video with the cell outlined and labeled
    '''
    def outline_cell(self, cell_id:int):
        return analysis.outline_cell(self.first_frame, cell_id, self.cell_locations[int(cell_id)], self.max_cell_size)


    '''
    Initializes coordinate and area data found in the first frame about the currently tracked cell into the 
    self.tracked_cell_data and self.tracked_cell_coords data structures
    '''
    def initialize_tracker_data(self):
        # Record data about tracked cell from previous frame
        self.tracked_cell_coords[self.tracked_cell_id] = list()
        self.tracked_cell_coords[self.tracked_cell_id].append(list(self.cell_locations[self.tracked_cell_id]))

        # Convert area to mm^2
        area_mm = self.cell_areas[self.tracked_cell_id]
        area_mm = area_mm * (self.pixels_to_mm ** 2)

        self.tracked_cell_data[f'Area ({self.units}^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data[f'X Position ({self.units})'].append(coordinates_mm[0])
        self.tracked_cell_data[f'Y Position ({self.units})'].append(coordinates_mm[1])

        # Record Time from start
        self.tracked_cell_data['Time (mins)'].append((self.frame_num - 2) * self.time_between_frames)


    '''
    Determines if the given cell id relates to a known cell
    @return True if id is known/valid, and False if not
    '''
    def is_valid_id(self, cell_id:int):
        valid = False
        # Ensure that given cell id is a positive integer within the range of known ids
        if 0 <= int(cell_id) < len(self.cell_locations):
            valid = True

        return valid


    '''
    Saves a visualization of the path that cell traveled during this video
    Note: the entirety of self.vid must have already been played before calling this
    @param filename Optional path + filename to save this image to. 
    If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    @param path_color bgr values for color of the path to draw
    @param start_color bgr values for the color of the starting cell position to be drawn
    @param end_color bgr values for color of final cell position to be drawn
    '''
    def export_final_path(self, filename=None, path_color=(255, 255, 255), start_color=(255, 0, 0), end_color=(0, 0, 255)):
        if self.final_frame is None or self.first_frame is None:
            raise Exception("Video Must finish playing before exporting the cell's path")

        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Path.png"

        # Create Color Image containing the path the tracked cell took
        # Scale image to match
        final_photo = analysis.rescale_frame(self.final_frame, self.scale)

        # Draw Boundary for Cell's starting position
        final_photo = analysis.draw_initial_cell_boundary(self.first_frame, self.tracked_cell_coords[self.tracked_cell_id][0],
                                                          final_photo, start_color)
        # Draw a line for every frame of movement going from its last position to its next position
        for i in range(1, len(self.tracked_cell_coords[self.tracked_cell_id])):
            cv2.line(final_photo, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    path_color, 2)

        # Draw dot at final centroid
        cv2.circle(final_photo, self.tracked_cell_coords[self.tracked_cell_id][len(self.tracked_cell_coords[self.tracked_cell_id]) - 1], 4,
                  end_color, cv2.FILLED)

        # Save Image
        cv2.imwrite(filename, final_photo)


    '''
    Exports recorded positional, area, and statistical data about the tracked cell to an excel spreadsheet
    @param filename Optional path + filename to save this data to, should end in extension .xlsx. 
        If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_excel(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Data.xlsx"

        # Export data to excel
        export.individual_to_excel_file(filename, self.tracked_cell_data, self.time_between_frames, units=self.units, sheetname=f"Cell {self.tracked_cell_id}")


    '''
    Exports recorded positional and area data about the tracked cell to a csv file
    @param filename Optional path + filename to save this data to, should end in extension .csv. 
        If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_csv(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Data.csv"

        # Export data to excel
        export.individual_to_csv_file(filename, self.tracked_cell_data)


    '''
    Creates a line chart visualizing selected data of an individual cell
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param labels Optional. Iterable container of labels for each point
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    @param Title Optional. Title of the chart
    @param color Name of the color to plot the points with
    '''
    def export_graph(self, xaxis, yaxis, title=None, labels=None, num_labels=2, filename=None, color="blue"):
        # Use matplotlib to graph given data
        matplotlib_graphing.export_individual_cell_data(self.tracked_cell_data, xaxis, yaxis, filename, labels, num_labels, title, color)


    '''
    Creates a line chart with the tracked cell's x position on the x axis and its y position on the y axis, points are
    labeled with their respective timestamps 
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_movement_graph(self, num_labels=2, filename=None):
        self.export_graph(f"X Position ({self.units})", f"Y Position ({self.units})", f"Cell {self.tracked_cell_id}: Movement", self.tracked_cell_data["Time (mins)"], num_labels, filename)


    '''
    Creates a line chart with the tracked cell's area on the x axis and time on the y axis
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_area_graph(self, num_labels=2, filename=None):
        self.export_graph("Time (mins)", f"Area ({self.units}^2)", f"Cell {self.tracked_cell_id}: Area over Time", filename=filename)


    # Release the video source when the object is destroyed
    def __del__(self):
        if not is_image(self.source) and self.vid.isOpened():
            self.vid.release()


class SpheroidTracker:
    def __init__(self, source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):

        self.source = source
        # Open the source
        if is_image(source):
            # If given file is an image, use imread
            self.vid = cv2.imread(source)
            self.frames = 1
        else:
            # if video use VideoCapture
            self.vid = cv2.VideoCapture(source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", source)
            # Get source frames, if source is an image default to 1
            self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1

        self.height_mm = float(height_mm)
        self.width_mm = float(width_mm)
        # Real world time in minutes that pass between each image being taken
        self.time_between_frames = time_between_frames

        # Max/Min Size of Objects to detect as cells within video
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

        # Define Constants for video editing
        self.scale = scale
        self.contrast = contrast
        self.brightness = brightness
        self.blur_intensity = blur_intensity

        # Keep track of cell id
        self.tracker = ct.CentroidTracker()
        self.tracked_cell_id = 0
        self.tracked_cell_data = {'Time (mins)': [], f'X Position ({self.units})': [], f'Y Position ({self.units})': [], f'Area ({self.units}^2)': []}
        self.tracked_cell_coords = OrderedDict()

        # Keep track of last played frame's tracker data
        self.cell_locations = None
        self.cell_areas = None

        # Keep Track of Photo Data for exports later
        self.first_frame = None
        self.final_frame = None
        self.Xmin = None
        self.Ymin = None
        self.Xmax = 0
        self.Ymax = 0

    '''
    Updates the currently tracked cell to that of the given ID.
    @param tracked_cell_id Positive integer pertaining to the ID of one of the tracked cells
    '''
    def set_tracked_cell(self, tracked_cell_id:int):
        self.tracked_cell_id = tracked_cell_id


    # '''
    # Updates the min_cell_size field
    # @param min_size Positive integer pertaining to smallest size cell to track
    # '''
    # def set_min_size(self, min_size:int):
    #     self.min_cell_size = min_size
    #
    #
    # '''
    # Updates the max_cell_size field
    # @param min_size Positive integer pertaining to largest size cell to track
    # '''
    # def set_max_size(self, max_size:int):
    #     self.max_cell_size = max_size


    # '''
    # Updates the contrast field
    # '''
    # def set_contrast(self, contrast):
    #     self.contrast = contrast
    #
    #
    # '''
    # Updates the brightness field
    # '''
    # def set_brightness(self, brightness):
    #     self.brightness = brightness
    #
    #
    # '''
    # Updates the blur_intensity field
    # '''
    # def set_blur_intensity(self, blur_intensity):
    #     self.blur_intensity = blur_intensity

    '''
    Proceeds to the first frame of the video and initializes the centroid tracker 
    @returns Unedited first frame and Edited first frame of the video with each cell detected and given an ID for the user to choose
    '''
    def get_first_frame(self):
        # Note: first frame = selection page
        global START, FRAME1
        # If source is an image return the already read image type, otherwise grab the first frame
        if is_image(self.source):
            valid = True
            frame = self.vid
        else:
            # Open First Frame of Video and Detect all cells within it, making sure to label them
            valid, frame = self.vid.read()

        if not valid:
            raise Exception("Source cannot be read")

        # Set global variable to be referenced by spheroid_selection()
        # resize frame because it is huge
        FRAME1 = analysis.rescale_frame(frame, self.scale)
        # Save Reference to edited first frame
        self.first_frame = FRAME1

        # # User draws circle around desired spheroid, updates START
        # self.select_spheroid()

        # # initialize first centroid from user-drawn circle
        # centroid, radius = START
        # print("START:", START)
        #
        # # Process Image to detect spheroid
        # processed = self.image_processing(frame, centroid, radius)  # returns blob
        # # processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)
        #
        # # Set up units
        # if self.pixels_to_mm is None or self.pixels_to_mm == 0:
        #     # Grab Frame's dimensions in order to convert pixels to mm
        #     (h, w) = processed.shape[:2]
        #     # If instead the dimensions of the image were given then calculate the pixel conversion using those
        #     # If selected units were micro meters (µm)
        #     if self.units == "µm":
        #         self.pixels_to_mm = (((self.height_mm / h) + (self.width_mm / w)) / 2) * 1000
        #     else:
        #         # Otherwise use millimeters as unit
        #         self.pixels_to_mm = ((self.height_mm / h) + (self.width_mm / w)) / 2
        # else:
        #     # If pixels to mm were already given, then convert it to our new scale for the image
        #     if self.units == "µm":
        #         # If µm was selected as the units convert between mm and µm (mm * 1000)
        #         self.pixels_to_mm = (self.pixels_to_mm * self.scale) * 1000
        #     else:
        #         # Otherwise use millimeters as unit
        #         self.pixels_to_mm = self.pixels_to_mm * self.scale
        #
        # # Detect minimum cell boundaries and display edited photo
        # # returns cont = image with contours outlined and shapes = centroid, int tuple (x, y)
        # # cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)
        # cont = self.outline_spheroid(processed)  # returns image showing largest contour
        # shapes = self.get_shape(frame, cont)
        #
        # # update argument must be a dictionary mapping centroids of detected objects to their area
        # self.cell_locations, self.cell_areas = self.tracker.update(shapes)
        #
        # # Update Tracking information
        # self.update_tracker_data()
        #
        # # update START to hold new centroid and radius to be used in next frame
        # START = self.next_circle_position(frame, cont)
        #
        # # # Label all cells with cell id only if there is a small amount of cells
        # # if len(self.cell_locations) <= 25:
        # #     labeled_img = analysis.label_cells(processed, self.cell_locations)
        # # else:
        # #     labeled_img = processed
        #
        # # Increment Frame Counter
        # self.frame_num += 1

        # Return unedited first frame twice
        return frame, FRAME1

    '''
    Returns the number of cells found within the first frame
    '''

    def get_num_cells_found(self):
        return 1

    # define mouse callback function to draw circle
    def select_spheroid(self):
        # to show on selection page.  Take care of first frame
        global FRAME1, IX, IY, DRAWING, PREVIOUS, START

        # # Open video
        # valid, frame = self.vid.read()
        # # If next frame is not found return None?
        # if not valid:
        #     return None, None
        #
        # # resize frame because it is huge
        # frame = analysis.rescale_frame(frame, self.scale)
        # User selects their spheroid
        # FRAME1 = frame

        # user draws circle, updates START
        # initialize first centroid from user-drawn circle

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
            if k == 32:  # hit spacebar to close window (replace this with a button in gui)
                cv2.imwrite("circled_img.jpg", PREVIOUS)
                break

        cv2.destroyWindow('Drag Circle Window')

        # initialize first centroid from user-drawn circle
        centroid, radius = START
        print("START after drawing circle:", START)

        # Process Image to detect spheroid
        processed = self.image_processing(FRAME1, centroid, radius)  # returns blob
        # processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)

        # Set up units
        if self.pixels_to_mm is None or self.pixels_to_mm == 0:
            # Grab Frame's dimensions in order to convert pixels to mm
            (h, w) = processed.shape[:2]
            print("(h,w):", (h, w))
            # If instead the dimensions of the image were given then calculate the pixel conversion using those
            # If selected units were micro meters (µm)
            if self.units == "µm":
                self.pixels_to_mm = (((self.height_mm / h) + (self.width_mm / w)) / 2) * 1000
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = ((self.height_mm / h) + (self.width_mm / w)) / 2
        else:
            # If pixels to mm were already given, then convert it to our new scale for the image
            if self.units == "µm":
                # If µm was selected as the units convert between mm and µm (mm * 1000)
                self.pixels_to_mm = (self.pixels_to_mm * self.scale) * 1000
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = self.pixels_to_mm * self.scale
        print("pixels to mm:", self.pixels_to_mm)
        # Detect minimum cell boundaries and display edited photo
        # returns cont = image with contours outlined and shapes = centroid, int tuple (x, y)
        # cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)
        cont = self.outline_spheroid(processed)  # returns image showing largest contour
        shapes = self.get_shape(FRAME1, cont)

        # update argument must be a dictionary mapping centroids of detected objects to their area
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)
        print("cell_locations:", self.cell_locations)
        print("cell_areas:", self.cell_areas)

        # Update Tracking information
        # self.update_tracker_data()

        # update START to hold new centroid and radius to be used in next frame
        START = self.next_circle_position(FRAME1, cont)

        # # Label all cells with cell id only if there is a small amount of cells
        # if len(self.cell_locations) <= 25:
        #     labeled_img = analysis.label_cells(processed, self.cell_locations)
        # else:
        #     labeled_img = processed

        # Increment Frame Counter
        self.frame_num += 1

    '''
    Retrieves the next frame of the current video and records data about the tracked cell. If no frame is found returns None instead
    @:returns unedited frame, edited frame
    '''
    def next_frame(self):
        global FRAME1, START
        print("made it to frame", self.frame_num)
        valid, frame = self.vid.read()

        # If next frame is not found return None?
        if not valid:
            return None, None

        # resize frame because it is huge
        frame = analysis.rescale_frame(frame, self.scale)

        # if self.frame_num == 1:
        #     # User selects their spheroid
        #     FRAME1 = frame
        #     self.select_spheroid()  # user draws circle, updates START
        #     # initialize first centroid from user-drawn circle

        centroid, radius = START
        print("START:", START)

        # Process Image to detect spheroid
        processed = self.image_processing(frame, centroid, radius)  # returns blob
        # processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)

        if self.pixels_to_mm is None or self.pixels_to_mm == 0:
            # Grab Frame's dimensions in order to convert pixels to mm
            (h, w) = processed.shape[:2]
            # If instead the dimensions of the image were given then calculate the pixel conversion using those
            # If selected units were micro meters (µm)
            if self.units == "µm":
                self.pixels_to_mm = (((self.height_mm / h) + (self.width_mm / w)) / 2) * 1000  # avg mm/pixel
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = ((self.height_mm / h) + (self.width_mm / w)) / 2
        else:
            # If pixels to mm were already given, then convert it to our new scale for the image
            if self.units == "µm":
                # If µm was selected as the units convert between mm and µm (mm * 1000)
                self.pixels_to_mm = (self.pixels_to_mm * self.scale) * 1000
            else:
                # Otherwise use millimeters as unit
                self.pixels_to_mm = self.pixels_to_mm * self.scale

        # Detect minimum cell boundaries and display edited photo
        # returns cont = image with contours outlined and shapes = centroid, int tuple (x, y)
        # cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)
        cont = self.outline_spheroid(processed)  # returns image showing largest contour
        shapes = self.get_shape(processed, cont)
        print("shapes:", shapes)

        # update argument must be a dictionary mapping centroids of detected objects to their area
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # processed = analysis.process_image

        # # obtain centroid and radius from previous frame
        # centroid, radius = START

        # # # Process Image to better detect cells
        # processed = self.image_processing(self, frame, centroid, radius)  # returns blob
        # # processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast,
        #                                     self.brightness, self.blur_intensity)

        # Detect minimum cell boundaries and their centroids for tracker
        # cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)
        # cont = self.outline_spheroid(processed)  # returns largest contour

        # Draw a line for every frame of movement going from its last position to its next position
        for i in range(1, len(self.tracked_cell_coords[self.tracked_cell_id])):
            cv2.line(processed, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    (255, 255, 255), 2)

        # # Use Tracker to label and record coordinates of all cells
        # self.cell_locations, self.cell_areas = self.tracker.update(centroid)

        # Update Tracking information
        self.update_tracker_data()

        # Update START to hold new centroid and radius to be used in next frame
        START = self.next_circle_position(processed, cont)

        # Increment Frame Counter
        self.frame_num += 1

        # If this is the final frame of the video record it for export later
        if self.frame_num >= self.frames:
            self.final_frame = frame

        # returns unedited frame, edited frame
        return frame, processed


    '''
    Updates recorded data about the currently tracked cell based on data collected in the last frame 
    '''
    def update_tracker_data(self):
        # Record data about tracked cell from previous frame
        # print("tracked_cell_coords:", self.tracked_cell_coords)
        # print("self.cell_locations[self.tracked_cell_id]:", self.cell_locations[self.tracked_cell_id])
        self.tracked_cell_coords[self.tracked_cell_id].append(list(self.cell_locations[self.tracked_cell_id]))

        # Keep Track of min/max x/y value the tracked cell is at
        if self.Xmin is None or self.cell_locations[self.tracked_cell_id][0] < self.Xmin:
            self.Xmin = self.cell_locations[self.tracked_cell_id][0]

        if self.cell_locations[self.tracked_cell_id][0] > self.Xmax:
            self.Xmax = self.cell_locations[self.tracked_cell_id][0]

        if self.Ymin is None or self.cell_locations[self.tracked_cell_id][1] < self.Ymin:
            self.Ymin = self.cell_locations[self.tracked_cell_id][1]

        if self.cell_locations[self.tracked_cell_id][1] > self.Ymax:
            self.Ymax = self.cell_locations[self.tracked_cell_id][1]

        # Convert area to mm^2
        area_mm = self.cell_areas[self.tracked_cell_id] * (self.pixels_to_mm**2)
        print("area_mm:", area_mm)
        self.tracked_cell_data[f'Area ({self.units}^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data[f'X Position ({self.units})'].append(coordinates_mm[0])
        self.tracked_cell_data[f'Y Position ({self.units})'].append(coordinates_mm[1])

        # Record Time from start
        self.tracked_cell_data['Time (mins)'].append((self.frame_num - 1) * self.time_between_frames)




    '''
    Masks the first frame of the image to only show the area around the selected cell
    @param cell_id Which Cell to highlight
    @return first_frame of the video with the cell outlined and labeled
    '''
    def outline_spheroid(self, processed: np.ndarray) -> np.ndarray:
        """Detect outline of spheroid
        	Code adapted from:
        	https://opencv4-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
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

    def image_processing(self, image, centroid, radius):
        """Create a mask of spheroid (drawn circle) and blur out surroundings
		to allow for better tracking of cells in spheroid.
		Code adapted from:
		https://www.digitalocean.com/community/tutorials/arithmetic-bitwise-and-masking-python-opencv
		https://www.tutorialspoint.com/how-to-mask-an-image-in-opencv-python
		blurring:
		https://stackoverflow.com/questions/73035362/how-to-draw-a-blurred-circle-in-opencv-python
		https://learnopencv2.com/alpha-blending-using-opencv-cpp-python/
		https://theailearner.com/2019/05/06/gaussian-blurring/
		https://theailearner.com/tag/cv-medianblur/
		"""
        # declare local constants
        gaus_kernel = (9, 9)  # must be odd and positive
        med_kernel_size = 23  # must be odd and positive, always a square so only one value
        intensity = 1
        halo_multiplier = 1.5

        # Make light blur around whole background, such that unselected cells are still identifiable:
        # (this is to catch parts of spheroid that got cut off by the drawn circle, and those in next frame)
        # create a white image that is the same size as the spheroid image
        mask = (np.ones(image.shape, dtype="uint8")) * 255
        # create a filled black circle on the mask
        cv2.circle(mask, centroid, radius, 0, -1)  # (image, (center_x, center_y), radius, color, thickness)
        # apply light gaussian blur to entire original image
        # cv2.imshow("original", image)
        light_blur = cv2.GaussianBlur(image, gaus_kernel, 1)
        # paste blurred image on white section of mask (background) and untouched image in black circle in mask (selected)
        blur1 = np.where(mask > 0, light_blur, image)
        # cv2.imshow("first blur - background", blur1)

        # Create stronger blur in a halo around non-blurred region
        # make new mask with bigger circle
        mask2 = (np.ones(image.shape, dtype="uint8")) * 255
        cv2.circle(mask2, centroid, int(radius * halo_multiplier), 0, -1)
        # apply stronger median blur to white regions of mask2 (hide background contours)
        strong_blur = cv2.medianBlur(image, med_kernel_size)
        # paste strong blur onto white region of mask2, fill black circle of mask2 with the first blurred image
        blur2 = np.where(mask2 > 0, strong_blur, blur1)
        blur2 *= intensity  # multiplied by an int as effective means of increasing contrast
        # cv2.imshow("halo", blur2)

        # Merge cell shapes into one shape, return this image
        # median blur over processed image to create "shadowed" region where spheroid is
        blob = cv2.medianBlur(blur2, med_kernel_size)
        # cv2.imshow("lumped", blob)
        return blob

    def get_shape(self, image, outline):
        """Run calculations and get data for each frame
		find approximate center of mass, centroid, radius, and area
		save/track center of mass and area
		pass along centroid and radius for creation of new blur circles on next frame
		Code adapted from:
		https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
		https://www.tutorialspoint.com/how-to-find-the-minimum-enclosing-circle-of-an-object-in-opencv-python"""
        # return dictionary of spheroid objects and area

        # Create Dictionary Mapping detected centroids to their area
        centroids = {}

        # Find approximate center of mass (CoM)
        M = cv2.moments(outline)
        CoM_x = int(M["m10"] / M["m00"])
        CoM_y = int(M["m01"] / M["m00"])
        CoM = (CoM_x, CoM_y)

        # # display results for testing purposes:
        cv2.circle(image, CoM, 8, (200, 50, 100), -1)
        # cv2.putText(image, "center of mass", (CoM_x - 50, CoM_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CIRCLE_COLOR, 2)
        cv2.drawContours(image, outline, -1, (200, 50, 100), 2)

        # Find area
        area = cv2.contourArea(outline)

        # Record Centroid and its area
        centroids[CoM] = area

        # display results for testing purposes:
        # cv2.putText(image, "Area: " + str(area), (CoM_x - 50, CoM_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CIRCLE_COLOR, 2)
        # cv2.imshow("result", image)
        return centroids


    def next_circle_position(self, image, outline):
        # Update radius and centroid
        # draw a bounding circle around the spheroid
        (centroid_x, centroid_y), radius = cv2.minEnclosingCircle(outline)
        centroid = int(centroid_x), int(centroid_y)
        radius = int(radius)

        # display results for testing purposes:
        # cv2.circle(image, centroid, radius, (0, 0, 0), thickness=2)
        # cv2.circle(image, centroid, 5, (0, 0, 0), -1)
        # cv2.putText(image, "centroid to pass to next frame", (int(centroid_x) - 75, int(centroid_y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # cv2.imshow("final", image)
        # cv2.waitKey(0)

        return centroid, radius


    '''
    Initializes coordinate and area data found in the first frame about the currently tracked cell into the 
    self.tracked_cell_data and self.tracked_cell_coords data structures
    '''
    def initialize_tracker_data(self):
        # Record data about tracked cell from previous frame
        self.tracked_cell_coords[self.tracked_cell_id] = list()
        self.tracked_cell_coords[self.tracked_cell_id].append(list(self.cell_locations[self.tracked_cell_id]))

        # Convert area to mm^2
        area_mm = self.cell_areas[self.tracked_cell_id]
        print("Initializing tracker")
        print("cell_areas[self.tracked_cell_id]:", area_mm)
        area_mm = area_mm * (self.pixels_to_mm ** 2)
        print("area_mm:", area_mm)

        self.tracked_cell_data[f'Area ({self.units}^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data[f'X Position ({self.units})'].append(coordinates_mm[0])
        self.tracked_cell_data[f'Y Position ({self.units})'].append(coordinates_mm[1])

        # Record Time from start
        self.tracked_cell_data['Time (mins)'].append((self.frame_num - 2) * self.time_between_frames)


    '''
    Determines if the given cell id relates to a known cell
    @return True if id is known/valid, and False if not
    '''
    def is_valid_id(self, cell_id:int):
        valid = False
        # Ensure that given cell id is a positive integer within the range of known ids
        if 0 <= int(cell_id) < len(self.cell_locations):
            valid = True

        return valid


    '''
    Saves a visualization of the path that cell traveled during this video
    Note: the entirety of self.vid must have already been played before calling this
    @param filename Optional path + filename to save this image to. 
    If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    @param path_color bgr values for color of the path to draw
    @param start_color bgr values for the color of the starting cell position to be drawn
    @param end_color bgr values for color of final cell position to be drawn
    '''
    def export_final_path(self, filename=None, path_color=(255, 255, 255), start_color=(255, 0, 0), end_color=(0, 0, 255)):
        if self.final_frame is None or self.first_frame is None:
            raise Exception("Video Must finish playing before exporting the cell's path")

        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Path.png"

        # Create Color Image containing the path the tracked cell took
        # Scale image to match
        final_photo = analysis.rescale_frame(self.final_frame, self.scale)

        # Draw Boundary for Cell's starting position
        final_photo = analysis.draw_initial_cell_boundary(self.first_frame, self.tracked_cell_coords[self.tracked_cell_id][0],
                                                          final_photo, start_color)
        # Draw a line for every frame of movement going from its last position to its next position
        for i in range(1, len(self.tracked_cell_coords[self.tracked_cell_id])):
            cv2.line(final_photo, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    path_color, 2)

        # Draw dot at final centroid
        cv2.circle(final_photo, self.tracked_cell_coords[self.tracked_cell_id][len(self.tracked_cell_coords[self.tracked_cell_id]) - 1], 4,
                  end_color, cv2.FILLED)

        # Save Image
        cv2.imwrite(filename, final_photo)


    '''
    Exports recorded positional, area, and statistical data about the tracked cell to an excel spreadsheet
    @param filename Optional path + filename to save this data to, should end in extension .xlsx. 
        If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_excel(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Data.xlsx"

        # Export data to excel
        export.individual_to_excel_file(filename, self.tracked_cell_data, self.time_between_frames, units=self.units, sheetname=f"Cell {self.tracked_cell_id}")


    '''
    Exports recorded positional and area data about the tracked cell to a csv file
    @param filename Optional path + filename to save this data to, should end in extension .csv. 
        If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_csv(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Cell{self.tracked_cell_id}_Data.csv"

        # Export data to excel
        export.individual_to_csv_file(filename, self.tracked_cell_data)


    '''
    Creates a line chart visualizing selected data of an individual cell
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param labels Optional. Iterable container of labels for each point
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    @param Title Optional. Title of the chart
    @param color Name of the color to plot the points with
    '''
    def export_graph(self, xaxis, yaxis, title=None, labels=None, num_labels=2, filename=None, color="blue"):
        # Use matplotlib to graph given data
        matplotlib_graphing.export_individual_cell_data(self.tracked_cell_data, xaxis, yaxis, filename, labels, num_labels, title, color)


    '''
    Creates a line chart with the tracked cell's x position on the x axis and its y position on the y axis, points are
    labeled with their respective timestamps 
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_movement_graph(self, num_labels=2, filename=None):
        self.export_graph(f"X Position ({self.units})", f"Y Position ({self.units})", f"Cell {self.tracked_cell_id}: Movement", self.tracked_cell_data["Time (mins)"], num_labels, filename)


    '''
    Creates a line chart with the tracked cell's area on the x axis and time on the y axis
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_area_graph(self, num_labels=2, filename=None):
        self.export_graph("Time (mins)", f"Area ({self.units}^2)", f"Cell {self.tracked_cell_id}: Area over Time", filename=filename)


    # Release the video source when the object is destroyed
    def __del__(self):
        if not is_image(self.source) and self.vid.isOpened():
            self.vid.release()


"""
Defines class that manages the tracking of raw data for an entire culture of cells within a video
"""
class CultureTracker:
    def __init__(self, source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=0.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):
        self.source = source
        # Open the source
        if is_image(source):
            # If given file is an image, use imread
            self.vid = cv2.imread(source)
            self.frames = 1
        else:
            # if video use VideoCapture
            self.vid = cv2.VideoCapture(source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", source)
            # Get source frames, if source is an image default to 1
            self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1


        self.height_mm = float(height_mm)
        self.width_mm = float(width_mm)
        self.area_mm = None
        # Real world time in minutes that pass between each image being taken
        self.time_between_frames = time_between_frames

        # Max/Min Size of Objects to detect as cells within video
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

        # Define Constants for video editing
        self.scale = scale
        self.contrast = contrast
        self.brightness = brightness
        self.blur_intensity = blur_intensity

        # Tracker to collect data about each cell each frame
        self.tracker = ct.CentroidTracker()

        # Keep track of last played frame's tracker data
        self.cell_locations = None
        self.cell_areas = None

        # Dictionaries to keep track of raw data collected
        self.cell_positions_mm = OrderedDict()
        self.cell_sizes_mm = OrderedDict()

    '''
    Updates the min_cell_size field
    @param min_size Positive integer pertaining to smallest size cell to track
    '''
    def set_min_size(self, min_size:int):
        self.min_cell_size = min_size


    '''
    Updates the max_cell_size field
    @param min_size Positive integer pertaining to largest size cell to track
    '''
    def set_max_size(self, max_size:int):
        self.max_cell_size = max_size


    '''
    Updates the contrast field
    '''
    def set_contrast(self, contrast):
        self.contrast = contrast


    '''
    Updates the brightness field
    '''
    def set_brightness(self, brightness):
        self.brightness = brightness


    '''
    Updates the blur_intensity field
    '''
    def set_blur_intensity(self, blur_intensity):
        self.blur_intensity = blur_intensity


    '''
    Retrieves the next frame of the current video and records data about the tracked cell. If no frame is found returns None instead
    @:returns unedited frame, edited frame
    '''
    def next_frame(self):
        if is_image(self.source):
            # Only Display Images one time
            if self.frame_num <= self.frames:
                valid = True
            else:
                valid = False
            frame = self.vid
        else:
            valid, frame = self.vid.read()
        # If next frame is not found return None
        if not valid:
            return None, None

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast,
                                            self.brightness, self.blur_intensity)

        # Detect minimum cell boundaries and their centroids for tracker
        processed, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)

        # If this is the first frame find the pixels per mm measurement
        if self.frame_num == 1:
            # Grab Frame's dimensions in order to convert pixels to mm
            (h, w) = processed.shape[:2]
            if self.pixels_to_mm is None or self.pixels_to_mm == 0:
                # If instead the dimensions of the image were given then calculate the pixel conversion using those
                # If selected units were micro meters (µm)
                if self.units == "µm":
                    self.pixels_to_mm = (((self.height_mm / h) + (self.width_mm / w)) / 2) * 1000
                    # Calculate the area of the video by multiplying the dimensions
                    self.area_mm = self.height_mm * self.width_mm * 1000
                else:
                    # Otherwise use millimeters as unit
                    self.pixels_to_mm = ((self.height_mm / h) + (self.width_mm / w)) / 2
                    # Calculate the area of the video by multiplying the dimensions
                    self.area_mm = self.height_mm * self.width_mm
            else:
                # If pixels to mm were already given, then convert it to our new scale for the image
                # If selected units were micro meters (µm)
                if self.units == "µm":
                    self.pixels_to_mm = self.pixels_to_mm * self.scale * 1000
                    # Calculate the area by converting the area in pixels to the area in micrometers
                    self.area_mm = ((h * w) * (self.pixels_to_mm ** 2)) * 1000
                else:
                    # Otherwise use millimeters as unit
                    self.pixels_to_mm = self.pixels_to_mm * self.scale
                    # Calculate the area by converting the area in pixels to the area in mm
                    self.area_mm = (h * w) * (self.pixels_to_mm ** 2)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Update Tracking information
        self.update_tracker_data()

        # Increment Frame Counter
        self.frame_num += 1

        # Return original frame and one with all cells encompassed
        return frame, processed


    '''
    Updates recorded data about the currently tracked cell based on data collected in the last frame 
    '''
    def update_tracker_data(self):
        # Record Data about Cell position, and cell size
        # Record positional data given by tracker
        for cell_id, coordinates in self.cell_locations.items():
            # If no entry exist for that cell create it
            if not (cell_id in self.cell_positions_mm):
                self.cell_positions_mm[cell_id] = list()
                # Append zeroes as placeholder until we reach our current frame.
                # So that recorded data is accurate for the frame it was found on
                for i in range(1, self.frame_num):
                    self.cell_positions_mm[cell_id].append((0, 0))

            # Convert coordinates to mm
            # Coordinates correspond to centroids distance from the left and top of the image
            coordinates_mm = list(coordinates)
            coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
            coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)

            self.cell_positions_mm[cell_id].append(coordinates_mm)

        # Record Area
        for cell_id, area in self.cell_areas.items():
            # If no entry exist for that cell create it
            if not (cell_id in self.cell_sizes_mm):
                self.cell_sizes_mm[cell_id] = list()
                # Append zeroes as placeholder until we reach our current frame.
                # So that recorded data is accurate for the frame it was found on
                for i in range(1, self.frame_num):
                    self.cell_sizes_mm[cell_id].append(0)

            # Convert area to mm^2
            area_mm = area
            area_mm = area_mm * (self.pixels_to_mm ** 2)
            self.cell_sizes_mm[cell_id].append(area_mm)


    '''
    Exports recorded positional, area, and statistical data about the tracked cell to an excel spreadsheet
    @param filename Optional path + filename to save this data to, should end in extension .xlsx. 
           If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_excel(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Culture_Data.xlsx"

        positional_headers = ["Cell ID", f"Initial X Position ({self.units})", f"Initial Y Position ({self.units})"]
        size_headers = ["Cell ID", f"Initial Size ({self.units}^2)"]

        # Generate Headers
        for i in range(2, self.frame_num):
            size_headers.append(f"Frame {i} Size")
            positional_headers.append(f"Frame {i} X Position")
            positional_headers.append(f"Frame {i} Y Position")

        # Add Final Columns for calculations
        # positional_headers.append("Distance between Initial Position and Final Position")
        size_headers.append("Final Growth")
        size_headers.append("Largest Growth in one interval")


        # Export Data to excel sheet
        export.culture_to_excel_file(filename, self.cell_positions_mm, self.cell_sizes_mm, self.time_between_frames,
                                    (self.area_mm), positional_headers, size_headers, self.units)


    '''
    Exports recorded positional and area data about the entire culture to a csv file
    @param filename Optional path + filename to save this data to, should end in extension .csv. 
        If not specified an autogenerated name will be used and the file will be placed into the user's downloads folder
    '''
    def export_to_csv(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            home_dir = os.path.expanduser("~")
            home_dir += "/Downloads/"
            timestamp = datetime.now().strftime("%b%d_%Y_%H-%M-%S")
            filename = f"{home_dir}{timestamp}_Culture_Data.csv"

        positional_headers = ["Cell ID", f"Initial X Position ({self.units})", f"Initial Y Position ({self.units})"]
        size_headers = [f"Initial Size ({self.units}^2)"]

        # Generate Headers
        for i in range(2, self.frame_num):
            size_headers.append(f"Frame {i} Size")
            positional_headers.append(f"Frame {i} X Position")
            positional_headers.append(f"Frame {i} Y Position")


        # Export data to a csv file
        export.culture_to_csv_file(filename, self.cell_positions_mm, self.cell_sizes_mm, positional_headers, size_headers)


    '''
    Creates a line chart visualizing selected data of an individual cell
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param labels Optional. Iterable container of labels for each point
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    @param Title Optional. Title of the chart
    @param color Name of the color to plot the points with
    '''
    def export_graph(self, data, xaxis, yaxis, title=None, labels=None, num_labels=2, filename=None, color="blue"):
        # Use matplotlib to graph given data
        matplotlib_graphing.export_line_chart(data, xaxis, yaxis, filename, labels, num_labels, title, color)


    '''
    Creates a line chart with the tracked cell's area on the x axis and time on the y axis
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_area_graph(self, num_labels=2, filename=None):
        graph_data = {"Time (mins)": [], f"Average Area ({self.units}^2)": []}
        # Calculate the average area of each cell per frame
        # Loop through all frames and generate the needed data
        for i in range(0, self.frame_num - 1):
            areas = []
            # Record Time at this frame for X axis
            graph_data["Time (mins)"].append(i * self.time_between_frames)

            # Loop through position dict and record the distance each cell traveled between last frame and this
            for key, data in self.cell_sizes_mm.items():
                # If value we are trying to grab is not None or 0 (placeholder value)
                if data[i] is not None and data[i] != 0:
                    # Grab area
                    areas.append(data[i])

            # Average out the recorded areas traveled this frame and append it to our list
            if areas:
                avg = float(sum(areas)/len(areas))
                graph_data[f"Average Area ({self.units}^2)"].append(avg)

        # Call Generic Export graph method with created parameters
        self.export_graph(graph_data, "Time (mins)", f"Average Area ({self.units}^2)", f"Average Area of Each Cell",
                          filename=filename)


    '''
        Creates a line chart with the tracked cell's area on the x axis and time on the y axis
        @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
        @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
              if set to 1 only the first point will be labeled
        '''
    def export_average_speed_graph(self, num_labels=2, filename=None):
        # Create Dictionary containing an entry for Time and entry for the average displacement (distance traveled between last frame and current frame)
        graph_data = {"Time (mins)": [], f"Average Speed ({self.units}/min)": []}

        # Loop through all frames and generate the needed data
        for i in range(1, self.frame_num - 1):
            speeds = []
            # Record Time at this frame for X axis
            graph_data["Time (mins)"].append(i * self.time_between_frames)
            # Loop through position dict and record the distance each cell traveled between last frame and this
            for key, data in self.cell_positions_mm.items():
                # If value we are trying to grab is not None
                if data[i] is not None:
                    # Grab x and y coordinates
                    x = data[i][0]
                    y = data[i][1]
                    prevx = data[i - 1][0]
                    prevy = data[i - 1][1]
                    # Only record displacement if coordinates are not 0 (since it's a placeholder value)
                    if prevx != 0 and x != 0:
                        # Calc speed traveled between frames
                        distance = math.dist([prevx, prevy], [x, y])
                        speed = distance/self.time_between_frames
                        speeds.append(speed)

            # Average out the recorded distances traveled this frame and append it to our list
            avg = float(sum(speeds)/len(speeds))
            graph_data[f"Average Speed ({self.units}/min)"].append(avg)

        # Call Generic Export graph method with created parameters
        self.export_graph(graph_data, "Time (mins)", f"Average Speed ({self.units}/min)", f"Average Speed",
                          filename=filename)


    '''
        Creates a line chart with the tracked cell's area on the x axis and time on the y axis
        @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
        @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
              if set to 1 only the first point will be labeled
    '''
    def export_average_displacement_graph(self, num_labels=2, filename=None):
        # Create Dictionary containing an entry for Time and entry for the average displacement (distance traveled between last frame and current frame)
        graph_data = {"Time (mins)": [], f"Average Displacement ({self.units})": []}

        # Loop through all frames and generate the needed data
        for i in range(1, self.frame_num - 1):
            distances = []
            # Record Time at this frame for X axis
            graph_data["Time (mins)"].append(i * self.time_between_frames)
            # Loop through position dict and record the distance each cell traveled between last frame and this
            for key, data in self.cell_positions_mm.items():
                # If value we are trying to grab is not None
                if data[i] is not None:
                    # Grab x and y coordinates
                    x = data[i][0]
                    y = data[i][1]
                    prevx = data[i - 1][0]
                    prevy = data[i - 1][1]
                    # Only record displacement if coordinates are not 0 (since it's a placeholder value)
                    if prevx != 0 and x != 0:
                        # Calc Distance traveled between frames
                        distance = math.dist([prevx, prevy], [x, y])
                        distances.append(distance)

            # Average out the recorded distances traveled this frame and append it to our list
            avg = sum(distances)/len(distances)
            graph_data[f"Average Displacement ({self.units})"].append(avg)

        # Call Generic Export graph method with created parameters
        self.export_graph(graph_data, "Time (mins)", f"Average Displacement ({self.units})", f"Average Displacement Between Time Intervals", filename=filename)


    # Release the video source when the object is destroyed
    def __del__(self):
        if not is_image(self.source) and self.vid.isOpened():
            self.vid.release()

'''
Determines if given file relates to an image or not
Supported file types: .jpeg, .png, .tif
'''
def is_image(filename:str):
    VALID_FILE_TYPES = [".png", ".jpeg", ".tif", ".tiff", ".jpg", ".jpe"]
    valid = False
    if os.path.exists(filename):
        if os.path.splitext(filename)[1].upper() in (ftype.upper() for ftype in VALID_FILE_TYPES):
            valid = True
    return valid


# Used for spheroid tracking, only works outside of class
def draw_circle(event, x, y, flags, param):
    """Drawing circle on image based on mouse movements
    code adapted from:
    https://pythonprogramming.net/drawing-writing-python-opencv-tutorial/
    https://www.life2coding.com/paint-opencv-images-save-image/
    https://www.tutorialspoint.com/opencv-python-how-to-draw-circles-using-mouse-events"""
    img_copy = FRAME1.copy()  # sets fresh image as canvas to clear the slate
    global IX, IY, DRAWING, PREVIOUS, START
    circle_color = (200, 50, 100)
    if event == cv2.EVENT_LBUTTONDOWN:  # when left button on mouse is clicked...
        DRAWING = True
        # take note of where the mouse was located
        IX, IY = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        DRAWING = True
    elif event == cv2.EVENT_LBUTTONUP:  # length dragged = diameter of circle
        radius = int((math.sqrt(((IX - x) ** 2) + ((IY - y) ** 2))) / 2)
        center_x = int((IX - x) / 2) + x
        center_y = int((IY - y) / 2) + y
        START = (center_x, center_y), radius
        cv2.circle(img_copy, START[0], START[1], circle_color,
                  thickness=2)  # circle: (image, (center_x, center_y), radius, color, thickness)
        DRAWING = False
        PREVIOUS = img_copy  # sets global variable to image with circle so it can be referenced outside of this method
        cv2.imshow('Drag Circle Window', img_copy)