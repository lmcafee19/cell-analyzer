'''
@brief Contains Classes to provide functionality to the two options from the GUI
    Individual Tracker: Contains functions to track a single cell within the image or video
    Culture Tracker: Contains functions to track all cells within an image or video
'''
import math
import cv2 as cv
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing
from collections import OrderedDict
from datetime import datetime

"""
Defines class that manages the tracking of a specified individual cell within a video
"""
class IndividualTracker:
    def __init__(self, video_source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):
        # TODO If given file is an image, open it a different way
        # Open the video source
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1
        self.frames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)

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
            cv.line(processed, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
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
        area_mm = area_mm * (self.pixels_to_mm**2)
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
        # Open First Frame of Video and Detect all cells within it, making sure to label them
        valid, frame = self.vid.read()
        if not valid:
            raise Exception("Video cannot be read")

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)

        # Save Reference to edited first frame for export of cell's path
        self.first_frame = processed

        # Detect minimum cell boundaries and display edited photo
        cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Label all cells with cell id
        labeled_img = analysis.label_cells(processed, self.cell_locations)

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
            cv.line(final_photo, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    path_color, 2)

        # Draw dot at final centroid
        cv.circle(final_photo, self.tracked_cell_coords[self.tracked_cell_id][len(self.tracked_cell_coords[self.tracked_cell_id]) - 1], 4,
                  end_color, cv.FILLED)

        # Save Image
        cv.imwrite(filename, final_photo)


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
        if self.vid.isOpened():
            self.vid.release()


"""
Defines class that manages the tracking of raw data for an entire culture of cells within a video
"""
class CultureTracker:
    def __init__(self, video_source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=0.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):
        # Open the video source
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1
        self.frames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)

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
        valid, frame = self.vid.read()

        # If next frame is not found return None?
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
        if self.vid.isOpened():
            self.vid.release()


"""
Defines class that manages the tracking of a specified spheroid within a video
"""
class SpheroidTracker:
    def __init__(self, video_source, time_between_frames, width_mm=0, height_mm=0, pixels_per_mm=None, min_cell_size=10, max_cell_size=600, scale=.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10, units="mm"):
        # TODO If given file is an image, open it a different way
        # Open the video source
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.pixels_to_mm = pixels_per_mm
        # Units can either be millimeter(mm) or micrometer(μm)
        # This will change the pixels_to_mm conversion and the titles of columns and exported data
        self.units = units
        self.frame_num = 1
        self.frames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)

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
            cv.line(processed, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
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
        area_mm = area_mm * (self.pixels_to_mm**2)
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
        # Open First Frame of Video and Detect all cells within it, making sure to label them
        valid, frame = self.vid.read()
        if not valid:
            raise Exception("Video cannot be read")

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast, self.brightness, self.blur_intensity)

        # Save Reference to edited first frame for export of cell's path
        self.first_frame = processed

        # Detect minimum cell boundaries and display edited photo
        cont, shapes = analysis.detect_shape_v2(processed, self.min_cell_size, self.max_cell_size)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Label all cells with cell id
        labeled_img = analysis.label_cells(processed, self.cell_locations)

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
            cv.line(final_photo, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    path_color, 2)

        # Draw dot at final centroid
        cv.circle(final_photo, self.tracked_cell_coords[self.tracked_cell_id][len(self.tracked_cell_coords[self.tracked_cell_id]) - 1], 4,
                  end_color, cv.FILLED)

        # Save Image
        cv.imwrite(filename, final_photo)


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
        if self.vid.isOpened():
            self.vid.release()