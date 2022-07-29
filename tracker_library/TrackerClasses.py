import cv2 as cv
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing
from collections import OrderedDict
from datetime import datetime

"""
Defines class that manages the tracking of a specified individual cell
"""
class IndividualTracker:
    def __init__(self, video_source, width_mm, height_mm, time_between_frames, scale=.25, contrast=1.25, brightness=0.1,
                 blur_intensity=10):
        # Open the video source
        self.vid = cv.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)
        self.height_mm = int(height_mm)
        self.width_mm = int(width_mm)
        # Real world time in minutes that pass between each image being taken
        self.time_between_frames = time_between_frames

        # Define Constants for video editing
        self.scale = scale
        self.contrast = contrast
        self.brightness = brightness
        self.blur_intensity = blur_intensity

        # Keep track of cell id
        self.tracker = ct.CentroidTracker()
        self.tracked_cell_id = -1
        self.tracked_cell_data = {'Time': [0], 'X Position (mm)': [], 'Y Position (mm)': [], 'Area (mm^2)': []}
        self.tracked_cell_coords = OrderedDict()

        # Keep track of last played frame's tracker data
        self.cell_locations = None
        self.cell_areas = None

        # Keep Track of Photo Data for exports later
        self.pixels_to_mm = None
        self.frame_num = 1
        self.first_frame = None
        self.final_frame = None
        self.Xmin = None
        self.Ymin = None
        self.Xmax = 0
        self.Ymax = 0


    def get_frame(self):
        """
        Return the next frame
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, frame
            else:
                return ret, None
        else:
            return 0, None


    def goto_frame(self, frame_no):
        """
        Go to specific frame
        """
        if self.vid.isOpened():
            self.vid.set(cv.CAP_PROP_POS_FRAMES, frame_no)  # Set current frame
            ret, frame = self.vid.read()  # Retrieve frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None


    '''
    Retrieves the next frame of the current video and records data about the tracked cell. If no frame is found returns None instead
    @:returns unedited frame, edited frame
    '''
    def next_frame(self):
        valid, frame = self.vid.read()

        # TODO If next frame is not found return None?
        if not valid:
            return None, None

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, self.scale, self.contrast,
                                            self.brightness, self.blur_intensity)

        # Detect minimum cell boundaries and their centroids for tracker
        cont, shapes = analysis.detect_shape_v2(processed)

        # Draw a line for every frame of movement going from its last position to its next position
        for i in range(1, len(self.tracked_cell_coords[self.tracked_cell_id])):
            cv.line(processed, self.tracked_cell_coords[self.tracked_cell_id][i - 1], self.tracked_cell_coords[self.tracked_cell_id][i],
                    (255, 255, 255), 2)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Update Tracking information
        self.update_tracker_data()

        # If this is the final frame of the video record it for export later
        if self.frame_num >= self.frames:
            self.final_frame = frame

        # Increment Frame Counter
        self.frame_num += 1

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
        area_mm = self.cell_areas[self.tracked_cell_id] * (self.pixels_to_mm ** 2)
        self.tracked_cell_data['Area (mm^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data['X Position (mm)'].append(coordinates_mm[0])
        self.tracked_cell_data['Y Position (mm)'].append(coordinates_mm[1])

        # Record Time from start
        self.tracked_cell_data['Time'].append(self.frame_num * self.time_between_frames)


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
        cont, shapes = analysis.detect_shape_v2(processed)

        # Use Tracker to label and record coordinates of all cells
        self.cell_locations, self.cell_areas = self.tracker.update(shapes)

        # Label all cells with cell id
        labeled_img = analysis.label_cells(processed, self.cell_locations)

        # Grab Frame's dimensions in order to convert pixels to mm
        (h, w) = frame.shape[:2]
        self.pixels_to_mm = self.height_mm / h

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
        area_mm = self.cell_areas[self.tracked_cell_id] * (self.pixels_to_mm ** 2)
        self.tracked_cell_data['Area (mm^2)'].append(area_mm)

        # Convert Coordinates to mm
        coordinates_mm = list(self.cell_locations[self.tracked_cell_id])
        coordinates_mm[0] = float(coordinates_mm[0] * self.pixels_to_mm)
        coordinates_mm[1] = float(coordinates_mm[1] * self.pixels_to_mm)
        self.tracked_cell_data['X Position (mm)'].append(coordinates_mm[0])
        self.tracked_cell_data['Y Position (mm)'].append(coordinates_mm[1])


    '''
    Updates the currently tracked cell to that of the given ID.
    @param tracked_cell_id Positive integer pertaining to the ID of one of the tracked cells
    '''
    def set_tracked_cell(self, tracked_cell_id:int):
        self.tracked_cell_id = tracked_cell_id


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
    @param filename Optional path + filename to save this image to. If not specified an autogenerated name will be used
    @param path_color bgr values for color of the path to draw
    @param start_color bgr values for the color of the starting cell position to be drawn
    @param end_color bgr values for color of final cell position to be drawn
    '''
    def export_final_path(self, filename=None, path_color=(255, 255, 255), start_color=(255, 0, 0), end_color=(0, 0, 255)):
        if self.final_frame is None or self.first_frame is None:
            raise Exception("Video Must finish playing before exporting the cell's path")

        # Create default filename using the timestamp
        if filename is None:
            filename = f"{str(datetime.now())}_Cell{self.tracked_cell_id}_Path.png"

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
                    If not specified an autogenerated name will be used
    '''
    def export_to_excel(self, filename=None):
        # Create default filename using the timestamp
        if filename is None:
            filename = f"{str(datetime.now())}_Cell{self.tracked_cell_id}_Data.xlsx"

        # Export data to excel
        export.individual_to_excel_file(filename, self.tracked_cell_data, self.time_between_frames, f"Cell {self.tracked_cell_id}")


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
        self.export_graph("X Position (mm)", "Y Position (mm)", f"Cell {self.tracked_cell_id}: Movement", self.tracked_cell_data["Time"], num_labels, filename)


    '''
    Creates a line chart with the tracked cell's area on the x axis and time on the y axis
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
          if set to 1 only the first point will be labeled
    '''
    def export_area_graph(self, num_labels=2, filename=None):
        self.export_graph("Time", "Area (mm^2)", f"Cell {self.tracked_cell_id}: Area over Time", filename=filename)


    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
