import cv2 as cv
import os
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing
from collections import OrderedDict

"""
Defines class that manages the tracking of a specified individual cell
"""
class IndividualTracker:
    def __init__(self, video_source, width_mm, height_mm, time_between_frames, scale=0.25, contrast=1.25, brightness=0.1, blur_intensity=10):
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

        # Keep Track of Photo Data for exports later
        self.pixels_to_mm = None
        self.frame_num = 0
        self.first_frame = None
        self.final_frame = None
        self.Xmin = None
        self.Ymin = None
        self.Xmax = 0
        self.Ymax = 0

    def set_tracked_cell(self, cellid):
        self.tracked_cell = cellid

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
        cell_locations, cell_areas = self.tracker.update(shapes)

        # Label all cells with cell id
        labeled_img = analysis.label_cells(processed, cell_locations)

        # Grab Frame's dimensions in order to convert pixels to mm
        (h, w) = labeled_img.shape[:2]
        self.pixels_to_mm = self.height_mm / h

        # Return unedited first frame and the img with cell ids labeled
        return frame, labeled_img

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
