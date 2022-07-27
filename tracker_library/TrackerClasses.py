import cv2
import os
from collections import OrderedDict
from tracker_library import centroid_tracker as ct

"""
Defines class that manages the tracking of a specified individual cell
"""
class IndividualTracker:
    def __init__(self, video_source, width_mm, height_mm, time_between_frames, scale=0.25, contrast=1.25, brightness=0.1, blur_intensity=10):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.height_mm = height_mm
        self.width_mm = width_mm
        # Real world time in minutes that pass between each image being taken
        self.time_between_frames = time_between_frames

        # Define Constants for video editing
        self.scale = scale
        self.contrast = contrast
        self.brightness = brightness
        self.blur_intensity = blur_intensity

        # Keep track of cell id
        self.tracker = ct.CentroidTracker()
        self.tracked_cell = None
        self.tracked_cell_data = {'Time': [0], 'X Position (mm)': [], 'Y Position (mm)': [], 'Area (mm^2)': []}
        self.tracked_cell_coords = OrderedDict()

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
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # Set current frame
            ret, frame = self.vid.read()  # Retrieve frame
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return 0, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
