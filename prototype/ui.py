import sys
import threading
import time
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import os
import PySimpleGUI as sg
from collections import OrderedDict
from tracker_library import TrackerClasses
from tracker_library import centroid_tracker as ct
from tracker_library import cell_analysis_functions as analysis
from tracker_library import export_data as export
from tracker_library import matplotlib_graphing

class App:
    """
    TODO: change slider resolution based on vid length
    TODO: make top menu actually do something :P    """
    def __init__(self):

        # ------ App states ------ #
        self.play = True  # Is the video currently playing?
        self.delay = 0.023  # Delay between frames - not sure what it should be, not accurate playback
        self.frame = 1  # Current frame
        self.frames = None  # Number of frames
        # ------ Other vars ------ #
        self.edited_vid = None
        self.vid = None
        self.photo = None
        self.edited = None
        self.next = "1"
        # ------ Tracker Instances ------- #
        self.video_player = None
        # ------ Menu Definition ------ #
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]

        # Main Menu Layout
        layout1 = [[sg.Menu(menu_def)],
                   [sg.Text('Select video')], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],  # File Selector
                   [sg.Text('Select Type of Cell Tracking'), sg.Push(), sg.Text('Tracker Settings')],
                   # Section to select type of analysis with radio buttons
                   [sg.R('Individual Cell Tracking', 1, key="individual_radio"), sg.Push(),
                    sg.Text('Real World Width of the Video (mm)'), sg.Input(key="video_width_mm")],
                   # Take Input for Constants
                   [sg.R('Full Culture Tracking', 1, key="culture_radio"), sg.Push(),
                    sg.Text('Real World Height of the Video (mm)'), sg.Input(key="video_height_mm")],
                   [sg.Push(), sg.Text('Time Between Images (mins)'), sg.Input(key="time_between_frames")],
                   [sg.Push(), sg.Text('Min Cell Size (Positive Integer. Default = 10)'), sg.Input(key="min_size")],
                   [sg.Push(), sg.Text('Max Cell Size (Positive Integer. Default = 500)'), sg.Input(key="max_size")],
                   [sg.Push(), sg.Text('Contrast (Positive Floating Point. Default = 1.25)'), sg.Input(key="contrast")],
                   [sg.Push(), sg.Text('Brightness (Positive Floating Point. 0 leaves the brightness unchanged. Default = .1)'), sg.Input(key="brightness")],
                   [sg.Push(), sg.Text('Blur Intensity (Positive Integer. Default = 10)'), sg.Input(key="blur")],
                   [sg.Button('Run'), sg.Button('Exit')]]

        # Video Player Layout
        layout2 = [[sg.Menu(menu_def)],
                   [sg.Text('Original Video'), sg.Push(), sg.Text('Tracker Video', justification='r')],
                   # Titles for each video window
                   [sg.Canvas(size=(400, 300), key="canvas", background_color="blue"),
                    sg.Canvas(size=(400, 300), key="edited_video", background_color="blue")],
                   # Windows for edited/original video to play
                   [sg.Slider(size=(30, 20), range=(0, 100), resolution=100, key="slider", orientation="h",
                              enable_events=True), sg.T("0", key="counter", size=(10, 1))],  # Frame Slider
                   [sg.Button('Next frame'), sg.Button("Pause", key="Play"),
                    sg.Button('Export Data', disabled=False),
                    sg.Button('Exit')]]  # Play/Pause Buttons, Next Frame Button
        # Export/Quit buttons. Disabled by default but comes online when video is done playing

        # Cell Selection (For Individual Tracking)
        layout3 = [[sg.Menu(menu_def)],
                   [sg.Text('Original Video'), sg.Push(), sg.Text('Tracker Video', justification='r')],
                   # Titles for each video window
                   [sg.Canvas(size=(400, 300), key="original_first_frame", background_color="blue"),
                    sg.Canvas(size=(400, 300), key="edited_first_frame", background_color="blue")],
                   # Windows for edited/original video to play
                   [sg.Text('Enter Id number of cell you wish to track:'), sg.Input(key="cell_id")],
                   # Take input of Cell ID Number
                   [sg.Button('Track', key="track_individual"), sg.Button("Exit")]]  # Run and Exit Buttons

        # Export Data Menu
        layout4 = [[sg.Menu(menu_def)],
                   [sg.Text("Select Export Settings")],
                   [sg.Check('Export Data to Excel Sheet', key='excel_export')],
                   [sg.Text('Excel File to Export to (.xls):'), sg.Input(key="excel_filename")],
                   [sg.Text('Select Graphs to Export:')],
                   [sg.Check('Time vs Size', key="Time vs Size")],
                   [sg.Check('Movement over Time', key='Movement over Time')],
                   [sg.Check('Simplified Movement', key='Simplified Movement')],
                   [sg.Text('Select Images to Export', key="images_label", visible=False)],
                   [sg.Check('Export Final Path of Tracked Cell', key="image_tracked", visible=False)],
                   [sg.Button('Export'), sg.Button("Cancel", key="Cancel")],
                   # Export Button finishes script and program, Cancel returns to previous page
                   [sg.Text("Data Currently Exporting. Application will close once process is finished",
                            key="export_message", text_color="red", visible=False)]]

        # State Constants
        MAIN_MENU = 1
        VIDEO_PLAYER = 2
        CELL_SELECTION = 3
        EXPORT = 4

        num_layouts = 4

        # ----------- Create actual layout using Columns and a row of Buttons ------------- #
        layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-'),
                   sg.Column(layout3, visible=False, key='-COL3-'), sg.Column(layout4, visible=False, key='-COL4-')],
                  [sg.Button('Cycle Layout'), sg.Button('1'), sg.Button('2'), sg.Button('3'), sg.Button('4'),
                   sg.Button('Exit')]]

        self.window = sg.Window('Cell Analyzer', layout, resizable=True, size=(800, 600)).Finalize()
        # set return_keyboard_events=True to make hotkeys for video playback
        # Get the tkinter canvas for displaying the video
        canvas = self.window.Element("canvas")
        self.canvas = canvas.TKCanvas
        self.edited_canvas = self.window.Element("edited_video").TKCanvas
        self.first_frame_orig = self.window.Element("original_first_frame").TKCanvas
        self.first_frame_edited = self.window.Element("edited_first_frame").TKCanvas

        # Start video display thread
        self.load_video()

        layout = 1
        while True:  # Main event Loop
            event, values = self.window.Read()
            print(event, values)

            # ---- Global Events ---- #
            # Event to change layout, at the moment just jumps to the next layout
            if event == 'Cycle Layout':
                self.window[f'-COL{layout}-'].update(visible=False)
                layout = ((layout + 1) % num_layouts)
                if layout == 0:
                    layout += 1
                self.window[f'-COL{layout}-'].update(visible=True)
            elif event in '1234':
                self.window[f'-COL{layout}-'].update(visible=False)
                layout = int(event)
                self.window[f'-COL{layout}-'].update(visible=True)

            # Exit Event
            if event is None or event.startswith('Exit'):
                """Handle exit"""
                break

            # ---- Main Menu Events ---- #
            # File Selection Browse Button
            if event == "Browse":
                """Browse for files when the Browse button is pressed"""
                # Open a file dialog and get the file path
                video_path = None
                try:
                    video_path = sg.filedialog.askopenfile().name
                except AttributeError:
                    print("no video selected, doing nothing")

                if video_path:
                    print(video_path)
                    # Initialize video
                    self.vid = MyVideoCapture(video_path)
                    # Calculate new video dimensions
                    self.vid_width = 500
                    self.vid_height = int(self.vid_width * self.vid.height / self.vid.width)
                    # print("old par: %f" % (self.vid.width / self.vid.height))
                    # print("new par: %f" % (self.vid_width / self.vid_height))
                    # print(self.vid.fps)
                    # print(int(self.vid.frames))
                    self.frames = int(self.vid.frames)

                    # Update slider to match amount of frames
                    self.window.Element("slider").Update(range=(0, int(self.frames)), value=0)
                    # Update right side of counter
                    self.window.Element("counter").Update("0/%i" % self.frames)
                    # change canvas size approx to video size
                    self.canvas.config(width=self.vid_width, height=self.vid_height)

                    # Reset frame count
                    self.frame = 0
                    self.delay = 1 / self.vid.fps

                    # Update the video path text field
                    self.window.Element("_FILEPATH_").Update(video_path)

            # Check input values then run subsequent tracking script
            if event == "Run":
                # Grab References to each field
                file = self.window["_FILEPATH_"].get()
                width = self.window["video_width_mm"].get()
                height = self.window["video_height_mm"].get()
                mins = self.window["time_between_frames"].get()

                # Check that all fields have been filled out with valid data then determine next action based on tracking type
                if isValidParameters(file, width, height, mins):
                    # TODO maybe Initialize Video. This will likely be covered by indiv/culture trackers

                    # If individual tracking has been selected
                    if self.window.Element("individual_radio").get():
                        # Initialize Individual Tracker
                        self.video_player = TrackerClasses.IndividualTracker(file, width, height, mins)

                        # Continue to Individual Cell Selection Page
                        self.window[f'-COL{MAIN_MENU}-'].update(visible=False)
                        self.window[f'-COL{CELL_SELECTION}-'].update(visible=True)

                        # TODO Display First Frame of Edited and UnEdited Video on Cell Selection View
                        self.display_first_frame(self.video_player)


                    # Culture Tracking is selected
                    elif self.window.Element("culture_radio").get():
                        # Initialize Culture Tracker
                        #self.video_player = TrackerClasses.CultureTracker(file, width, height, mins)

                        # Continue to video player page
                        self.window[f'-COL{MAIN_MENU}-'].update(visible=False)
                        self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=True)

                        # TODO Play Unedited and Edited Video on Video Player View

                    # No Method is Selected do not run
                    else:
                        sg.PopupError("Method of Tracking must be selected before running")

                # If all Required field are not filled or have invalid input, show popup
                else:
                    sg.PopupError("Invalid Input in the Fields.")

            # ---- Cell Selection Events ---- #
            if event == "track_individual":
                # Ensure Valid Cell ID has been entered
                #
                if not isValidID(values['cell_id']):
                    sg.PopupError("Invalid or Missing Cell ID Number")
                # Valid Cell Id given move onto video player with appropriate tracking values
                else:
                    # Continue to video player page
                    self.window[f'-COL{CELL_SELECTION}-'].update(visible=False)
                    self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=True)

                    # TODO Play Unedited and Edited Video on Video Player View

            # ---- Video Player Events ---- #
            if event == "Play":
                if self.play:
                    self.play = False
                    self.window.Element("Play").Update("Play")
                else:
                    self.play = True
                    self.window.Element("Play").Update("Pause")

            if event == 'Next frame':
                # Jump forward a frame TODO: let user decide how far to jump
                self.set_frame(self.frame + 1)

            if event == "slider":
                # self.play = False
                # Set video to frame at percentage of slider
                percent = values["slider"] / 100 * self.frames
                self.set_frame(int(percent))
                # print(values["slider"])

            # TODO set trigger within video player to enable the export/exit buttons when finished playing video
            if event == "video_finished":
                self.window["Export Data"].update(visible=True)

            if event == "Export Data":
                # Continue to export interface
                self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=False)
                self.window[f'-COL{EXPORT}-'].update(visible=True)
                # Enable individual cell tracking specifics exports if it meets the reqs
                if self.window.Element("individual_radio").get():
                    self.window['images_label'].update(visible=True)
                    self.window['image_tracked'].update(visible=True)

            # ---- Export Events ---- #
            if event == "Export":
                # Display Export Message
                self.window['export_message'].update(visible=True)

                # Grab all values for exports
                exportExcel = self.window.Element("excel_export").get()
                excelfile = self.window.Element("excel_filename").get()
                exportgraph_size = self.window.Element("Time vs Size").get()
                exportgraph_movement = self.window.Element("Movement over Time").get()
                exportgraph_simple = self.window.Element("Simplified Movement").get()

                # TODO Call export functions based on what boxes were checked
                # Continue Script and Export Data

                # Close app once Export is finished
                break

            # Return to previous page
            if event == "Cancel":
                # Continue to export interface
                self.window[f'-COL{EXPORT}-'].update(visible=False)
                self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=True)

        # Exiting
        print("bye :)")
        self.window.Close()
        sys.exit()


    #################
    # Video methods #
    #################
    def load_video(self):
        """Start video display in a new thread"""
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = 1
        thread.start()

    def update(self):
        """Update the canvas element with the next video frame recursively"""
        start_time = time.time()
        if self.vid:
            if self.play:

                # Get a frame from the video source only if the video is supposed to play
                ret, frame = self.vid.get_frame()

                if ret:
                    self.photo = PIL.ImageTk.PhotoImage(
                        image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST)
                    )
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                    self.frame += 1
                    self.update_counter(self.frame)

            # Uncomment these to be able to manually count fps
            # print(str(self.next) + " It's " + str(time.ctime()))
            # self.next = int(self.next) + 1
        # The tkinter .after method lets us recurse after a delay without reaching recursion limit. We need to wait
        # between each frame to achieve proper fps, but also count the time it took to generate the previous frame.
        self.canvas.after(abs(int((self.delay - (time.time() - start_time)) * 1000)), self.update)

    def set_frame(self, frame_no):
        """Jump to a specific frame"""
        if self.vid:
            # Get a frame from the video source only if the video is supposed to play
            ret, frame = self.vid.goto_frame(frame_no)
            self.frame = frame_no
            self.update_counter(self.frame)

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(
                    image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_counter(self, frame):
        """Helper function for updating slider and frame counter elements"""
        self.window.Element("slider").Update(value=frame)
        self.window.Element("counter").Update("{}/{}".format(frame, self.frames))

    '''
        Outputs first frame to the cell selection screen
    '''

    def display_first_frame(self, individual_tracker: TrackerClasses.IndividualTracker):
        # Grab First Frame from video
        ret, frame = individual_tracker.get_frame()

        # Process Image to better detect cells
        processed = analysis.process_image(frame, analysis.Algorithm.CANNY, individual_tracker.scale,
                                           individual_tracker.contrast, individual_tracker.brightness,
                                           individual_tracker.blur_intensity)

        # Detect minimum cell boundaries and display edited photo
        cont, shapes = analysis.detect_shape_v2(processed)

        # Use Tracker to label and record coordinates of all cells
        cell_locations, cell_areas = individual_tracker.tracker.update(shapes)

        # Label all cells with cell id
        processed = analysis.label_cells(processed, cell_locations)

        # Calculate new video dimensions
        self.vid_width = 400
        self.vid_height = 300
        self.frames = int(self.vid.frames)

        # Update slider to match amount of frames
        self.window.Element("slider").Update(range=(0, int(self.frames)), value=0)
        # Update right side of counter
        self.window.Element("counter").Update("0/%i" % self.frames)
        # change canvas size approx to video size
        #self.canvas.config(width=self.vid_width, height=self.vid_height)

        # Reset frame count
        self.frame = 0
        self.delay = 1 / self.vid.fps

        # Display Original photo in left frame of selected view
        # scale image to fit inside the frame
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(frame).resize((self.vid_width, self.vid_height), Image.NEAREST)
        )
        self.first_frame_orig.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Display edited photo in right frame of selected window
        self.edited = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(processed).resize((self.vid_width, self.vid_height), Image.NEAREST)
        )
        self.first_frame_edited.create_image(0, 0, image=self.edited, anchor=tk.NW)

        self.frame += 1
        self.update_counter(self.frame)



class MyVideoCapture:
    """
    Defines a new video loader with openCV
    Original code from https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
    Modified by me
    """

    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        """
        Return the next frame
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


'''
    Checks if the given tracker parameters are valid
    @param videofile The path to the video file to check
    @param width Width of the video frame in mm
    @param height Height of the video frame in mm
    @time_between_frames Time in minutes between each image in the video
    @return True if all given parameters are valid, false if not
'''
def isValidParameters(videofile, width, height, time_between_frames):
    valid = False

    if isValidVideo(videofile) and isValidDimensions(width, height) and isValidTime(time_between_frames):
        valid = True

    return valid


'''
    Checks if the given video file is of correct file type and can be opened by opencv
    @param videofile The path to the video file to check
    @return True if the tracker can analyze it, false if it cannot
'''
def isValidVideo(videofile):
    valid = False
    if os.path.exists(videofile) and videofile.endswith(".mp4"):
        valid = True
    return valid


'''
    Checks if the given dimensions are positive integers
    @param width Width in mm should be a positive integer
    @return True if valid, false if not
'''
def isValidDimensions(width, height):
    valid = False
    try:
        width = int(width)
        height = int(height)
        if 0 < width and 0 < height:
            valid = True
    except ValueError:
        valid = False

    return valid


'''
    Checks if the given number of minutes in a positive integer
    @param mins 
    @return True if valid, false if not
'''
def isValidTime(mins):
    valid = False
    try:
        val = int(mins)
        if 0 < val:
            valid = True
    except ValueError:
        valid = False

    return valid


'''
    Checks if the given cellID is an integer corresponding to a known cell
    @param cellID Integer
    @return True if cellID is
'''
def isValidID(cellID):
    # return (0 <= int(cellID) < numcells (maybe found from len(Cell_locations)
    try:
        valid = 0 <= int(cellID)
    except ValueError:
        valid = False
    return valid


if __name__ == '__main__':
    App()
