import sys
import threading
import time
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import PySimpleGUI as sg


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
        self.vid = None
        self.photo = None
        self.next = "1"
        # ------ Menu Definition ------ #
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]

        # Main Menu Layout
        layout1 = [[sg.Menu(menu_def)],
                  [sg.Text('Select video')], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],                         # File Selector
                  [sg.Text('Select Type of Cell Analysis'), sg.Push(), sg.Text('Settings')],                            # Section to select type of analysis with radio buttons
                  [sg.R('Individual Cell Tracking', 1), sg.Push(),
                   sg.Text('Real World Width of the Video (mm)'), sg.Input(key="video_width_mm")],                      # Take Input for Constants
                  [sg.R('Full Culture Tracking', 1), sg.Push(),
                   sg.Text('Real World Height of the Video (mm)'), sg.Input(key="video_height_mm")],
                  [sg.Push(), sg.Text('Time Between Images (mins)'), sg.Input(key="time_between_frames")],
                  [sg.Button('Run'), sg.Button('Exit')]]

        # Video Player Layout
        layout2 = [[sg.Menu(menu_def)],
                  [sg.Text('Original Video'), sg.Push(), sg.Text('Tracker Video', justification='r')],                  # Titles for each video window
                  [sg.Canvas(size=(400, 300), key="canvas", background_color="blue"),
                   sg.Canvas(size=(400, 300), key="edited_video", background_color="blue")],                            # Windows for edited/original video to play
                  [sg.Slider(size=(30, 20), range=(0, 100), resolution=100, key="slider", orientation="h",
                             enable_events=True), sg.T("0", key="counter", size=(10, 1))],                              # Frame Slider
                  [sg.Button('Next frame'), sg.Button("Pause", key="Play")],                                            # Play/Pause Buttons, Next Frame Button
                  [sg.Button('Export Data', disabled=True), sg.Button('Exit', disabled=True)]]                          # Export/Quit buttons. Disabled by default but comes online when video is done playing

        # Cell Selection (For Individual Tracking)
        layout3 = [[sg.Menu(menu_def)],
                  [sg.Text('Select video')], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],
                  [sg.Text('Original Video'), sg.Push(), sg.Text('Tracker Video', justification='r')],                  # Titles for each video window
                  [sg.Canvas(size=(400, 300), key="canvas", background_color="blue"),
                   sg.Canvas(size=(400, 300), key="edited_video", background_color="blue")],                            # Windows for edited/original video to play
                  [sg.Slider(size=(30, 20), range=(0, 100), resolution=100, key="slider", orientation="h",
                             enable_events=True), sg.T("0", key="counter", size=(10, 1))],                              # Frame Slider
                  [sg.Text('Enter Id number of cell you wish to track:'), sg.Input(key="cell_id")],                     # Take input of Cell ID Number
                  [sg.Button('Track', key="track_individual"), sg.Button("Cancel", key="Cancel")]]                      # Run and Cancel Buttons

        # Export Data Menu
        layout4 = [[sg.Menu(menu_def)],
                   [sg.Text("Select Export Settings")],
                   [sg.Check('Export Data to Excel Sheet')],
                   [sg.Text('Excel File to Export to (.xls):'), sg.Input(key="excel_filename")],
                   [sg.Text('Select Graphs to Export:')],
                   [sg.Check('Time vs Size')],
                   [sg.Check('Movement over Time')],
                   [sg.Check('Simplified Movement')]]

        num_layouts = 4

        # ----------- Create actual layout using Columns and a row of Buttons ------------- #
        layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-'),
                   sg.Column(layout3, visible=False, key='-COL3-'), sg.Column(layout4, visible=False, key='-COL4-')],
                  [sg.Button('Cycle Layout'), sg.Button('1'), sg.Button('2'), sg.Button('3'), sg.Button('4'), sg.Button('Exit')]]

        self.window = sg.Window('Cell Analyzer', layout, resizable=True, size=(800, 600)).Finalize()
        # set return_keyboard_events=True to make hotkeys for video playback
        # Get the tkinter canvas for displaying the video
        canvas = self.window.Element("canvas")
        self.canvas = canvas.TKCanvas

        # Start video display thread
        self.load_video()

        # TODO Create Layout Variable which keeps track of which layout to display
        layout = 1
        while True:  # Main event Loop
            event, values = self.window.Read()
            print(event)

            # print(event, values)
            if event is None or event == 'Exit' or event == 'Cancel':
                """Handle exit"""
                break
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
                percent = values["slider"]/100 * self.frames
                self.set_frame(int(percent))
                # print(values["slider"])

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


if __name__ == '__main__':
    App()