import sys
import threading
import time
import tkinter as tk
import PIL
from PIL import Image, ImageTk
import cv2
import os
import re
import PySimpleGUI as sg
from tracker_library import TrackerClasses


# State Constants
MAIN_MENU = 1
VIDEO_PLAYER = 2
CELL_SELECTION = 3
EXPORT = 4

class App:
    """
    TODO: change slider resolution based on vid length
    TODO: make top menu actually do something :P    """
    def __init__(self):

        # ------ App states ------ #
        self.play = True  # Is the video currently playing?
        self.delay = 1000  # Delay between frames in ms
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
        # ------ Theme Settings ------ #
        TITLE_COLOR = "#023e8a"
        BACKGROUND_COLOR = "#EDE7E3"
        sg.theme_background_color(BACKGROUND_COLOR)
        sg.theme_text_element_background_color(BACKGROUND_COLOR)
        sg.theme_text_color("#0096c7")
        sg.theme_button_color("#023e8a")

        # ------ Menu Definition ------ #
        menu_def = [['&File', ['&Open', '&Save', '---', 'Properties', 'E&xit']],
                    ['&Edit', ['Paste', ['Special', 'Normal', ], 'Undo'], ],
                    ['&Help', '&About...']]

        # Main Menu Layout
        layout1 = [[sg.Menu(menu_def)],
                   [sg.Text('Select Video:', text_color=TITLE_COLOR)], [sg.Input(key="_FILEPATH_"), sg.Button("Browse")],  # File Selector
                   [sg.Text('Select Playback Speed:', text_color=TITLE_COLOR), sg.Push(), sg.Text('Tracker Settings. *Required', text_color=TITLE_COLOR)],
                   [sg.R('1x', 2, key="playback_radio_1x", default=True, background_color=BACKGROUND_COLOR), sg.R('2x', 2, key="playback_radio_2x", background_color=BACKGROUND_COLOR), sg.R('5x', 2, key="playback_radio_5x", background_color=BACKGROUND_COLOR), sg.R('10x', 2, key="playback_radio_10x", background_color=BACKGROUND_COLOR), sg.Push(),
                    sg.Text('Real World Width of the Video (mm)*'), sg.Input(key="video_width_mm")],
                   [sg.Text('Select Type of Cell Tracking:', text_color=TITLE_COLOR), sg.Push(),
                    sg.Text('Real World Height of the Video (mm)*'), sg.Input(key="video_height_mm")],
                   # Section to select type of analysis with radio buttons
                   [sg.R('Individual Cell Tracking', 1, key="individual_radio", background_color=BACKGROUND_COLOR), sg.Push(), sg.Text('Number of pixels per mm* (If height/width are unknown)'), sg.Input(key='pixels_per_mm')],
                   # Take Input for Constants
                   [sg.R('Full Culture Tracking', 1, key="culture_radio", background_color=BACKGROUND_COLOR), sg.Push(), sg.Text('Time Between Images (mins)*'), sg.Input(key="time_between_frames")],
                   [sg.Push(), sg.Text('Min Cell Size (Default = 10)'), sg.Input(key="min_size")],
                   [sg.Push(), sg.Text('Max Cell Size (Default = 500)'), sg.Input(key="max_size")],
                   [sg.Push(), sg.Text('Video Editor Settings', text_color=TITLE_COLOR)],
                   [sg.Push(), sg.Text('Contrast (Default = 1.25)'), sg.Input(key="contrast")],
                   [sg.Push(), sg.Text('Brightness (0 leaves the brightness unchanged. Default = .1)'), sg.Input(key="brightness")],
                   [sg.Push(), sg.Text('Blur Intensity (Default = 10)'), sg.Input(key="blur")],
                   [sg.Button('Run'), sg.Button('Exit')]]

        # Video Player Layout
        layout2 = [[sg.Menu(menu_def)],
                   [sg.Text('Original Video', text_color=TITLE_COLOR), sg.Push(), sg.Text('Tracker Video', text_color=TITLE_COLOR, justification='r')],
                   # Titles for each video window
                   [sg.Canvas(size=(400, 300), key="canvas", background_color="blue"),
                    sg.Canvas(size=(400, 300), key="edited_video", background_color="blue")],
                   # Windows for edited/original video to play
                   [sg.T("0", key="counter", size=(10, 1))], # Frame Counter
                   [sg.Button("Pause", key="Play"), sg.Button('Next frame'),
                    sg.Push(),
                    sg.Button('Export Data', disabled=True),
                    sg.Button('Exit')]]  # Play/Pause Buttons, Next Frame Button
        # Export/Quit buttons. Disabled by default but comes online when video is done playing

        # Cell Selection (For Individual Tracking)
        layout3 = [[sg.Menu(menu_def)],
                   [sg.Text('Original Video', text_color=TITLE_COLOR), sg.Push(), sg.Text('Tracker Video', text_color=TITLE_COLOR, justification='r')],
                   # Titles for each video window
                   [sg.Canvas(size=(400, 300), key="original_first_frame", background_color="blue"),
                    sg.Canvas(size=(400, 300), key="edited_first_frame", background_color="blue")],
                   # Windows for edited/original video to play
                   [sg.Text('Enter Id number of cell you wish to track:'), sg.Input(key="cell_id")],
                   # Take input of Cell ID Number
                   [sg.Button('Track', key="track_individual"), sg.Button("Exit")]]  # Run and Exit Buttons

        # Export Data Menu
        layout4 = [[sg.Menu(menu_def)],
                   [sg.Text("Select Export Settings", text_color=TITLE_COLOR)],
                   [sg.Check('Export Data to Excel Sheet', key='excel_export', enable_events=True, background_color=BACKGROUND_COLOR)],
                   [sg.Text('Excel File to Export to (.xlsx):\n  Leave blank for auto generated filename', key='excel_file_label', visible=False), sg.Input(key="excel_filename", visible=False)],
                   [sg.Text('Select Graphs to Export:', text_color=TITLE_COLOR)],

                   [sg.Check('Area over Time', key="Area over Time", enable_events=True, background_color=BACKGROUND_COLOR)],
                   [sg.Text('Filename to save graph to (.pdf):\n  Leave blank for customizable settings and manual save after graph creation',
                            key='area_graph_label', visible=False), sg.Input(key="area_graph_filename", visible=False)],

                   # Individual Tracker Specific Export Items
                   [sg.Check('Movement over Time', key='Movement over Time', enable_events=True, visible=False, background_color=BACKGROUND_COLOR)],
                   [sg.Text('Number of Points to Label.\n  By default only the First and Last point will be labeled, reducing this number improves visual clarity', key="num_labels_desc", visible=False), sg.Input(key='num_labels', visible=False)],
                   [sg.Text(
                       'Filename to save graph to (.pdf):\n  Leave blank for customizable settings and manual save after graph creation',
                       key='individual_movement_graph_label', visible=False), sg.Input(key="individual_movement_graph_filename", visible=False)],

                   [sg.Text('Select Images to Export', key="images_label", visible=False, text_color=TITLE_COLOR)],
                   [sg.Check('Export Final Path of Tracked Cell', key="path_image", visible=False, enable_events=True, background_color=BACKGROUND_COLOR)],
                   [sg.Text(
                       'Filename to save image to (.png):\n  Leave blank for autogenerated filename',
                       key='final_path_image_label', visible=False), sg.Input(key="final_path_image_filename", visible=False)],

                   # Culture Tracker Specific Export Items
                   [sg.Check('Average Displacement', key="average_displacement", visible=False, enable_events=True, background_color=BACKGROUND_COLOR)],
                   [sg.Text(
                       'Filename to save graph to (.pdf):\n  Leave blank for customizable settings and manual save after graph creation',
                       key='culture_displacement_graph_label', visible=False), sg.Input(key="culture_displacement_graph_filename", visible=False)],

                   [sg.Check('Average Speed', key="average_speed", visible=False, enable_events=True, background_color=BACKGROUND_COLOR)],
                   [sg.Text(
                       'Filename to save graph to (.pdf):\n  Leave blank for customizable settings and manual save after graph creation',
                       key='culture_speed_graph_label', visible=False), sg.Input(key="culture_speed_graph_filename", visible=False)],

                   [sg.Button('Export'), sg.Button("Cancel", key="Cancel")],
                   # Export Button finishes script and program, Cancel returns to previous page
                   [sg.Text("Data Currently Exporting. Application will close once process is finished",
                            key="export_message", text_color="red", visible=False)]]


        num_layouts = 4

        # ----------- Create actual layout using Columns and a row of Buttons ------------- #
        layout = [[sg.Column(layout1, key='-COL1-'), sg.Column(layout2, visible=False, key='-COL2-'),
                   sg.Column(layout3, visible=False, key='-COL3-'), sg.Column(layout4, visible=False, key='-COL4-')],
                  [sg.Button('Cycle Layout'), sg.Button('1'), sg.Button('2'), sg.Button('3'), sg.Button('4'),
                   sg.Button('Exit')]]

        # Get User's screen size and set window size and scale accordingly
        screen_width, screen_height = sg.Window.get_screen_size()
        print(screen_width, screen_height)
        screen_scaling = get_scaling()

        sg.set_options(scaling=screen_scaling)
        self.window = sg.Window('Cell Analyzer', layout, resizable=True, size=(screen_width, screen_height)).Finalize()

        # Get the tkinter canvases  for displaying the video
        self.canvas = self.window.Element("canvas").TKCanvas
        self.edited_canvas = self.window.Element("edited_video").TKCanvas
        self.first_frame_orig = self.window.Element("original_first_frame").TKCanvas
        self.first_frame_edited = self.window.Element("edited_first_frame").TKCanvas

        layout = 1
        running = True
        # Main event Loop
        while running:
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
                running = False

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

                    # Update right side of counter
                    self.window.Element("counter").Update("0/%i" % self.frames)
                    # change canvas size approx to video size
                    #self.canvas.config(width=self.vid_width, height=self.vid_height)

                    # Reset frame count
                    self.frame = 1

                    # Update the video path text field
                    self.window.Element("_FILEPATH_").Update(video_path)

            # Check input values then run subsequent tracking script
            if event == "Run":
                # Grab References to each field
                file = self.window["_FILEPATH_"].get()
                width = self.window["video_width_mm"].get()
                height = self.window["video_height_mm"].get()
                pixels_per_mm = self.window["pixels_per_mm"].get()
                mins = self.window["time_between_frames"].get()
                min_size = self.window["min_size"].get()
                max_size = self.window["max_size"].get()
                contrast = self.window["contrast"].get()
                brightness = self.window["brightness"].get()
                blur = self.window["blur"].get()

                # Check that all fields have been filled out with valid data then determine next action based on tracking type
                if isValidParameters(file, width, height, mins, pixels_per_mm, min_size, max_size, contrast, brightness, blur):

                    # Set Playback Speed
                    # If 1x or no option is selected keep delay as is
                    # If 2x is selected cut delay in half
                    if self.window.Element("playback_radio_2x").get():
                        self.delay = self.delay / 2
                    # if 5x is selected / 5
                    elif self.window.Element("playback_radio_5x").get():
                        self.delay = self.delay / 5
                    # If 10x is selected, / delay by 10
                    elif self.window.Element("playback_radio_10x").get():
                        self.delay = self.delay / 10

                    # If individual tracking has been selected
                    if self.window.Element("individual_radio").get():
                        # Initialize Individual Tracker with given arguments
                        # If valid pixels per mm were given then call the individual tracker with that parameter
                        if isValidPixels(pixels_per_mm):
                            self.video_player = TrackerClasses.IndividualTracker(file, int(mins),
                                                                                 pixels_per_mm=float(pixels_per_mm))
                        else:
                            # Otherwise call it with the video's height/width
                            self.video_player = TrackerClasses.IndividualTracker(file, int(mins), width_mm=float(width), height_mm=float(height))

                        # Set all extra input arguments if they are valid
                        if isValidInt(min_size) and (min_size != "" and min_size is not None):
                            self.video_player.set_min_size(int(min_size))
                        if isValidInt(max_size) and (max_size != "" and max_size is not None):
                            self.video_player.set_max_size(int(max_size))
                        if isValidFloat(contrast) and (contrast != "" and contrast is not None):
                            self.video_player.set_contrast(float(contrast))
                        if isValidFloat(brightness) and (brightness != "" and brightness is not None):
                            self.video_player.set_brightness(float(brightness))
                        if isValidInt(blur) and blur != "" and blur is not None:
                            self.video_player.set_blur_intensity(int(blur))


                        # Continue to Individual Cell Selection Page
                        self.window[f'-COL{MAIN_MENU}-'].update(visible=False)
                        self.window[f'-COL{CELL_SELECTION}-'].update(visible=True)

                        # Display First Frame of Edited and UnEdited Video on Cell Selection View
                        self.display_first_frame()

                        # Start video display thread
                        self.load_video()


                    # Culture Tracking is selected
                    elif self.window.Element("culture_radio").get():
                        # Initialize Culture Tracker
                        # If valid pixels per mm were given then call the individual tracker with that parameter
                        if isValidPixels(pixels_per_mm):
                            self.video_player = TrackerClasses.CultureTracker(file, int(mins),
                                                                                 pixels_per_mm=float(pixels_per_mm))
                        else:
                            # Otherwise call it with the video's height/width
                            self.video_player = TrackerClasses.CultureTracker(file, int(mins), width_mm=float(width),
                                                                                 height_mm=float(height))

                        # Set all extra input arguments if they are valid
                        if isValidInt(min_size) and (min_size != "" and min_size is not None):
                            self.video_player.set_min_size(int(min_size))
                        if isValidInt(max_size) and (max_size != "" and max_size is not None):
                            self.video_player.set_max_size(int(max_size))
                        if isValidFloat(contrast) and (contrast != "" and contrast is not None):
                            self.video_player.set_contrast(float(contrast))
                        if isValidFloat(brightness) and (brightness != "" and brightness is not None):
                            self.video_player.set_brightness(float(brightness))
                        if isValidInt(blur) and blur != "" and blur is not None:
                            self.video_player.set_blur_intensity(int(blur))

                        # Continue to video player page
                        self.window[f'-COL{MAIN_MENU}-'].update(visible=False)
                        self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=True)

                        # Start video display thread
                        self.load_video()

                    # No Method is Selected do not run
                    else:
                        sg.PopupError("Method of Tracking must be selected before running")

            # ---- Cell Selection Events ---- #
            if event == "track_individual":
                # When Track button is pressed update the individual tracker to keep track of the input cell
                # and then attempt to move forward to video player stage

                selected = self.select_cell()

                # If user has entered a valid cell id and the tracker has been updated Continue to video player page
                if selected:
                    self.window[f'-COL{CELL_SELECTION}-'].update(visible=False)
                    self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=True)
                    # Video should start playing due to self.update method


            # ---- Video Player Events ---- #
            if event == "Play":
                if self.play:
                    self.play = False
                    self.window.Element("Play").Update("Play")
                else:
                    self.play = True
                    self.window.Element("Play").Update("Pause")

            if event == 'Next frame':
                # Jump forward a frame
                self.set_frame(self.frame + 1)

            if event == "Export Data":
                # Continue to export interface
                self.window[f'-COL{VIDEO_PLAYER}-'].update(visible=False)
                self.window[f'-COL{EXPORT}-'].update(visible=True)
                # Enable individual cell tracking specifics exports if it meets the reqs
                if self.window.Element("individual_radio").get():
                    self.window['Movement over Time'].update(visible=True)
                    self.window['images_label'].update(visible=True)
                    self.window['path_image'].update(visible=True)

                # Enable culture tracker specific exports
                elif self.window.Element("culture_radio").get():
                    self.window['average_displacement'].update(visible=True)
                    self.window['average_speed'].update(visible=True)


            # ---- Export Events ---- #
            if event == "Export":
                # Grab all values for exports
                export_excel = self.window.Element("excel_export").get()
                excelfile = self.window.Element("excel_filename").get()

                exportgraph_area = self.window.Element("Area over Time").get()
                area_graph_filename = self.window.Element("area_graph_filename").get()


                # Individual Tracker Specific Checkboxes
                exportgraph_movement = self.window.Element("Movement over Time").get()
                num_labels = self.window.Element("num_labels").get()
                individual_movement_graph_filename = self.window.Element("individual_movement_graph_filename").get()

                exportpath_image = self.window.Element("path_image").get()
                path_image_filename = self.window.Element("final_path_image_filename").get()

                # Culture Tracker Specific Checkboxes
                export_average_displacement = self.window.Element("average_displacement").get()
                culture_displacement_graph_filename = self.window.Element("culture_displacement_graph_filename").get()

                export_average_speed = self.window.Element("average_speed").get()
                culture_speed_graph_filename = self.window.Element("culture_speed_graph_filename").get()

                # Check if all entered filenames are formatted correctly
                valid_filenames = True

                # If excel filename field is entered
                if excelfile != '' and excelfile is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidExcelFilename(excelfile):
                        valid_filenames = False
                        sg.PopupError("Given Excel File Name is in an incorrect format.\nEnsure the filename ends in "
                                      ".xlsx or leave the field blank for an autogenerated name")

                # If area graph filename is entered
                if valid_filenames and area_graph_filename != '' and area_graph_filename is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidGraphFilename(area_graph_filename):
                        valid_filenames = False
                        sg.PopupError("Given Area over Time Graph's File Name is in an incorrect format.\nEnsure the filename ends in "
                                      ".pdf or leave the field blank for manual editing of settings and saving")

                # If individual movement graph filename is given
                if valid_filenames and individual_movement_graph_filename != '' and individual_movement_graph_filename is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidGraphFilename(individual_movement_graph_filename):
                        valid_filenames = False
                        sg.PopupError(
                            "Given Movement Graph's File Name is in an incorrect format.\nEnsure the filename ends in "
                            ".pdf or leave the field blank for manual editing of settings and saving")

                # If Export individual path image filename is entered:
                if valid_filenames and path_image_filename != '' and path_image_filename is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidImageFilename(path_image_filename):
                        valid_filenames = False
                        sg.PopupError("Given Path Image filename is in an incorrect format.\nEnsure the filename ends in "
                                      ".png or leave the field blank for an autogenerated name")

                # If culture displacement graph filename is entered
                if valid_filenames and culture_displacement_graph_filename != '' and culture_displacement_graph_filename is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidGraphFilename(culture_displacement_graph_filename):
                        valid_filenames = False
                        sg.PopupError(
                            "Given Average Displacement Graph's File Name is in an incorrect format.\nEnsure the filename ends in "
                            ".pdf or leave the field blank for manual editing of settings and saving")

                # if culture speed graph filename is entered
                if valid_filenames and culture_speed_graph_filename != '' and culture_speed_graph_filename is not None:
                    # If given filename is incorrectly formatted display error and do not export
                    if not isValidGraphFilename(culture_speed_graph_filename):
                        valid_filenames = False
                        sg.PopupError(
                            "Given Average Speed Graph's File Name is in an incorrect format.\nEnsure the filename ends in "
                            ".pdf or leave the field blank for manual editing of settings and saving")


                # Validate Inputs as needed
                if isValidExportParameters(num_labels) and valid_filenames:
                    # Display Export Message
                    self.window['export_message'].update(visible=True)

                    # Continue Script and Export Data
                    # If export raw excel data was selected call excel export data
                    if export_excel:
                        # If a filename was supplied pass it as a parameter
                        if excelfile != '' and excelfile is not None:
                            self.video_player.export_to_excel(excelfile)
                        else:
                            self.video_player.export_to_excel()

                    # Create Area vs Time graph
                    if exportgraph_area:
                        # If a filename was supplied pass it as a parameter
                        if area_graph_filename != '' and area_graph_filename is not None:
                            self.video_player.export_area_graph(filename=area_graph_filename)
                        else:
                            self.video_player.export_area_graph()

                    # Individual Tracker Exports
                    if self.window.Element("individual_radio").get():
                        # Create X vs Y Position graph
                        if exportgraph_movement:
                            # If a number of labels was supplied pass it as a parameter
                            if num_labels != '' and num_labels is not None:
                                # If a filename was supplied pass it as a parameter
                                if individual_movement_graph_filename != '' and individual_movement_graph_filename is not None:
                                    self.video_player.export_movement_graph(num_labels=int(num_labels), filename=individual_movement_graph_filename)
                                else:
                                    self.video_player.export_movement_graph(num_labels=int(num_labels))
                            else:
                                # If a filename was supplied pass it as a parameter
                                if individual_movement_graph_filename != '' and individual_movement_graph_filename is not None:
                                    self.video_player.export_movement_graph(filename=individual_movement_graph_filename)
                                else:
                                    self.video_player.export_movement_graph()

                        # Image Exports
                        if exportpath_image:
                            # If a filename was supplied pass it as a parameter
                            if path_image_filename != '' and path_image_filename is not None:
                                self.video_player.export_final_path(filename=path_image_filename)
                            else:
                                self.video_player.export_final_path()

                    elif self.window.Element("culture_radio").get():
                        # Create Average Displacement Graph
                        if export_average_displacement:
                            # If a filename was supplied pass it as a parameter
                            if culture_displacement_graph_filename != '' and culture_displacement_graph_filename is not None:
                                self.video_player.export_average_displacement_graph(filename=culture_displacement_graph_filename)
                            else:
                                self.video_player.export_average_displacement_graph()

                        # Create Average Speed Graph
                        if export_average_speed:
                            # If a filename was supplied pass it as a parameter
                            if culture_speed_graph_filename != '' and culture_speed_graph_filename is not None:
                                self.video_player.export_average_speed_graph(filename=culture_speed_graph_filename)
                            else:
                                self.video_player.export_average_speed_graph()

                    # Close app once Export is finished
                    running = False
                else:
                    # Invalid Parameters were given
                    sg.popup_error("Invalid Export Parameters")

            # Checkbox Events. When option is checked, display additional settings as needed for each export
            if event == "excel_export":
                # When Excel Export Checkbox is checked enable the input for a filename
                if self.window['excel_filename'].visible:
                    self.window['excel_file_label'].update(visible=False)
                    self.window['excel_filename'].update(visible=False)
                else:
                    self.window['excel_file_label'].update(visible=True)
                    self.window['excel_filename'].update(visible=True)

            if event == "Area over Time":
                # When area graph Export Checkbox is checked enable the input for a filename
                if self.window['area_graph_filename'].visible:
                    self.window['area_graph_label'].update(visible=False)
                    self.window['area_graph_filename'].update(visible=False)
                else:
                    self.window['area_graph_label'].update(visible=True)
                    self.window['area_graph_filename'].update(visible=True)


            # Individual tracker exports
            if event == "Movement over Time":
                # When graph for movement over time checkbox is checked enable the input for num labels and filename
                if self.window['num_labels_desc'].visible:
                    self.window['num_labels_desc'].update(visible=False)
                    self.window['num_labels'].update(visible=False)
                    self.window['individual_movement_graph_label'].update(visible=False)
                    self.window['individual_movement_graph_filename'].update(visible=False)
                else:
                    self.window['num_labels_desc'].update(visible=True)
                    self.window['num_labels'].update(visible=True)
                    self.window['individual_movement_graph_label'].update(visible=True)
                    self.window['individual_movement_graph_filename'].update(visible=True)

            if event == "path_image":
                # When area graph Export Checkbox is checked enable the input for a filename
                if self.window['final_path_image_filename'].visible:
                    self.window['final_path_image_label'].update(visible=False)
                    self.window['final_path_image_filename'].update(visible=False)
                else:
                    self.window['final_path_image_label'].update(visible=True)
                    self.window['final_path_image_filename'].update(visible=True)

            # Culture Specific Exports
            if event == "average_displacement":
                # When area graph Export Checkbox is checked enable the input for a filename
                if self.window['culture_displacement_graph_filename'].visible:
                    self.window['culture_displacement_graph_label'].update(visible=False)
                    self.window['culture_displacement_graph_filename'].update(visible=False)

                else:
                    self.window['culture_displacement_graph_label'].update(visible=True)
                    self.window['culture_displacement_graph_filename'].update(visible=True)

            if event == "average_speed":
                # When area graph Export Checkbox is checked enable the input for a filename
                if self.window['culture_speed_graph_filename'].visible:
                    self.window['culture_speed_graph_label'].update(visible=False)
                    self.window['culture_speed_graph_filename'].update(visible=False)
                else:
                    self.window['culture_speed_graph_label'].update(visible=True)
                    self.window['culture_speed_graph_filename'].update(visible=True)

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
        """Update the canvas elements within the video player interface with the next video frame recursively"""
        """Ran by Thread started by load_video"""
        start_time = time.time()
        if self.vid:
            # Only Update video while it is visible on video player interface and is supposed to play
            if self.window[f'-COL{VIDEO_PLAYER}-'].visible:
                if self.play:
                    # Retrieve the next frame from the video
                    original, edited = self.video_player.next_frame()

                    # Update the size of the video player as needed to fit the window size

                    # Take original video's height to width ratio
                    (h, w) = original.shape[:2]
                    ratio = h / w
                    # use formula: (1-%padding)/2 to figure out the max percentage of the width that can be alloted to each video without overlapping or cropping
                    video_width_percent = (1 - .03) / 2

                    # Calculate new video dimensions
                    # Set width to % of current window's width found using formula: (1-%used by padding)/2
                    self.vid_width = int(self.window.size[0] * video_width_percent)
                    # Calculate height using dimensions of the video to ensure nothing gets cropped
                    self.vid_height = int(self.vid_width * ratio)

                    # change canvas size approx to video size
                    self.canvas.config(width=self.vid_width, height=self.vid_height)
                    self.edited_canvas.config(width=self.vid_width, height=self.vid_height)

                    # next_frame() will return values of None if all frames have already been read
                    # If there are valid frames returned
                    if original is not None and edited is not None:
                        # Display next frame for unedited video
                        # convert image from BGR to RGB so that it is read correctly by PIL
                        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                        self.photo = PIL.ImageTk.PhotoImage(
                            image=PIL.Image.fromarray(original).resize((self.vid_width, self.vid_height), Image.NEAREST)
                        )
                        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                        # Display next frame for edited video
                        self.edited = PIL.ImageTk.PhotoImage(
                            image=PIL.Image.fromarray(edited).resize((self.vid_width, self.vid_height),
                                                                        Image.NEAREST)
                        )
                        self.edited_canvas.create_image(0, 0, image=self.edited, anchor=tk.NW)

                        # Update video frame counter
                        self.frame += 1
                        self.update_counter(self.frame)
                    else:
                        # Video is finished playing
                        # Stop Video playback (Set to Pause)
                        self.play = False
                        self.window.Element("Play").Update("Play")

                        # Make Export Button Clickable
                        self.window["Export Data"].update(disabled=False)

                        print("OUT OF FRAMES")

        # The tkinter .after method lets us recurse after a delay without reaching recursion limit. We need to wait
        # between each frame to achieve proper fps, but also count the time it took to generate the previous frame.
        #self.canvas.after(abs(int((self.delay - (time.time() - start_time)) * 1000)), self.update)
        self.canvas.after(abs(int(self.delay)), self.update)

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
        self.window.Element("counter").Update("{}/{}".format(frame, self.frames))

    '''
        Outputs first frame to the cell selection screen
    '''
    def display_first_frame(self):
        # Use Individual Tracker to grab and display the edited first frame
        unedited, processed = self.video_player.get_first_frame()

        # Take original video's height to width ratio
        (h, w) = unedited.shape[:2]
        ratio = h/w
        # use formula: (1-%padding)/2 to figure out the max percentage of the width that can be alloted to each video without overlapping or cropping
        video_width_percent = (1 - .03)/2

        # Calculate new video dimensions
        # Set width to % of current window's width found using formula: (1-%used by padding)/2
        self.vid_width = int(self.window.size[0] * video_width_percent)
        # Calculate height using dimensions of the video to ensure nothing gets cropped
        self.vid_height = int(self.vid_width * ratio)

        # change canvas size approx to video size
        self.first_frame_orig.config(width=self.vid_width, height=self.vid_height)
        self.first_frame_edited.config(width=self.vid_width, height=self.vid_height)

        # Update right side of counter
        self.frames = int(self.video_player.frames)
        self.window.Element("counter").Update("0/%i" % self.video_player.frames)

        # Reset frame count
        self.frame = 0

        # Display Original photo in left frame of selected view
        # scale image to fit inside the frame
        # convert image from BGR to RGB so that it is read correctly by PIL
        frame = cv2.cvtColor(unedited, cv2.COLOR_BGR2RGB)
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

    '''
    Selects the cell to track 
    if valid the id will be saved and the tracking data will be initialized based on info from the first frame, 
    otherwise it will display an error
    @:return True if the entered cell id is valid and the tracker has been successfully updated. Otherwise returns false
    '''
    def select_cell(self):
        success = False
        # Check if selected id is valid
        cell_id = self.window["cell_id"].get()

        try:
            cell_id = int(cell_id)

            if not self.video_player.is_valid_id(cell_id):
                # if invalid display error message
                sg.PopupError("Invalid Cell ID")
            else:
                # if selection is valid, set the tracker's cell id
                self.video_player.set_tracked_cell(cell_id)

                # Initialize tracker info
                self.video_player.initialize_tracker_data()
                success = True

        except ValueError or TypeError:
            # if invalid display error message
            sg.PopupError("Cell ID must be an integer")

        return success


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
            raise ValueError or TypeError("Unable to open video source", video_source)

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
    Checks if the given tracker parameters are valid. User only needs valid height/width or pixels. 
    Displays error popups if arguments given are invalid
    @param videofile The path to the video file to check
    @param width Width of the video frame in mm
    @param height Height of the video frame in mm
    @time_between_frames Time in minutes between each image in the video
    @pixels Pixels per mm
    @return True if all given parameters are valid, false if not
'''
def isValidParameters(videofile, width, height, time_between_frames, pixels, min_size, max_size, contrast, brightness, blur):
    valid = False

    # User is only required to enter valid dimensions or pixels so ensure one of them is correct
    # Video Validation
    if isValidVideo(videofile):
        # Time Validation
        if isValidTime(time_between_frames):
            # Validate Dimensions or Pixel measurement
            if isValidDimensions(width, height) or isValidPixels(pixels):
                # If the optional parameter is filled and valid, or left empty proceed
                # Validate min size
                if (isValidInt(min_size) and (min_size != "" and min_size is not None)) or min_size == "":
                    # Validate max size
                    if (isValidInt(max_size) and (max_size != "" and max_size is not None)) or max_size == "":
                        # Validate contrast
                        if (isValidFloat(contrast) and (contrast != "" and contrast is not None)) or contrast =="":
                            # Validate brightness
                            if (isValidFloat(brightness) and (brightness != "" and brightness is not None)) or brightness =="":
                                # Validate Blur
                                if (isValidInt(blur) and blur != "" and blur is not None) or blur =="":
                                    valid = True
                                # Display Blur Error Messsage
                                else:
                                    sg.popup_error("Entered: Blur intensity is invalid. Blur intensity must be a positive integer or left empty for the default value")
                            # Display brightness error message
                            else:
                                sg.popup_error(
                                    "Entered: Brightness is invalid. Brightness must be a positive integer/float or left empty for the default value")
                        # Contrast error message
                        else:
                            sg.popup_error(
                                "Entered: Contrast is invalid. Contrast must be a positive integer or left empty for the default value")
                    # Max Size error message
                    else:
                        sg.popup_error(
                            "Entered: Max size is invalid. Max size must be a positive integer or left empty for the default value")
                # Min size error message
                else:
                    sg.popup_error(
                        "Entered: min size is invalid. Min size must be a positive integer or left empty for the default value")
            # Dimensions/Pixel Measurement Error
            else:
                sg.popup_error("Entered: Dimensions or Pixels per mm is invalid. Either the width and height fields or the pixels per mm field must be filled. They must be a positive integer/float")
        # Time Error Message
        else:
            sg.popup_error(
                "Entered: time between frames is invalid. This Field must be filled with a positive integer.")
    # Video Validation
    else:
        sg.popup_error(
            "Entered: Video File is invalid. Supported File types: .mp4, .avi")



    # if isValidVideo(videofile) and isValidTime(time_between_frames) and (isValidDimensions(width, height) or isValidPixels(pixels)):
    #     valid = True

    return valid

'''
    Checks if the given video file is of correct file type and can be opened by opencv
    @param videofile The path to the video file to check
    @return True if the tracker can analyze it, false if it cannot
'''
def isValidVideo(videofile):
    valid = False
    if os.path.exists(videofile):
        if videofile.endswith(".avi") or videofile.endswith(".mp4"):
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
    except ValueError or TypeError:
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
    except ValueError or TypeError:
        valid = False

    return valid

'''
Checks if given pixel per mm value is a positive float
'''
def isValidPixels(pixels):
    valid = False
    try:
        val = float(pixels)
        if 0 < val:
            valid = True
    except ValueError or TypeError:
        valid = False

    return valid

'''
Checks if given value is a positive float
'''
def isValidFloat(var):
    valid = False
    try:
        val = float(var)
        if 0 < val:
            valid = True
    except ValueError or TypeError:
        valid = False

    return valid

'''
Checks if given value is a positive int
'''
def isValidInt(var):
    valid = False
    try:
        val = int(var)
        if 0 < val:
            valid = True
    except ValueError or TypeError:
        valid = False

    return valid

'''
Checks if given export parameters are valid
'''
def isValidExportParameters(num_labels):
    valid = True

    # If num_labels was entered, ensure that it is a positive integer
    if num_labels != '' and num_labels is not None:
        try:
            num_labels = int(num_labels)
            # if num labels is negative it is invalid
            # TODO make sure num_labels cannot be higher than number of points to be put on graph
            if num_labels < 0:
                valid = False
        except ValueError or TypeError:
            valid = False

    return valid

'''
Checks if given filename is correctly formatted
An excel file must end with.xlsx and not be blank
'''
def isValidExcelFilename(excelfile):
    valid_filename = False

    # If excel filename field is entered
    if excelfile != '' and excelfile is not None:
        # Determine if file is in the correct format
        if re.match(".*[.]xlsx$", excelfile):
            valid_filename = True

    return valid_filename

'''
Checks if given filename is correctly formatted
A graph file must end with .pdf, not be blank, and be unique as matplotlib is not authorized to save over files/delete files
'''
def isValidGraphFilename(filename):
    valid_filename = False

    # If graph filename field is entered
    if filename != '' and filename is not None:
        # Determine if file is in the correct format and that it does not already exist
        if re.match(".*[.]pdf$", filename) and not os.path.exists(filename):
            valid_filename = True

    return valid_filename

'''
Checks if given filename is correctly formatted
An Image file must end with .png, not be blank, and be unique 
'''
def isValidImageFilename(filename):
    valid_filename = False

    # If image filename field is entered
    if filename != '' and filename is not None:
        # Determine if file is in the correct format and that it does not already exist
        if re.match(".*[.]png$", filename) and not os.path.exists(filename):
            valid_filename = True

    return valid_filename

'''
Determines scale needed to adjust window to a certain screen size
@return scale - floating point
'''
def get_scaling():
    # called before window created
    root = sg.tk.Tk()
    scaling = root.winfo_fpixels('1i')/72
    root.destroy()
    return scaling


if __name__ == '__main__':
    App()
