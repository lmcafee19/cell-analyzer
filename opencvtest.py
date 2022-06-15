# Main Python Script which uses opencv to analyze footage of cells growing
# Author: zheath19@georgefox.edu
import cv2 as cv
import numpy as np
import os
from enum import Enum

# Edge Detection Algorithm Enum to store valid options
class Algorithm(Enum):
    CANNY = "CANNY"
    LAPLACIAN = "LAPLACIAN"
    SOBEL = "SOBEL"

# Define Constants for Path to file
PATH = 'videos/'
VIDEO = 'sample_cell_video.mp4'
SCALE = 0.25
CONTRAST = 4.0
BRIGHTNESS = 0
BLUR_INTENSITY = 75


def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{PATH}{VIDEO}'
    # Check if video exists
    if not os.path.exists(videoFile):
        raise Exception("File cannot be found")
    else:
        # Read in video frame by frame
        capture = cv.VideoCapture(videoFile)

        while True:
            isTrue, frame = capture.read()

            # # Process Frame to detect edges
            # processed_laplacian = process_image(frame, Algorithm.LAPLACIAN, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)
            #
            # # Display Proccessed Video
            # cv.imshow("Laplacian", processed_laplacian)

            # Edge Cascade Test: only display edges found.
            # Use Canny Edge Detection
            processed_canny = process_image(frame, Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)
            cv.imshow("Canny", processed_canny)

            # Laplacian Derivative
            processed_laplacian = process_image(frame, Algorithm.LAPLACIAN, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)
            cv.imshow("Laplacian", processed_laplacian)

            # Sobel XY
            processed_sobel = process_image(frame, Algorithm.SOBEL, SCALE, CONTRAST, BRIGHTNESS, BLUR_INTENSITY)
            cv.imshow("Sobel", processed_sobel)

            # Dilate frame. Can be used to thicken and define edges
            #dilated_frame = cv.dilate(canny_frame, (7, 7), iterations=3)
            #cv.imshow("Dilated", dilated_frame)

            # Erode. Can be used on dilated image to sharpen lines
            #eroded_frame = cv.erode(dilated_frame, (7, 7), iterations=3)
            #cv.imshow("Eroded", eroded_frame)

            # Adjust waitKey to change time each frame is displayed
            # Press q to exit out of opencv early
            if cv.waitKey(50) & 0xFF == ord('q'):
                break

        # Close opencv
        capture.release()
        cv.destroyAllWindows()


'''
    Adjusts image to better view individual cells
    This involves converting the image to grayscale, increasing contrast,
    applying filters, and applying the canny edge detector to bring out the edges of each cell
    @:param img Image to adjust
    @:param edge_alg Option from Algorithm enum
    @:param scale:   (0.0,  inf) with 1.0 leaving the scale as is
    @:param contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    @:param brightness: [-255, 255] with 0 leaving the brightness as is
    @:param blur: [0, 250] Blur Intensity
    @:return adjusted frame/image
'''
def process_image(img, edge_alg, scale:float=1.0, contrast:float=1.0, brightness:int=0, blur:int=0):
    # Scale image
    processed = rescale_frame(img, scale)
    # Convert to grayscale
    processed = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    # Increase contrast
    processed = adjust_contrast_brightness(processed, contrast, brightness)
    # Apply Bilateral Filter to Blur and reduce noise
    processed = cv.bilateralFilter(processed, 5, blur, blur)
    # Use Edge Detection Algorithm
    processed = detect_edges(processed, edge_alg)
    return processed


'''
    Resizes the given video frame or image to supplied scale
    @:param frame image or singular frame of video
    @:param scale (0.0,  inf) with 1.0 leaving the scale as is
    @:return resized frame
'''
def rescale_frame(frame, scale:float=1.0):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, cv.INTER_AREA)


'''
    Adjusts contrast and brightness of an uint8 image
    @:param contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    @:param brightness: [-255, 255] with 0 leaving the brightness as is
    @:return adjusted frame/image
'''
def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    brightness += int(round(255*(1-contrast)/2))
    return cv.addWeighted(img, contrast, img, 0, brightness)


'''
    Uses a specified edge detection algorithm to display only edges found in the image
    @:param img: image to detect edges in
    @:param edge_alg Option from Algorithm enum
    @:return edited frame/image
'''
def detect_edges(img, edge_alg):
    processed = img
    # Type checking
    if not isinstance(edge_alg, Algorithm):
        raise TypeError('Given Edge detection algorithm must be defined by Algorithm Enum')

    # Canny Edge Detection
    if edge_alg.value == "CANNY":
        processed = cv.Canny(img, 100, 150)
    # Laplacian Derivative
    elif edge_alg.value == "LAPLACIAN":
        processed = cv.Laplacian(img, cv.CV_8UC3)
    # Sobel XY
    elif edge_alg.value == "SOBEL":
        sobel = cv.Sobel(img, cv.CV_64F, 1, 1, 5)
        processed = cv.convertScaleAbs(sobel)

    return processed


main()