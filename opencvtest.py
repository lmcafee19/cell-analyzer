# Main Python Script which uses opencv to analyze footage of cells growing
# Author: zheath19@georgefox.edu
import cv2 as cv
import os

# Define Constants for Path to file
PATH = 'videos/'
VIDEO = 'sample_cell_video.mp4'
SCALE = 0.25

def main():
    # Print opencv version
    print("Your OpenCV version is: " + cv.__version__)

    videoFile = f'{PATH}{VIDEO}'
    # Check if file exists
    if not os.path.exists(videoFile):
        raise Exception("File cannot be found")
    else:
        # Read in video frame by frame
        capture = cv.VideoCapture(videoFile)

        while True:
            isTrue, frame = capture.read()
            # scale frame down
            scaled_frame = rescaleFrame(frame, SCALE)

            # Change frame to grayscale
            # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # cv.imshow('Gray', gray_frame)

            # Blur Frame
            blurred_frame = cv.bilateralFilter(scaled_frame, 3, 75, 75)
            #cv.imshow("Blurred", blurred_frame)

            # Edge Cascade: only display edges found.
            # Number of edges can be reduced by increasing blur
            canny_frame = cv.Canny(scaled_frame, 125, 175)
            #cv.imshow('Canny Edges', canny_frame)

            # Dilate frame. Can be used to thicken and define edges
            dilated_frame = cv.dilate(canny_frame, (7, 7), iterations=3)
            #cv.imshow("Dilated", dilated_frame)

            # Erode. Can be used on dilated image to sharpen lines
            eroded_frame = cv.erode(dilated_frame, (7, 7), iterations=3)
            #cv.imshow("Eroded", eroded_frame)

            # Show edited frame
            cv.imshow('Video', scaled_frame)

            # Press q to exit out of opencv early
            if cv.waitKey(20) & 0xFF == ord('q'):
                break

        # Close opencv
        capture.release()
        cv.destroyAllWindows()
'''
This method will resize the given video frame or image to the given scale
@:param frame image or singular frame of video
@:param scale floating point number to scale image dimensions by
@:return resized frame
'''
def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

main()