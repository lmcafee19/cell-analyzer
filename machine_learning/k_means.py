'''
   @brief Machine Learning Algorithm to detect all cells from a given edited image using
    the k-means algorithm from scikit learn
'''
import pandas
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tracker_library import cell_analysis_functions as analysis
from tracker_library import centroid_tracker as ct

# Define Constants
VIDEO = '../videos/img_4.png'
IMAGE = '../videos/img_4.png'
EXPORT_FILE = "../data/csvcompare_data.xlsx"
SCALE = 0.75
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10

# Define Constants for cell size. These indicate the size range in which we should detect and track cells
MIN_CELL_SIZE = 35
MAX_CELL_SIZE = 600

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 5
VIDEO_WIDTH_MM = 5.5

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10

def main():
    # Initialize Centroid tracker
    tracker = ct.CentroidTracker()

    # Take in Image and convert to numpy array
    img = cv.imread(IMAGE)

    # Canny k_means
    # Edit Image the same way as in the tracker
    processed_canny = analysis.process_image(img, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS,
                                             BLUR_INTENSITY)
    # Detect if cell is a circle or square and grab each objects centroid and area
    shapes_img, shapes = analysis.detect_shape_v2(processed_canny, MIN_CELL_SIZE, MAX_CELL_SIZE)
    #cv.imshow("Canny", processed_canny)
    # Normalize the data to all be between 0-1
    processed_canny = sklearn.preprocessing.normalize(processed_canny, norm='max')

    canny_coords = {"x": [], "y": []}
    # Loop through the normalized array and if the value is a 1(black/edge) then record its position
    for i in range(0, len(processed_canny)):
        for j in range(0, len(processed_canny)):
            if int(processed_canny[i][j]) == 1:
                canny_coords["x"].append(i)
                canny_coords["y"].append(j)

    # convert to a dataframe
    canny_df = pandas.DataFrame(canny_coords)

    # Use Tracker on image to estimate number of cells in photo
    cell_locations, cell_areas = tracker.update(shapes)

    # n_clusters: Number of clusters to place observations in
    # Set this = to my tracker's estimation of the cells within the frame
    N_CLUSTERS = len(cell_locations)
    MIN = int(N_CLUSTERS * 0.75)
    MAX = int(N_CLUSTERS * 1.25)

    sse = k_means(canny_df, MIN, MAX)
    visualize_results(MIN, MAX, sse, "Canny")

    # Sobel k_means
    # Edit Image the same way as in the tracker
    processed_sobel = analysis.process_image(img, analysis.Algorithm.SOBEL, SCALE, CONTRAST, BRIGHTNESS,
                                             BLUR_INTENSITY)
    # Detect if cell is a circle or square and grab each objects centroid and area
    shapes_img, sobel_shapes = analysis.detect_shape_v2(processed_sobel, MIN_CELL_SIZE, MAX_CELL_SIZE)
    #cv.imshow("Sobel", processed_sobel)
    # Normalize the data to all be between 0-1
    processed_sobel = sklearn.preprocessing.normalize(processed_sobel, norm='max')

    #print(processed_sobel)
    sobel_coords = {"x": [], "y": []}
    # Loop through the normalized array and if the value is a 1(black/edge) then record its position
    for i in range(0, len(processed_sobel)):
        for j in range(0, len(processed_sobel)):
            if int(processed_sobel[i][j]) == 1:
                sobel_coords["x"].append(i)
                sobel_coords["y"].append(j)

    #print(sobel_coords)
    # convert to a dataframe
    sobel_df = pandas.DataFrame(sobel_coords)


    # Use Tracker on image to estimate number of cells in photo
    cell_locations, cell_areas = tracker.update(sobel_shapes)

    # n_clusters: Number of clusters to place observations in
    # Set this = to my tracker's estimation of the cells within the frame
    N_CLUSTERS = len(cell_locations)
    MIN = int(N_CLUSTERS * 0.75)
    MAX = int(N_CLUSTERS * 1.25)

    sobel_sse = k_means(sobel_df, MIN, MAX)
    visualize_results(MIN, MAX, sobel_sse, "Sobel")

    # Canny + Sobel k_means
    # Average Data from the canny and sobel edge detection
    sob_can_pixels = (processed_sobel + processed_canny) / 2
    #cv.imshow("Sobel+Canny", sob_can_pixels)

    sob_can_coords = {"x": [], "y": []}
    # Loop through the normalized array and if the value is a 1(black/edge) then record its position
    for i in range(0, len(sob_can_pixels)):
        for j in range(0, len(sob_can_pixels)):
            if int(sob_can_pixels[i][j]) == 1:
                sob_can_coords["x"].append(i)
                sob_can_coords["y"].append(j)

    # convert to a dataframe
    sob_can_df = pandas.DataFrame(sob_can_coords)

    sob_can_sse = k_means(sob_can_df, MIN, MAX)
    visualize_results(MIN, MAX, sob_can_sse, "Sobel+Canny")

'''
    Runs k-means algo on the given image (as a normalized np array of pixels) and then returns a list of sse
'''
def k_means(normalized_image:np.array, MIN, MAX):
    # Initialize K-means params
    # Init: controls initialization technique
    INIT = "random"
    # n_init: number of initializations to perform. Default = 10 and return the one with lowest Sum of Squared Errors
    N_INIT = 10
    # random_state: int value used as a seed to make the results more reproducible
    RANDOM_STATE = 1

    sse = []
    # Run k means within a range around the given starting clusters
    for k in range(MIN, MAX + 1):
        kmeans_results = KMeans(init=INIT, n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        # Call Kmeans alg on the image file here
        kmeans_results.fit(normalized_image)
        sse.append(kmeans_results.inertia_)

    return sse

'''
    Plots the given kmeans results
'''
def visualize_results(min, max, sse, title=IMAGE):
    plt.plot(range(min, max+1), sse)
    plt.xticks(range(min, max+1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(title)
    plt.show()


main()