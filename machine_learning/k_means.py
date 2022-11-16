'''
   @brief Machine Learning Algorithm to detect all cells from a given edited image using
    the k-means algorithm from scikit learn
'''

import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tracker_library import cell_analysis_functions as analysis

def main():
    # Initialize K-means params
    # Init: controls initialization technique
    INIT = "random"
    # n_clusters: Number of clusters to place observations in
    # TODO set this = to my trackers estimation of the cells within the frame
    N_CLUSTERS = 8
    MIN = N_CLUSTERS * 0.8
    MAX = N_CLUSTERS * 1.2
    # n_init: number of initializations to perform. Default = 10 and return the one with lowest Sum of Squared Errors
    N_INIT = 10
    # random_state: int value used as a seed to make the results more reproducible
    RANDOM_STATE = 1

    # Take in Image and convert to numpy array
    img = cv.imread('../videos/img_4.png')
    # Edit Image the same way as in the tracker
    processed_canny = analysis.process_image(frame, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS,
                                             BLUR_INTENSITY)

    # Detect if cell is a circle or square and grab each objects centroid and area
    shapes_img, shapes = analysis.detect_shape_v2(processed_canny, MIN_CELL_SIZE, MAX_CELL_SIZE)

    pixel_array = np.reshape(img)
    # Normalize the data to all be between 0-1
    scaled_pixel_array = pixel_array / 255


    # Define Data Frame
    scaled_df = 0

    sse = []
    # Run k means within a range around the given starting clusters
    for k in range(MIN, MAX+1):
        kmeans_results = KMeans(init=INIT, n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        # Call Kmeans alg on the image file here
        kmeans_results.fit(scaled_pixel_array)
        sse.append(kmeans_results.inertia_)


'''
    Plots the given kmeans results
'''
def visualize_results(min, max, sse):
    plt.plot(range(min, max), sse)
    plt.xticks(range(min, max))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()


main()