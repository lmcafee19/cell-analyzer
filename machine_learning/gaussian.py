'''
   @brief Machine Learning Algorithm to detect all cells from a given edited image using
    the gaussian mxiture algorithm from scikit learn
'''
import random
import pandas
import numpy as np
import seaborn as sns
import cv2 as cv
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.mixture import GaussianMixture
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
    NUM_PIXELS = len(canny_df["x"])

    # List of Confidences for each number of clusters tried
    confidences = batch_gaussian_mixture(canny_df, MIN, MAX)

    # Plot Confidences against number of clusters ran
    visualize_confidence(MIN, MAX, confidences, "Canny Gaussian Confidence Levels")


    # Run Gaussian Mixture with
    # gm = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full", random_state=0).fit(canny_df)
    # # Means represents the center coordinates of each cluster found
    # means = gm.means_
    # # Add labels (which cluster each pixel belongs to) to our dataframe
    # labels = gm.predict(canny_df)
    # confidence = gm.predict_proba(canny_df)
    # canny_df["predicted_clusters"] = labels
    #
    # # Average the confidence for each cluster
    # conf_avg = 0
    # # For each pixel, access its confidence for being placed into the cluster it was placed in
    # for pixel in range(NUM_PIXELS):
    #     assigned_cluster = canny_df["predicted_clusters"][pixel]
    #     conf_avg += confidence[pixel][assigned_cluster]
    #
    # conf_avg = conf_avg/NUM_PIXELS
    # print(conf_avg)
    # for pixel in range(len(confidence)):
    #     avg = 0
    #     for cluster in range(N_CLUSTERS):
    #         if confidence[pixel][cluster] != 0:
    #             avg += confidence[pixel][cluster]
    #     conf_avg.append(avg/N_CLUSTERS)


    # Average Confidence for this number of clusters
    #avg_confidence = np.mean(confidence)

    #print(f"Average Confidence: {np.mean(canny_df['confidence'])}")

    # TODO Find RMSE (Root Means Squared Error)
    #sklearn.metrics.mean_squared_error()



    #gm.predict([[0, 0], [12, 3]])
    # sse = k_means(canny_df, MIN, MAX)
    # visualize_results(MIN, MAX, sse, "Canny")
    #
    # # Sobel k_means
    # # Edit Image the same way as in the tracker
    # processed_sobel = analysis.process_image(img, analysis.Algorithm.SOBEL, SCALE, CONTRAST, BRIGHTNESS,
    #                                          BLUR_INTENSITY)
    # # Detect if cell is a circle or square and grab each objects centroid and area
    # shapes_img, sobel_shapes = analysis.detect_shape_v2(processed_sobel, MIN_CELL_SIZE, MAX_CELL_SIZE)
    # #cv.imshow("Sobel", processed_sobel)
    # # Normalize the data to all be between 0-1
    # processed_sobel = sklearn.preprocessing.normalize(processed_sobel, norm='max')
    #
    # #print(processed_sobel)
    # sobel_coords = {"x": [], "y": []}
    # # Loop through the normalized array and if the value is a 1(black/edge) then record its position
    # for i in range(0, len(processed_sobel)):
    #     for j in range(0, len(processed_sobel)):
    #         if int(processed_sobel[i][j]) == 1:
    #             sobel_coords["x"].append(i)
    #             sobel_coords["y"].append(j)
    #
    # #print(sobel_coords)
    # # convert to a dataframe
    # sobel_df = pandas.DataFrame(sobel_coords)
    #
    # # Use Tracker on image to estimate number of cells in photo
    # cell_locations, cell_areas = tracker.update(sobel_shapes)
    #
    # # n_clusters: Number of clusters to place observations in
    # # Set this = to my tracker's estimation of the cells within the frame
    # N_CLUSTERS = len(cell_locations)
    # MIN = int(N_CLUSTERS * 0.75)
    # MAX = int(N_CLUSTERS * 1.25)
    #
    # sobel_sse = k_means(sobel_df, MIN, MAX)
    # visualize_results(MIN, MAX, sobel_sse, "Sobel")

'''
    Runs gaussian mixture algo on the given image data (as a dataframe of white pixels) and then returns a list of confidences for each
'''
def batch_gaussian_mixture(df:pandas.DataFrame, MIN, MAX):
    confidence_list = []
    for n in range(MIN, MAX + 1):
        # Run Gaussian Mixture with
        gm = GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(df)
        # Confidence represents the percent liklihood the pixel belongs to a detected cluster
        confidence = gm.predict_proba(df)
        # Add labels (which cluster each pixel belongs to) to our dataframe
        labels = gm.predict(df)

        # Average the confidence for each cluster
        conf_avg = 0
        # For each pixel, access its confidence for being placed into the cluster it was placed in
        for pixel in range(len(df["x"])):
            assigned_cluster = labels[pixel]
            conf_avg += confidence[pixel][assigned_cluster]

        conf_avg = conf_avg / len(df["x"])
        confidence_list.append(conf_avg)

    return confidence_list

'''
    Plots the given confidence of each number of clusters for the gaussian mixture algo
'''
def visualize_confidence(min, max, confidences, title=IMAGE):
    plt.plot(range(min, max+1), confidences)
    plt.xticks(range(min, max+1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Confidence")
    plt.title(title)
    plt.show()

'''
    Creates a scatter plot visualizing the location of each detected cluster 
    @param n_cluster Number of clusters predicted
    @param dataframe: pandas data frame containing pixel data on x, y, and predicted_clusters to indicate which cluster weach pixel is part of
    @param filename: optional parameter to save figure to file
'''
def visualize_clusters(n_clusters, dataframe:pandas.DataFrame, filename=None):
    color_palette = generate_random_colors(n_clusters)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=dataframe,
                    x="x",
                    y="y",
                    hue="predicted_clusters",
                    palette=color_palette)
    if filename:
        plt.savefig(filename,
                format='png', dpi=150)
    else:
        plt.show()

'''
    Generates no_of_colors number of hex codes for random colors
'''
def generate_random_colors(no_of_colors:int):
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(no_of_colors)]
    return colors


main()