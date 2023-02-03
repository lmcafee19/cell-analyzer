'''
   @brief Machine Learning Algorithm to detect all cells from a given edited image using
    the gaussian mixture algorithm from scikit learn
'''
import os.path
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
IMAGE = '../videos/spheroid_img.png'

SCALE = 0.75
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10

# Define Constants for cell size. These indicate the size range in which we should detect and track cells
MIN_CELL_SIZE = 25
MAX_CELL_SIZE = 1500

# Real World size of frame in mm
VIDEO_HEIGHT_MM = 5
VIDEO_WIDTH_MM = 5.5

# Minutes Passed between each frame in video
TIME_BETWEEN_FRAMES = 10

def main():
    if not os.path.exists(IMAGE):
        exit("Image does not exist")

    # Initialize Centroid tracker
    tracker = ct.CentroidTracker()

    # Take in Image and convert to numpy array
    img = np.asarray(cv.imread(IMAGE))

    # Scale img
    img = analysis.rescale_frame(img, SCALE)

    # Edit Image the same way as in the tracker
    processed_canny = analysis.process_image(img, analysis.Algorithm.CANNY, 1, CONTRAST, BRIGHTNESS,
                                             BLUR_INTENSITY)

    # Detect if cell is a circle or square and grab each objects centroid and area
    shapes_img, shapes = analysis.detect_shape_v2(processed_canny, MIN_CELL_SIZE, MAX_CELL_SIZE)
    #cv.imshow("Canny", processed_canny)

    # Normalize the data to all be between 0-1
    processed_canny = sklearn.preprocessing.normalize(processed_canny, norm='max')

    # convert to a dataframe
    #canny_df = x_y_dataframe(processed_canny)
    canny_df = x_y_color_dataframe(processed_canny, img)

    # Use Tracker on image to estimate number of cells in photo
    cell_locations, cell_areas = tracker.update(shapes)

    # n_clusters: Number of clusters to place observations in
    # Set this = to my tracker's estimation of the cells within the frame
    N_CLUSTERS = len(cell_locations)
    MIN = int(N_CLUSTERS * 0.70)
    MAX = int(N_CLUSTERS * 1.6)
    NUM_PIXELS = len(canny_df["x"])

    # List of Confidences for each number of clusters tried
    confidences = batch_gaussian_mixture(canny_df, MIN, MAX)

    # Plot Confidences against number of clusters ran
    visualize_confidence(MIN, MAX, confidences, "Canny Gaussian Confidence Levels")

    #print(canny_df.keys())
    # Plot clusters on graph to see how they line up
    #visualize_clusters(N_CLUSTERS, canny_df)


'''
    Runs gaussian mixture algo on the given image data (as a dataframe of white pixels) and then returns a list of confidences for each
'''
def batch_gaussian_mixture(df:pandas.DataFrame, MIN, MAX):
    confidence_list = []
    for n in range(MIN, MAX + 1):
        # Run Gaussian Mixture with
        gm = GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(df)
        # Confidence represents the percent likelihood the pixel belongs to a detected cluster
        confidence = gm.predict_proba(df)
        # Find labels (which cluster each pixel belongs to) to our dataframe
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
    Creates Pandas Dataframe to be used as labels for gaussian mixture alg
    Use the function for images where clustering is solely based on position
    img: matrix representing image after canny edge detection and other processing 
'''
def x_y_dataframe(img):
    canny_coords = {"x": [], "y": []}
    # Loop through the normalized array and if the value is a 1(black/edge) then record its position
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if int(img[i][j]) == 1:
                canny_coords["x"].append(i)
                canny_coords["y"].append(j)

    # convert to a dataframe
    return pandas.DataFrame(canny_coords)

'''
    Creates Pandas Dataframe to be used as labels for gaussian mixture alg
    Use the function for spheroid images where different cell types are given a specific color
    img: matrix representing image after canny edge detection and other processing 
'''
def x_y_color_dataframe(img, color_img):
    # Generate Labels for each pixel based on the color version of the image
    color_labels = generate_color_labels(color_img)

    canny_coords = {"x": [], "y": [], "color": []}
    # Loop through the normalized array and if the value is a 1(black/edge) then record its position and original color
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if int(img[i][j]) == 1:
                canny_coords["x"].append(i)
                canny_coords["y"].append(j)
                canny_coords["color"].append(color_labels[i][j])

    # convert to a dataframe
    return pandas.DataFrame(canny_coords)


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

'''
    Categorizes the primary color of each pixel within the given np ndarray
    Img: np matrix containing BGR data on each pixel
    @retun matrix the same shape as img. Labels are as follows: 0 = black, 1 = white, 2 = blue, 3 = green, 4 = red
'''
def generate_color_labels(img):
    color_labels = []
    i = 0
    for row in img:
        color_labels.append([])
        for col in row:
            if col[0] == 0 and col[0] == col[1] and col[0] == col[2]:
                label = 0
            elif col[0] == 255 and col[0] == col[1] and col[0] == col[2]:
                label = 1
            elif col[0] > col[1] and col[0] > col[2]:
                label = 2
            elif col[1] > col[0] and col[1] > col[2]:
                label = 3
            else:
                label = 4
            color_labels[i].append(label)
        i += 1

    return color_labels


main()