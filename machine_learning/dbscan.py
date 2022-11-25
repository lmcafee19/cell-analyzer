'''
   @brief Machine Learning Algorithm to detect all cells from a given edited image using
    the dbscan algorithm from scikit learn
'''
import os

import matplotlib
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn import metrics
from sklearn.cluster import DBSCAN
from tracker_library import cell_analysis_functions as analysis
from tracker_library import centroid_tracker as ct

# Define Constants
IMAGE = '../videos/img_4.png'
EXPORT_FILE = "../data/csvcompare_data.xlsx"
SCALE = 0.75
CONTRAST = 1.25
BRIGHTNESS = 0.1
BLUR_INTENSITY = 10

# Define Constants for cell size. These indicate the size range in which we should detect and track cells
MIN_CELL_SIZE = 35
MAX_CELL_SIZE = 600

EPSILON = 2.2

def main():
    # Initialize Centroid tracker
    tracker = ct.CentroidTracker()

    # Take in Image and convert to numpy array
    img = cv.imread(IMAGE)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    color_correct_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Canny dbscan
    # Edit Image the same way as in the tracker
    processed_canny = analysis.process_image(img, analysis.Algorithm.CANNY, SCALE, CONTRAST, BRIGHTNESS,
                                             BLUR_INTENSITY)
    # Detect if cell is a circle or square and grab each objects centroid and area
    shapes_img, shapes = analysis.detect_shape_v2(processed_canny, MIN_CELL_SIZE, MAX_CELL_SIZE)
    #cv.imshow("Canny", processed_canny)
    # Normalize the data to all be between 0-1
    normalized_pixel_array = sklearn.preprocessing.normalize(processed_canny, norm='max')

    # Use Tracker on image to estimate number of cells in photo
    cell_locations, cell_areas = tracker.update(shapes)

    # n_clusters: Number of clusters to place observations in
    # Set this = to my tracker's estimation of the cells within the frame
    N_CLUSTERS = len(cell_locations)
    MIN = int(N_CLUSTERS * 0.75)
    MAX = int(N_CLUSTERS * 1.25)

    db = DBSCAN(eps=EPSILON, min_samples=2).fit(processed_canny)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    #print(labels)
    #visualize_clusters(labels)

    # Plot Image
    fig, ax = plt.subplots()
    ax.imshow(color_correct_img)

    # TODO For each coordinate, determine if the pixel at that location belongs to a labeled cluster, if so plot a point in the corresponding color
    # and then all clusters found ontop
    plt.scatter(processed_canny[:, 0], processed_canny[:, 1], c=db.labels_, cmap="RdGy", alpha=.5)
    plt.title(f"DBSCAN for {len(np.unique(db.labels_)) - 1} clusters")
    plt.colorbar()
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # Grab a unique label and color for each cluster found
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        # Find all elements of labels which match our given labels
        class_member_mask = labels == k
        xy = normalized_pixel_array[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = normalized_pixel_array[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()

    # eps_data = {"eps": [], "n_clusters": []}
    # for e in range(5, 50):
    #     db = DBSCAN(eps=e/10, min_samples=2).fit(normalized_pixel_array)
    #     core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #     core_samples_mask[db.core_sample_indices_] = True
    #     labels = db.labels_
    #
    #     # Number of clusters in labels, ignoring noise if present.
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #     n_noise_ = list(labels).count(-1)
    #
    #     eps_data["eps"].append(e/10)
    #     eps_data["n_clusters"].append(n_clusters_)
    #
    #     #print("Estimated number of clusters: %d" % n_clusters_)
    #     #print("Estimated number of noise points: %d" % n_noise_)
    #
    # # Plot Data Found From Different Epsilon vals
    # plt.plot(eps_data["eps"], eps_data["n_clusters"])
    # plt.xlabel("Epsilon")
    # plt.ylabel("Number of Clusters")
    # plt.title("Number of Clusters Found")
    # plt.show()


    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print(
    #     "Adjusted Mutual Information: %0.3f"
    #     % metrics.adjusted_mutual_info_score(labels_true, labels)
    # )
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


def visualize_clusters(labels):
    fig = plt.figure()
    ax = plt.axes()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())

    plt.xlabel(len(labels))

    cmap = matplotlib.colors.ListedColormap(['gray', 'white', 'blue'])
    bounds = [-1, -0.1, 0.1, 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(labels, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title("Clusters Found in Image")
    # plt.savefig('{}.png'.format("clusters"))
    # plt.close(fig)
    plt.show()

main()