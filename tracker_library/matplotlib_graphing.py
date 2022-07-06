# Library containing functions to visualize data collected by the cell tracker

import os
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

'''
    Creates a line chart with given criteria and exports it to a pdf
    @param filename: Name of PDF file to save chart to
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param Title Optional Title of the chart
'''
def export_individual_cell_area(filename, data: dict, xaxis, yaxis, title=None):
    # If filename does not end in .pdf extension or already exists, exit
    if not filename.endswith(".pdf"):
        raise Exception("File must be of type .xls or .xlsx")
    elif os.path.exists(filename):
        raise Exception("Given File already exists")

    df = DataFrame(data, columns=[xaxis, yaxis])

    with PdfPages(filename) as export_pdf:
        # Create line chart and fill in information
        plt.plot(df[xaxis], df[yaxis], color='blue')
        if title:
            plt.title(title)
        plt.xlabel(xaxis, fontsize=8)
        plt.ylabel(yaxis, fontsize=8)
        plt.grid(True)

        # Save Chart to pdf and exit
        export_pdf.savefig()
        plt.close()


'''
    Creates a line chart visualizing selected data of an individual cell
    @param filename: Name of PDF file to save chart to
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param labels Optional. Iterable container of labels for each point
    @param Title Optional. Title of the chart
'''
def export_individual_cell_data(filename, data: dict, xaxis, yaxis, labels=None, title=None):
    # If filename does not end in .pdf extension or already exists, exit
    if not filename.endswith(".pdf"):
        raise Exception("File must be of type .xls or .xlsx")
    elif os.path.exists(filename):
        raise Exception("Given File already exists")

    df = DataFrame(data, columns=[xaxis, yaxis])

    with PdfPages(filename) as export_pdf:
        # Create line chart and fill in information
        plt.plot(df[xaxis], df[yaxis], color='blue', marker='o')
        if title:
            plt.title(title)
        plt.xlabel(xaxis, fontsize=8)
        plt.ylabel(yaxis, fontsize=8)
        plt.grid(True)

        # Add labels to each point on graph
        if labels:
            for i in range(0, len(labels)):
                plt.text(data[xaxis][i], data[yaxis][i], labels[i])

        # Save Chart to pdf and exit
        export_pdf.savefig()
        plt.close()


'''
    Creates a simplified line chart visualizing a selected number of points from the data of an individual cell
    @param filename: Name of PDF file to save chart to
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param numpoints Number of data points to display on the finished graph
    @param labels Optional. Iterable container of labels for each point
    @param Title Optional. Title of the chart
'''
def export_simplified_individual_cell_data(filename, data: dict, xaxis, yaxis, num_points=10, labels=None, title=None):
    # If labels were given call correct version of method
    if labels:
        simple_data = simplify_data(data, xaxis, yaxis, num_points)
        simple_labels = simplify_labels(labels, num_points)
        export_individual_cell_data(filename, simple_data, xaxis, yaxis, simple_labels, title)
    else:
        simple_data = simplify_data(data, xaxis, yaxis, num_points)
        export_individual_cell_data(filename, simple_data, xaxis, yaxis, labels, title)

'''
    Simplifies data into num points evenly distributed among the data set inorder to make it more readable
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param num_points Number of data points to reduce data set to
    @return Simplified data dict
'''
def simplify_data(data: dict, xaxis, yaxis, num_points=10):
    # Ensure given data has at least num_points
    if len(data[xaxis]) < num_points or len(data[yaxis]) < num_points:
        raise Exception(f"Given Data has less than {num_points} data entries")
    elif len(data[xaxis]) != len(data[yaxis]):
        raise Exception(f"Given Data must have equal entries on the x and y axis")

    # Add First point to simplified data set
    simplified = {xaxis: [], yaxis: []}
    simplified_labels = []

    # Calculate how number of positions to skip between points we grab
    # Subtract 2 from num_points and len since we will always add the first and last point
    step = (len(data[xaxis]) - 2) // (num_points - 2)

    for i in range(0, (len(data[xaxis])), step):
        simplified[xaxis].append(data[xaxis][i])
        simplified[yaxis].append(data[yaxis][i])

    # Add Last point to simplified data set
    simplified[xaxis].append(data[xaxis][len(data[xaxis]) - 1])
    simplified[yaxis].append(data[yaxis][len(data[yaxis]) - 1])

    return simplified


'''
    Simplifies data into num points evenly distributed among the data set inorder to make it more readable
    @param labels Labels corresponding to each data point
    @param num_points Number of data points to reduce data set to
    @returns List of simplified labels
'''
def simplify_labels(labels, num_points=10):
    # Ensure given data has at least num_points
    if len(labels) < num_points:
        raise Exception(f"Given Data has less than {num_points} data entries")

    simplified_labels = []

    # Calculate how number of positions to skip between points we grab
    # Subtract 2 from num_points and len since we will always add the first and last point
    step = (len(labels) - 2) // (num_points - 2)

    # Update labels
    for i in range(0, (len(labels)), step):
        simplified_labels.append(labels[i])

    # Add Last point to simplified data set
    simplified_labels.append(labels[len(labels) - 1])

    return simplified_labels














