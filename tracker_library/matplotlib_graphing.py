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
    @param data: Dictionary containing data about the cell
    @param xaxis Value to place on the xaxis, should also be key to data dictionary
    @param yaxis Value to place on the yaxis, should also be key to data dictionary
    @param filename Optional. Name of PDF file to save chart to. If not specified user will be prompted to edit and save graph
    @param labels Optional. Iterable container of labels for each point
    @param num_labels Optional. Number of points on the graph to label. By default only the first and last point will be labeled. 
           if set to 1 only the first point will be labeled
    @param Title Optional. Title of the chart
    @param color Name of the color to plot the points with
'''
def export_individual_cell_data(data: dict, xaxis, yaxis, filename=None, labels=None, num_labels=2, title=None, color='blue'):
    df = DataFrame(data, columns=[xaxis, yaxis])
    # If filename is specified then make graph and directly save it to pdf
    if filename:
        # If filename does not end in .pdf extension or already exists, exit
        if not filename.endswith(".pdf"):
            raise Exception("File must be of type .xls or .xlsx")
        elif os.path.exists(filename):
            raise Exception("Given File already exists")

        with PdfPages(filename) as export_pdf:
            # Create line chart and fill in information
            plt.plot(df[xaxis], df[yaxis], color=color)
            if title:
                plt.title(title)
            plt.xlabel(xaxis, fontsize=8)
            plt.ylabel(yaxis, fontsize=8)
            plt.grid(True)

            # Add Markers and labels to specified number of points
            if labels:
                # Label first point only
                if num_labels == 1:
                    plt.text(data[xaxis][0], data[yaxis][0], labels[0])
                    plt.plot(data[xaxis][0], data[yaxis][0], color=color, marker='o')

                # Label first and last, then the remaining amount evenly distributed along the line
                elif num_labels >= 2:
                    plt.text(data[xaxis][0], data[yaxis][0], labels[0])
                    plt.plot(data[xaxis][0], data[yaxis][0], color=color, marker='o')
                    plt.text(data[xaxis][len(xaxis) - 1], data[yaxis][len(yaxis) - 1], labels[len(labels) - 1])
                    plt.plot(data[xaxis][len(xaxis) - 1], data[yaxis][len(yaxis) - 1], color=color, marker='o')

                    # Subtract 2 from num_labels since we have already labeled the first and last
                    num_labels -= 2

                    # Label the rest of the points evenly
                    if num_labels > 0:
                        # Calculate the number of positions to step by when we label
                        # Subtract 2 from len since we have already added the first and last point
                        step = (len(data[xaxis]) - 2) // num_labels

                        for i in range(0, len(labels), step):
                            plt.text(data[xaxis][i], data[yaxis][i], labels[i])
                            plt.plot(data[xaxis][i], data[yaxis][i], color=color, marker='o')

            # Save Chart to pdf and exit
            export_pdf.savefig()
            plt.close()
    else:
        # Create line chart and fill in information
        plt.plot(df[xaxis], df[yaxis], color=color)
        if title:
            plt.title(title)
        plt.xlabel(xaxis, fontsize=8)
        plt.ylabel(yaxis, fontsize=8)
        plt.grid(True)

        # Add Markers and labels to specified number of points
        if labels:
            # Label first point only
            if num_labels == 1:
                plt.text(data[xaxis][0], data[yaxis][0], labels[0])
                plt.plot(data[xaxis][0], data[yaxis][0], color=color, marker='o')

            # Label first and last, then the remaining amount evenly distributed along the line
            elif num_labels >= 2:
                plt.text(data[xaxis][0], data[yaxis][0], labels[0])
                plt.plot(data[xaxis][0], data[yaxis][0], color=color, marker='o')
                plt.text(data[xaxis][len(xaxis) - 1], data[yaxis][len(yaxis) - 1], labels[len(labels) - 1])
                plt.plot(data[xaxis][len(xaxis) - 1], data[yaxis][len(yaxis) - 1], color=color, marker='o')

                # Subtract 2 from num_labels since we have already labeled the first and last
                num_labels -= 2

                # Label the rest of the points evenly
                if num_labels > 0:
                    # Calculate the number of positions to step by when we label
                    # Subtract 2 from len since we have already added the first and last point
                    step = (len(data[xaxis]) - 2) // num_labels

                    for i in range(0, len(labels), step):
                        plt.text(data[xaxis][i], data[yaxis][i], labels[i])
                        plt.plot(data[xaxis][i], data[yaxis][i], color=color, marker='o')

        # Display chart and prompt user to edit and save
        plt.show()


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

