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
        plt.plot(df[xaxis], df[yaxis], color='blue', marker='o')
        if title:
            plt.title(title)
        plt.xlabel(xaxis, fontsize=8)
        plt.ylabel(yaxis, fontsize=8)
        plt.grid(True)

        # Save Chart to pdf and exit
        export_pdf.savefig()
        plt.close()


