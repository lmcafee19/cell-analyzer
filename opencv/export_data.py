# Helper Python File containing functions to export recorded data about cells to either an excel spreadsheet or a csv file

import csv
import xlwt
from xlwt import Workbook

'''
    Exports given cell data to an excel spreadsheet
    @param filename: name given to excel sheet to save to, extension will be .xlsx
    @param data
    @param path: path to save excel file to or open from. By default will be set to current working directory
'''
def to_excel_file(filename, data, path=""):
    # TODO Check if given file is an existing excel file
    # If file already exists and is an excel sheet create a new sheet to store data

    # Otherwise create a new excel file to write to
    # Create Workbook to store all sheets and their data
    wb = Workbook()

    # Add Sheet to store column/row data about this iteration
    sheet = wb.add_sheet('First Iteration')

    # Write Data to sheet
    # First num is row, second is col, third is data to input
    # Create Headers
    sheet.write(0, 0, "Cell ID")
    sheet.write(0, 1, "Starting Position")

    # Loop through all data given then extract useful info


    # Save File
    wb.save(f"{path}{filename}.xls")


'''
    Exports given cell data to a comma separated value file
    @param filename: Name of File to append to or save data to. Should end with extension .csv
    @param data
    @param path: path to save file to or open from. By default will be set to current working directory
'''
def to_csv_file(filename, data, path=""):
    with open(filename, "r+") as file:
        return 0

to_excel_file("Test", "Cell Stuff", "../data/")