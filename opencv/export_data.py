# Helper Python File containing functions to export recorded data about cells to either an excel spreadsheet or a csv file

import os
import datetime
import csv
import openpyxl


'''
    Exports given cell data to an excel spreadsheet
    @param filename: Name of excel file to edit or write to. Should end with extension .xls or .xlsx
    @param data
'''
def to_excel_file(filename, data):
    # TODO Check if given file is an existing excel file
    # If filename does not end in .xls extension exit
    if not (filename.endswith(".xls") or filename.endswith(".xlsx")):
        raise Exception("File must be of type .xls or .xlsx")

    # If file already exists, create a new sheet to store data on
    if os.path.exists(f"{filename}"):
        # Open excel file for reading and writing
        wb = openpyxl.load_workbook(filename)
    else:
        # Otherwise create a new excel file to write to
        # Create Workbook to store all sheets and their data
        wb = openpyxl.Workbook()

    # Add Sheet to store column/row data about this iteration TODO make dynamic/better name
    sheetname = datetime.datetime.now()
    sheet = wb.create_sheet(str(sheetname))

    # Write Data to sheet
    # Create Headers
    # Cell (row, col, data) Base 1
    sheet.cell(1, 1, "Cell ID")
    sheet.cell(1, 2, "Initial Position")
    sheet.cell(1, 3, "Area")

    # Loop through all data given then extract useful info and append it
    # Adds data to new row. Argument must be iterable object
    # sheet.append("Cell ID")

    # Save File
    wb.save(f"{filename}")


'''
    Exports given cell data to a comma separated value file
    @param filename: Name of File to append to or save data to. Should end with extension .csv
    @param data
    @param path: path to save file to or open from. By default will be set to current working directory
'''
def to_csv_file(filename, data, path=""):
    with open(filename, "r+") as file:
        return 0




#to_excel_file("../data/Test.xlsx", "Cell Stuff")