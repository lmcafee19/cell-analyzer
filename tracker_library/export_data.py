# Helper Python File containing functions to export recorded data about cells to either an excel spreadsheet or a csv file

import os
import csv
import openpyxl
import math



'''
    Exports given cell data to an excel spreadsheet
    @param filename: Name of excel file to edit or write to. Should end with extension .xls or .xlsx
    @param data: Dictionary indexed by cell id, and containing data about each cell
    @param headers: iterable object containing headers for each column
'''
def to_excel_file(filename, data, headers=None, sheetname=None):
    if headers is None:
        headers = []

    current_row = 1
    current_col = 1

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

    # Add Sheet to store column/row data about this iteration
    sheet = wb.create_sheet(sheetname)

    # Write Data to sheet
    # Create Headers
    for i in range(0, len(headers)):
        # Cell (row, col, data) Base 1
        sheet.cell(current_row, current_col + i, headers[i])
    current_row += 1

    # Loop through all data given then extract useful info and append it
    # Adds data to new row. Argument must be iterable object
    for key, value in data.items():
        current_col = 1
        sheet.cell(current_row, current_col, key)
        current_col += 1
        for val in value:
            sheet.cell(current_row, current_col, str(val))
            current_col += 1
        current_row += 1

        #sheet.append(row)

    # Save File
    wb.save(f"{filename}")


'''
    Exports given dictionary containing data on an individuals cell's area and coordinates to excel file
    @param filename: Name of excel file to edit or write to. Should end with extension .xls or .xlsx
    @param data: Dictionary containing data about the cell
    @param headers: iterable object containing headers for each column
'''
def individual_to_excel_file(filename, data:dict, time_between_frames, sheetname=None):
    current_row = 1
    current_col = 1

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

    # Add Sheet to store column/row data about this iteration
    sheet = wb.create_sheet(sheetname)

    # Write Data to sheet
    # Loop through dictionary. For every key write all of its values within the same column then move onto the next
    for key, value in data.items():
        current_row = 1
        # Add Header
        sheet.cell(current_row, current_col, key)
        current_row += 1

        # Add Data
        for entry in value:
            sheet.cell(current_row, current_col, str(entry))
            current_row += 1

        current_col += 1

    # Generate Statistics using positional data about cells and export that to same excel sheet
    coordinates = merge(data["X Position (mm)"], data["Y Position (mm)"])
    stats = calc_individual_cell_statistics(coordinates, time_between_frames)

    # Loop Through Stats and add them to excel sheet
    for key, value in stats.items():
        # Add Header
        current_row = 1
        sheet.cell(current_row, current_col, key)
        current_row += 1

        # Add Data
        sheet.cell(current_row, current_col, value)
        current_col += 1

    # Save File
    wb.save(f"{filename}")


'''
    Exports given cell coordinates to an excel spreadsheet
    @param filename: Name of excel file to edit or write to. Should end with extension .xls or .xlsx
    @param data: Dictionary indexed by cell id, and containing data about each cell
    @param headers: iterable object containing headers for each column
'''
def coordinates_to_excel_file(filename, data, headers=None, sheetname=None):
    if headers is None:
        headers = []

    current_row = 1
    current_col = 1

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

    # Add Sheet to store column/row data about this iteration
    sheet = wb.create_sheet(sheetname)

    # Write Data to sheet
    # Create Headers
    for i in range(0, len(headers)):
        # Cell (row, col, data) Base 1
        sheet.cell(current_row, current_col + i, headers[i])
    current_row += 1

    # Loop through all data given then extract useful info and append it
    # Adds data to new row. Argument must be iterable object
    for key, value in data.items():
        current_col = 1
        sheet.cell(current_row, current_col, key)
        current_col += 1
        for val in value:
            # Remove [] from string to format it correctly for Excel
            val = str(val).replace("[", "")
            val = str(val).replace("]", "")
            sheet.cell(current_row, current_col, val)
            current_col += 1

        # Insert Excel Formula to display Euclidean Distance between the cell's initial and final position
        distance_formula = f'=SQRT((((LEFT(INDIRECT(ADDRESS({current_row}, {current_col - 1})),FIND(",",INDIRECT(ADDRESS({current_row}, {current_col - 1})))-1))-(LEFT(INDIRECT(ADDRESS({current_row}, 2)),FIND(",",INDIRECT(ADDRESS({current_row}, 2)))-1)))^2) + (((RIGHT(INDIRECT(ADDRESS({current_row}, {current_col - 1})),LEN(INDIRECT(ADDRESS({current_row}, {current_col - 1})))-FIND(",",INDIRECT(ADDRESS({current_row}, {current_col - 1})))-1)) - (RIGHT(INDIRECT(ADDRESS({current_row}, 2)),LEN(INDIRECT(ADDRESS({current_row}, 2)))-FIND(",",INDIRECT(ADDRESS({current_row}, 2)))-1)))^2))'
        sheet.cell(current_row, current_col, distance_formula)
        current_col += 1
        current_row += 1

    # Save File
    wb.save(f"{filename}")


'''
    Exports given cell areas to an excel spreadsheet
    @param filename: Name of excel file to edit or write to. Should end with extension .xls or .xlsx
    @param data: Dictionary indexed by cell id, and containing data about each cell
    @param headers: iterable object containing headers for each column
'''
def area_to_excel_file(filename, data, headers=None, sheetname=None):
    if headers is None:
        headers = []

    current_row = 1
    current_col = 1

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

    # Add Sheet to store column/row data about this iteration
    sheet = wb.create_sheet(sheetname)

    # Write Data to sheet
    # Create Headers
    for i in range(0, len(headers)):
        # Cell (row, col, data) Base 1
        sheet.cell(current_row, current_col + i, headers[i])
    current_row += 1

    # Loop through all data given then extract useful info and append it
    # Adds data to new row. Argument must be iterable object
    for key, value in data.items():
        current_col = 1
        sheet.cell(current_row, current_col, key)
        current_col += 1
        for val in value:
            # Remove [] from string to format it correctly for Excel
            val = str(val).replace("[", "")
            val = str(val).replace("]", "")
            sheet.cell(current_row, current_col, val)
            current_col += 1

        # Insert Excel Formula to display Total Growth the cell Underwent
        growth_formula = f'=INDIRECT(ADDRESS({current_row}, {current_col - 1})) - INDIRECT(ADDRESS({current_row}, 2))'
        sheet.cell(current_row, current_col, growth_formula)
        current_col += 1
        # Insert Excel Formula to display largest amount of change between two time intervals
        change_formula = f'=_xlfn.AGGREGATE(14, 6, INDIRECT(ADDRESS({current_row}, 2)):INDIRECT(ADDRESS({current_row}, {current_col - 2}))-INDIRECT(ADDRESS({current_row}, 3)):INDIRECT(ADDRESS({current_row}, {current_col - 1})), 1)'
        sheet.cell(current_row, current_col, change_formula)
        current_col += 1
        current_row += 1

    # Save File
    wb.save(f"{filename}")


'''
    Exports given cell data to a comma separated value file
    @param filename: Name of File to append to or save data to. Should end with extension .csv and contain full path as necessary
    @param data: 2d iterable object containing data about each cell
    @param headers: iterable object containing headers for each column
'''
def to_csv_file(filename, data, headers=None):
    if headers is None:
        headers = []
    # If filename does not end in .xls extension exit
    if not filename.endswith(".csv"):
        raise Exception("File must be of type .csv")

    # If file already exists, append data to the end
    if os.path.exists(f"{filename}"):
        # Open csv file in append mode
        with open(filename, "a") as file:
            csvwriter = csv.writer(file)

            # Loop through data and write each row
            for row in data:
                csvwriter.writerow(row)

    # Otherwise create a new csv file to write to
    else:
        with open(filename, "w") as file:
            csvwriter = csv.writer(file)

            # Write headers
            csvwriter.writerow(headers)

            # Loop through data and write each row
            for row in data:
                csvwriter.writerow(row)


'''
    Calculates Statistics from positional data from one cell. These Stats include:
    Total Displacement: Total distance moved, Final Distance from origin, maximum distance from origin, average distance from origin,
    Maximum speed, average speed, average angle of direction, and final angle of direction between final point and origin
    @param data List of tuples/list containing x/y coordinates of given cells location
    @param time_between_frames Time in minutes between each frame of cell growth video
    @return Dictionary containing statistics generated 
'''
def calc_individual_cell_statistics(data, time_between_frames):
    # Create dictionary to hold all calculated statistics
    stats = {}
    distances = []
    # In mm / min
    speeds = []
    # Angle in degrees between last and current point
    angle_of_direction = []
    final_angle = 0

    if data is not None:
        # Grab origin point
        origin_x = data[0][0]
        origin_y = data[0][1]
        x = 0
        y = 0

        # Loop through all positions and calculate stats between points
        for i in range(1, len(data)):
            # Grab x and y coordinates
            x = data[i][0]
            y = data[i][1]
            prevx = data[i-1][0]
            prevy = data[i-1][1]

            # Calc distance from origin
            distance = math.dist([origin_x, origin_y], [x, y])
            distances.append(distance)
            # calc current speed
            speeds.append(distance/time_between_frames)
            # calc angle of direction from last point
            angle = math.atan2(y - prevy, x - prevx) * (180 / math.pi)
            angle_of_direction.append(angle)

            # If on final coordinate calculate angle between this and origin point
            if i == (len(data) - 1):
                final_angle = math.atan2(y - origin_y, x - origin_x) * (180 / math.pi)

        # Total Displacement (Total Distance Traveled)
        stats["Total Displacement (mm)"] = sum(distances)
        # Final Distace from Origin
        stats["Final Distance from Origin (mm)"] = math.dist([origin_x, origin_y], [x, y])
        # Maximum Distance from origin
        stats["Maximum Distance from Origin (mm)"] = max(distances)
        # Average Distance from origin
        stats["Average Distance from Origin (mm)"] = sum(distances)/len(distances)
        # Max Speed (distance/time)
        stats["Maximum Speed (mm/min)"] = max(speeds)
        # Average Speed
        stats["Average Speed (mm/min)"] = sum(speeds)/len(speeds)
        # Average Angle of direction from origin in degrees
        stats["Average Angle of Direction from Origin (degrees)"] = sum(angle_of_direction)/len(angle_of_direction)
        # TODO determine direction of movement
        # Angle of direction from origin to final point
        stats["Angle of Direction between Origin and Final Point (degrees)"] = final_angle
    else:
        raise Exception("Empy Data Set Given")

    return stats


'''
    Combines two lists together into one list containing tuples of (list1[i], list2[i])
    @param list1 List of elements the same length as list2
    @param list2 List of elements the same length as list1
    @return Merged List of tuples
'''
def merge(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list


# stats = calc_individual_cell_statistics([[1, 2], [3, 4], [6, 7], [8, 9]], 5)
# print(stats)