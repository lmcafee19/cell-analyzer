pyinstaller -F --onedir cell-analyzer.py ../tracker_library/TrackerClasses.py ../tracker_library/cell_analysis_functions.py ../tracker_library/centroid_tracker.py ../tracker_library/export_data.py ../tracker_library/matplotlib_graphing.py
Copy-Item "bruin.png" -Destination "./dist/cell-analyzer"
Copy-Item -Path "../tracker_library" -Destination "./dist/cell-analyzer" -Recurse