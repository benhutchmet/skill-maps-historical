#!/usr/bin/env python

"""
regrid-merge-time.py
====================

A script which takes arguments for the model, variable and region and
merges the files along their time dimension. The script then regrids according
to the gridspec file for the region specified.

Creates merged files and regridded files for the historical data for a given
model, variable and region.

Will not overwrite existing files.

Usage:
------

    regrid-merge-time.py <model> <variable> <region>
    
    model:    Model name, e.g. 'HadGEM3-GC31-MM'
    variable: Variable name, e.g. 'tas'
    region:   Region name, e.g. 'north-atlantic'
    
    e.g. regrid-merge-time.py HadGEM3-GC31-MM tas north-atlantic
    
"""

# Imports
import os
import sys
import glob
import re

# Import CDO module
from cdo import *
cdo = Cdo()

# Set up the location of the dictionaries
dict_dir = '/home/users/benhutch/skill-maps-historical/'
sys.path.append(dict_dir)

# Import the dictionaries
import dictionaries as dic

# Write the function to merge the files
def merge_time_axis(model, var, run, init, physics, forcing, base_path):
    """
    Function to merge the time axis of historical files.
    """

    # First construct the directory in which the files are stored
    dir_path = base_path + '/*/' + model + '/historical/' + 'r' + str(run) + 'i' + str(init) + 'p' + str(physics) + 'f' + str(forcing) + '/Amon/' + var + '/g?/files/d????????/'

    # Print the directory path
    print("dir_path: ", dir_path)

    # Now set up the output directory in canari
    output_dir = '/gws/nopw/j04/canari/users/benhutch/historical/' + var + '/' + model + '/mergetime'

    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If there is only one file in the directory, copy it to the output directory
    if len(glob.glob(dir_path + '*.nc')) == 1:
        # print a message to say that there is only one file
        print("Only one file available")
        
        # copy the file to the output directory
        os.system('cp ' + dir_path + '*.nc ' + output_dir + '/')

        # find the path to the file which has been copied
        copied_file = glob.glob(output_dir + '/*.nc')

        # find the full path and filename of the copied file
        copied_file = copied_file[0]

        # print a message to say that the file has been copied
        # and print the path to the copied file
        print("File copied successfully")
        print("copied_file: ", copied_file)

        return copied_file
    # If there are multiple files, merge them
    else:
        # find the directories which match the path
        dirs = glob.glob(dir_path + '*.nc')

        # Check that the list of directories is not empty
        if len(dirs) == 0:
            print("No files available")
            return None
        
        # print the list of directories
        # print("dirs: ", dirs)
        # print the length of the list of directories
        print("len(dirs): ", len(dirs))
        # print the type of the list of directories
        print("type(dirs): ", type(dirs))

        # if the length of the dirs is 1, then split the first element of the list
        if len(dirs) == 1:
            print("Only one directory found")
            filenames = dirs[0].split('/')[-1]
        else:
            print("[WARNING] More than one directory found")
            print("len(dirs): ", len(dirs))
            # loop over the directories and split the last element of each directory
            filenames = [dir.split('/')[-1] for dir in dirs]

        # Check that the list of filenames is not empty
        if len(filenames) == 0:
            print("No files available")
            return None
        
        # extract the years from the filenames
        # initialize the years list
        years = []
        for file in filenames:
            year_str = re.findall(r'\d{4}', file)
            if len(year_str) == 2:
                years.append(year_str)

        # flatten the list of year strings
        years = [year for sublist in years for year in sublist]
        
        # convert the list of strings to a list of integers
        years = list(map(int, years))

        # find the min and max years
        min_year = min(years)
        max_year = max(years)

        print("min_year: ", min_year)
        print("max_year: ", max_year)
    
        # Now construct the output file name
        output_filename = var + '_' + 'Amon' + '_' + model + '_' + 'historical' + '_' + 'r' + str(run) + 'i' + str(init) + 'p' + str(physics) + 'f' + str(forcing) + '_' + 'g?' + '_' + str(min_year) + '-' + str(max_year) + '.nc'

        # construct the output path
        output_file = os.path.join(output_dir, output_filename)

        # use a try except block to catch errors
        try:

            # if the output file already exists, don't do anything
            if os.path.exists(output_file):
                print("Output file already exists")
                return output_file
            else:
                print("Output file does not exist")
        
                # Now merge the files
                # Using cdo mergetime
                cdo.mergetime(input=dir_path + '*.nc', output=output_file)

                # Print a message to say that the files have been merged
                print("Files merged successfully")

                # Return the output file
                return output_file
        except Exception as err:
            print("Error, failed to use cdo mergetime: ", err)
            return None