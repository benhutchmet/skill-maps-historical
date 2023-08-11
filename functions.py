# Functions for testing the historical skill maps

# Imports
import argparse
import os
import sys
import glob
import re

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import scipy.stats as stats
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image

# Import CDO
from cdo import *
cdo = Cdo()

# Write a function which uses CDO to merge the time axis of historical files
# This function takes as arguments, the model name, the variable name, the initialization number, the run number
# and the path to the directory containing the files
# base_path_example: /badc/cmip6/data/CMIP6/CMIP
# /badc/cmip6/data/CMIP6/CMIP/BCC/BCC-CSM2-MR/historical/r1i1p1f1/Amon/psl/gn/files/d20181126/
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
        os.system('cp ' + dir_path + '*.nc ' + output_dir + '/')
        return
    # If there are multiple files, merge them
    else:
        # find the directories which match the path
        dirs = glob.glob(dir_path)

        # Check that the list of directories is not empty
        if len(dirs) == 0:
            print("No files available")
            return None
        
        # extract the filenames in the final directory
        # after the last /
        filenames = dirs.split('/')[-1]

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
        output_file = output_dir + '/' + output_filename

        # use a try except block to catch errors
        try:
            # Now merge the files
            # Using cdo mergetime
            cdo.mergetime(input=dir_path + '*.nc', output=output_file)

            # Print a message to say that the files have been merged
            print("Files merged successfully")

            # Return the output file
            return output_file
        except e as err:
            print("Error, failed to use cdo mergetime: ", err)
            return None
