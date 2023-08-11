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

# Import the dictionaries
import dictionaries as dic

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
        except e as err:
            print("Error, failed to use cdo mergetime: ", err)
            return None


# Define a function which will regrid the data according to the parameters of a gridspec file
# This function will takes as arguments: the model, the variable, the run, the initialization number, the physics number, 
# the forcing number, the merged file path and the region
def regrid(model, var, run, init, physics, forcing, region):
    """
    Function to regrid the data according to the parameters of a gridspec file.
    """
    
    # set up the gridspec path
    gridspec_path  = dic.gridspec_path

    # Now set up the gridspec file to be used based on the region provided
    gridspec_file = gridspec_path + '/' + 'gridspec' + '-' + region + '.txt'

    # Check that the gridspec file exists
    if not os.path.exists(gridspec_file):
        print("Error, gridspec file does not exist for region: ", region)
        return None
    
    # Now set up the output directory in canari
    output_dir = dic.canari_base_path_historical + '/' + var + '/' + model + '/regrid'

    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Now find the merged file which matches the r, i, p, f specifications
    # construct the directory first
    merged_dir = dic.canari_base_path_historical + '/' + var + '/' + model + '/mergetime'

    # Now construct the merged file name
    merged_filename = var + '_' + 'Amon' + '_' + model + '_' + 'historical' + '_' + 'r' + str(run) + 'i' + str(init) + 'p' + str(physics) + 'f' + str(forcing) + '_' + 'g?' + '_*.nc'

    # Now construct the merged file path
    merged_file = merged_dir + '/' + merged_filename

    # Check that the merged file exists
    if not os.path.exists(merged_file):
        print("Error, merged file does not exist: ", merged_file)
        return None
    
    # Now construct the output file name
    # from the base name of the merged file
    output_filename = merged_filename.split('.')[0] + '_' + region + '_regrid.nc'
    # Now construct the output file path
    output_file = output_dir + '/' + output_filename

    # if the output file already exists, don't do anything
    if os.path.exists(output_file):
        print("Output file already exists")
        return output_file
    else:
        print("Output file does not exist")
        
        try:
            # Now regrid the file
            # Using cdo remapbil
            cdo.remapbil(gridspec_file, input=merged_file, output=output_file)

            # Print a message to say that the files have been regridded
            print("Files regridded successfully")

            # Return the output file
            return output_file
        except e as err:
            print("Error, failed to use cdo remapbil: ", err)
            return None
            
# Now we want to write a function which will call the mergetime and regrid functions
# for each given model, run, init, physics and forcing combination
# This function will take as arguments: the dictionary of models, the variable name, the region name
def call_mergetime_regrid(model_dict, var, region):
    """
    Loops over the models, runs, inits, physics and forcings defined in the model_dict and calls the mergetime and regrid functions.
    """

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in model_dict:
        # Print the model name
        print("processing model: ", model['model_name'])

        # Extract the runs for this model
        runs = model['runs']

        # if the runs are a range e.g. 1-3, then split them
        if '-' in runs:
            runs = runs.split('-')
            # convert the strings to integers
            runs = list(map(int, runs))
            # create a list of runs
            runs = list(range(runs[0], runs[1]+1))
        elif ',' in runs:
            runs = runs.split(',')
            # convert the strings to integers
            runs = list(map(int, runs))
        else:
            runs = [int(runs)]

        # Print the runs
        print("runs for model ", model['model_name'], ": ", runs)

        # Loop over the runs
        for run in runs:

            # Print the run being processed
            print("processing run: ", run)

            # Extract the initialization schemes for this model
            init_scheme = model['init_schemes']

            # if the init schemes are not a single number, then echo an error
            # and exit
            if ',' in init_scheme:
                print("Error, init schemes are not a single number")
                return None
            elif '-' in init_scheme:
                print("Error, init schemes are not a single number")
                return None
            
            # Extract the physics schemes for this model
            physics_schemes = model['physics_scheme']
            
            # If the physics schemes are a range, then split them
            # and loop over them
            if '-' in physics_schemes:
                physics_schemes = physics_schemes.split('-')
                # convert the strings to integers
                physics_schemes = list(map(int, physics_schemes))
                # create a list of init schemes
                physics_schemes = list(range(physics_schemes[0], physics_schemes[1]+1))
            elif ',' in physics_schemes:
                physics_schemes = physics_schemes.split(',')
                # convert the strings to integers
                physics_schemes = list(map(int, physics_schemes))

                # Loop over the init schemes
                for p in physics_schemes:

                    # Extract the forcing schemes for this model
                    forcing_scheme = model['forcing_scheme']

                    # if the forcing schemes are not a single number, then echo an error
                    # and exit
                    if ',' in forcing_scheme:
                        print("Error, forcing schemes are not a single number")
                        return None
                    elif '-' in forcing_scheme:
                        print("Error, forcing schemes are not a single number")
                        return None
                    else:
                        forcing_scheme = [int(forcing_scheme)]

                    # Merge the time axis of the files
                    # using the merge_time_axis function
                    merged_file = merge_time_axis(model['model_name'], var, run, init_scheme, p, forcing_scheme, dic.base_path_example)

                    # Check that the merged file exists
                    if merged_file is None:
                        print("Error, merged file does not exist")
                        return None

                    # Now regrid the file
                    # using the regrid function
                    regridded_file = regrid(model['model_name'], var, run, init_scheme, p, forcing_scheme, region)

                    # Check that the regridded file exists
                    if regridded_file is None:
                        print("Error, regridded file does not exist")
                        return None
            else:
                # Set up the physics scheme
                physics_scheme = [int(physics_schemes)]

                # Set up the forcing scheme
                forcing_scheme = [int(model['forcing_scheme'])]

                # Merge the time axis of the files
                # using the merge_time_axis function
                merged_file = merge_time_axis(model['model_name'], var, run, init_scheme, physics_scheme, forcing_scheme, dic.base_path_example)

                # Check that the merged file exists
                if merged_file is None:
                    print("Error, merged file does not exist")
                    return None
                
                # Now regrid the file
                # using the regrid function
                regridded_file = regrid(model['model_name'], var, run, init_scheme, physics_scheme, forcing_scheme, region)

                # Check that the regridded file exists
                if regridded_file is None:
                    print("Error, regridded file does not exist")
                    return None


            


