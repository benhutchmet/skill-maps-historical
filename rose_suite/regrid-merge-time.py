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
import argparse

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
        
# Write the function to regrid the files
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
    merged_file = os.path.join(merged_dir, merged_filename)

    # Print a message to say that the merged file is being searched for
    print("Searching for merged file: ", merged_file)

    # use glob to find the merged file
    merged_file = glob.glob(merged_file)

    # Print the type and value of the merged file from glob
    print("type(merged_file): ", type(merged_file))
    print("merged_file: ", merged_file)

    # if merged_file is an empty list then continue
    if len(merged_file) == 0:
        print("Error, merged file not found")
        return None

    # Check that the merged file exists
    if not os.path.exists(merged_file[0]):
        print("Error, merged file does not exist regrid: ", merged_file[0])
        return None

    print("merged_file[0]: ", merged_file[0])

    # Set up the merged file name
    merged_filename = merged_file[0].split('/')[-1]
    
    # Now construct the output file name
    # from the base name of the merged file
    output_filename = merged_filename + '_' + region + '_regrid.nc'
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
        except Exception as err:
            print("Error, failed to use cdo remapbil: ", err)
            return None
        
# Now we want to write a function to call the merge_time_axis 
# and regrid functions
def call_mergetime_regrid(model, variable, region):
    """
    Function to call the merge_time_axis and regrid functions.
    
    Arguments:
    ----------
    model:    Model name, e.g. 'HadGEM3-GC31-MM'
    variable: Variable name, e.g. 'tas'
    region:   Region name, e.g. 'north-atlantic'

    Returns:
    --------

    """

    # Print the model, variable and region
    print("Calling merge_time_axis and regrid functions for:")
    print("model: ", model)
    print("variable: ", variable)
    print("region: ", region)

    # First check that whether a dictionary exists for the variable
    # Using the model dictionary list
    model_dictionary_list = dic.model_dictionary_list

    # Loop over to see whether one exists for the variable
    for model_dictionary in model_dictionary_list:
        if variable in model_dictionary:
            print("Found model dictionary for variable: ", variable)
            print("model_dictionary: ", model_dictionary)
            # Now set the model_dictionary to be used
            model_dictionary = model_dictionary
            break
        else:
            print("No model dictionary found for variable: ", variable)
            print("model_dictionary: ", model_dictionary)
            # Now set the model_dictionary to be used
            model_dictionary = None

    # Check that the model_dictionary is not None
    if model_dictionary is None:
        print("Error, model_dictionary is None")
        return None
    
    # Now we need to check whether the model_dictionary contains the model
    # these are stored as 'model_name' in the dictionary
    # Loop over the model_dictionary to see whether the model exists
    for model_name in model_dictionary['model_name']:
        if model == model_name:
            print("Found model: ", model)
            print("model_name: ", model_name)
            # Now set the model_name to be used
            model_name = model_name
            break
        else:
            print("No model found: ", model)
            print("model_name: ", model_name)
            # Now set the model_name to be used
            model_name = None

    # Check that the model_name is not None
    if model_name is None:
        print("Error, model_name is None")
        return None
    
    # Extract the runs for the model
    runs = model_dictionary['runs'][model_name]

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
    print("runs for model ", model_dictionary[model_name], ": ", runs)

    # Loop over the runs
    for run in runs:

        # Print the run
        print("processing run: ", run)

        # Extract the init schemes for the model
        init_scheme = model_dictionary['init_schemes'][model_name]

        # if the init schemes are not a single number, then echo an error
        # and exit
        if ',' in init_scheme:
            print("Error, init schemes are not a single number")
            return None
        elif '-' in init_scheme:
            print("Error, init schemes are not a single number")
            return None

        # Print the init scheme being processed
        print("init_scheme: ", init_scheme)

        # Extract the physics schemes for the model
        # these are stored as 'physics_scheme' in the dictionary
        physics_schemes = model_dictionary['physics_scheme'][model_name]

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

            # Print that there are multiple physics schemes
            # and that they will be looped over
            print("Multiple physics schemes found for model: ", model_name)

            for physics_scheme in physics_schemes:
                
                # Print the physics scheme being processed
                print("physics_scheme: ", physics_scheme)

                # Extract the forcing schemes for the model
                # these are stored as 'forcing_scheme' in the dictionary
                # and are a string
                forcing_scheme = model_dictionary['forcing_scheme'][model_name]

                # if the forcing schemes are not a single number, then echo an error
                # and exit
                if ',' in forcing_scheme:
                    print("Error, forcing schemes are not a single number")
                    return None
                elif '-' in forcing_scheme:
                    print("Error, forcing schemes are not a single number")
                    return None
                else:
                    forcing_scheme = int(forcing_scheme)

                # Now call the merge_time_axis function
                # to merge the files along the time axis
                merged_file = merge_time_axis(model_name, variable, run, init_scheme, physics_scheme, forcing_scheme, dic.base_path_example)

                # print the type of the merged_file
                print("type(merged_file): ", type(merged_file))

                # Check that the merged_file is not None
                if merged_file is None:
                    print("Error, merged_file is None")
                    #return None

                # Now call the regrid function
                regridded_file = regrid(model_name, variable, run, init_scheme, physics_scheme, forcing_scheme, region)

                # if the regridded_file is None, then continue
                if regridded_file is None:
                    print("Error, regridded_file is None")
                    #return None

        else:
            # Set the physics scheme to be used
            physics_scheme = int(physics_schemes)

            # Print the physics scheme being processed
            print("physics_scheme: ", physics_scheme)

            # Extract the forcing schemes for the model
            # these are stored as 'forcing_scheme' in the dictionary
            forcing_scheme = model_dictionary['forcing_scheme'][model_name]

            # print the forcing scheme
            print("forcing_scheme: ", forcing_scheme)

            # if the forcing schemes are not a single number, then echo an error
            # and exit
            if ',' in forcing_scheme:
                print("Error, forcing schemes are not a single number")
                return None
            
            # Merge the files along the time axis
            merged_file = merge_time_axis(model_name, variable, run, init_scheme, physics_scheme, forcing_scheme, dic.base_path_example)

            # print the type of the merged_file
            print("type(merged_file): ", type(merged_file))

            # Check that the merged_file is not None
            if merged_file is None:
                print("Error, merged_file is None")
                #return None

            # Now call the regrid function
            regridded_file = regrid(model_name, variable, run, init_scheme, physics_scheme, forcing_scheme, region)

            # if the regridded_file is None, then continue
            if regridded_file is None:
                print("Error, regridded_file is None")
                #return None

# Define a main function
# which extracts the model, variable and region from the command line
# and calls the call_mergetime_regrid function
def main():
    """
    Main function to extract the model, variable and region from the command line. Then calls the call_mergetime_regrid function.
    
    :return: None
    """    

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name, e.g. HadGEM3-GC31-MM')
    parser.add_argument('variable', type=str, help='Variable name, e.g. tas')
    parser.add_argument('region', type=str, help='Region name, e.g. north-atlantic')

    # Extract the arguments
    args = parser.parse_args()

    # Extract the model, variable and region
    model = args.model
    variable = args.variable
    region = args.region

    # Print the model, variable and region
    print("model: ", model)
    print("variable: ", variable)
    print("region: ", region)

    try:
        # Call the call_mergetime_regrid function
        call_mergetime_regrid(model, variable, region)
    except Exception as err:
        print("[ERROR] Failed to call call_mergetime_regrid function for model: ", model + ", variable: ", variable + ", region: ", region, err)

# Call the main function
# If the modules is executed directly
if __name__ == '__main__':
    main()