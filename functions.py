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
import multiprocessing
import dask.array as da
import dask.distributed as dd
import dask

# Import CDO
from cdo import *
cdo = Cdo()

# Import the dictionaries
import dictionaries as dic

# Broad TODOs:
# TODO: Also create plots of the MSSS and RPC for the MMM and for each model
# TODO: Quantify the benefit of initialization by comparing the scores between the dcppA-hindcast and historical experiments
# TODO: look into quantifying the benefit of NAO-matching for spatial skill (wind speed)

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
        print("dirs: ", dirs)
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
            
            # Print the init scheme being processed
            print("processing init scheme: ", init_scheme)
            
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

                    # Print the physics scheme being processed
                    print("processing physics scheme: ", p)

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
                        forcing_scheme = int(forcing_scheme)

                    # Merge the time axis of the files
                    # using the merge_time_axis function
                    merged_file = merge_time_axis(model['model_name'], var, run, init_scheme, p, forcing_scheme, dic.base_path_example)

                    # print the merged_file
                    print("type of merged file", type(merged_file))
                    print("merged_file", merged_file)
                    
                    # Check that the merged file exists
                    if merged_file is None:
                        print("Error, merged file does not exist in call_mergetime_regrid")

                    # Now regrid the file
                    # using the regrid function
                    regridded_file = regrid(model['model_name'], var, run, init_scheme, p, forcing_scheme, region)

                    # Check that the regridded file exists
                    if regridded_file is None:
                        print("Error, regridded file does not exist")
                        
            else:
                # Set up the physics scheme
                physics_scheme = int(physics_schemes)

                # Set up the forcing scheme
                forcing_scheme = int(model['forcing_scheme'])

                # Print the physics scheme being processed
                print("processing physics scheme: ", physics_scheme)

                # Print the forcing scheme being processed
                print("processing forcing scheme: ", forcing_scheme)

                # Merge the time axis of the files
                # using the merge_time_axis function
                copied_file = merge_time_axis(model['model_name'], var, run, init_scheme, physics_scheme, forcing_scheme, dic.base_path_example)
                
                # Now regrid the file
                # using the regrid function
                regridded_file = regrid(model['model_name'], var, run, init_scheme, physics_scheme, forcing_scheme, region)

                # Print the type of the regridded file
                print("type of regridded file", type(regridded_file))
                # print("regridded_file", regridded_file[0])

                # if the type of the regridded file is none
                # then echo an error and exit
                if regridded_file is None:
                    print("Error, regridded file does not exist")


# Define a new function to load the processed historical data
# This function will take as arguments: the base directory 
# where the data are stored, the models, the variable name, the region name, the forecast range and the season.
def load_processed_historical_data(base_dir, models, variable, region, forecast_range, season):
    """
    Load the processed historical data from its base directory into a dictionary of datasets.

    Arguments:
        base_dir -- the base directory where the data are stored
        models -- the models
        variable -- the variable name
        region -- the region name
        forecast_range -- the forecast range

    Returns:
        historical_data -- a dictionary of datasets grouped by model and member.
    """

    # Initialize the dictionary to store the data
    historical_data = {}

    # Loop over the models
    for model in models:
            
        # Print the model name
        print("processing model: ", model)

        # Create an empty list to store the datasets
        # within the dictionary
        historical_data[model] = []

        # Set up the file path for the model
        # First the directory path
        # base_dir = "/home/users/benhutch/skill-maps-processed-data/historical"
        files_path = base_dir + '/' + variable + '/' + model + '/' + region + '/' + 'years_' + forecast_range + '/' + season + '/' + 'outputs' + '/' + 'processed' + '/' + '*.nc'

        # Print the files path
        print("files_path: ", files_path)

        # Find the files which match the path
        files = glob.glob(files_path)

        # If the list of files is empty, then print a warning and exit
        if len(files) == 0:
            print("Warning, no files found for model: ", model)
            return None

        # Loop over the files
        for file in files:
            # Open the dataset using xarray
            data = xr.open_dataset(file, chunks={'time': 100, 'lat': 45, 'lon': 45})

            # Extract the variant_label
            variant_label = data.attrs['variant_label']

            # Print the variant_label
            print("loading variant_label: ", variant_label)

            # Add the data to the dictionary
            # Using the variant_label as the key
            historical_data[model].append(data)

    # Return the historical data dictionary
    return historical_data

# Set up a function which which given the dataset and variable
# will extract the data for the given variable and the time dimension
def process_historical_members(model_member, variable):
    """
    For a given model member, extract the data for the given variable and the time dimension.
    """
    
    # Print the variable name
    print("processing variable: ", variable)

    # Check that the variable name is valid
    if variable not in dic.variables:
        print("Error, variable name not valid")
        return None

    try:
        # Extract the data for the given variable
        variable_data = model_member[variable]
    except Exception as error:
        print(error)
        print("Error, failed to extract data for variable: ", variable)
        return None
    
    # Extract the time dimension
    try:
        # Extract the time dimension
        historical_time = model_member["time"].values

        # Change these to a year format
        historical_time = historical_time.astype('datetime64[Y]').astype(int) + 1970
    except Exception as error:
        print(error)
        print("Error, failed to extract time dimension")
        return None
    
    # If either the variable data or the time dimension are None
    # then return None
    if variable_data is None or historical_time is None:
        print("Error, variable data or time dimension is None")
        sys.exit(1)

    return variable_data, historical_time

# Now write the outer function which will call this inner function
def extract_historical_data(historical_data, variable):
    """
    Outer function to process the historical data by extracting the data 
    for the given variable and the time dimension.
    """

    # Create empty dictionaries to store the data
    variable_data_by_model = {}
    historical_time_by_model = {}

    # Loop over the models
    for model in historical_data:
        # Print the model name
        print("processing model: ", model)

        # Create empty lists to store the data
        variable_data_by_model[model] = []
        historical_time_by_model[model] = []

        # Loop over the members
        for member in historical_data[model]:
            # Print the member name
            print("processing member: ", member)

            # print the type of the historical data
            print("type of historical_data: ", type(member))

            # print the dimensions of the historical data
            print("dimensions of historical_data: ", member.dims)

            # Format as a try except block    
            try:
                # Process the historical data
                variable_data, historical_time = process_historical_members(member, variable)

                # Append the data to the list
                variable_data_by_model[model].append(variable_data)
                historical_time_by_model[model].append(historical_time)
            except Exception as error:
                print(error)
                print("Error, failed to process historical data using process_historical_members in outer function")
                return None
            
    # Return the data
    return variable_data_by_model, historical_time_by_model


# Now we want to define a function to load the historical data
# As a dictionary of xarray datasets for each model
# This will have dimensions: [model, members, time, lat, lon]
# This function will take as arguments: the dictionary of models, the variable name, the region name
def load_historical_data(model_dict, var, region):
    """
    Loads the historical data as a dictionary of xarray datasets for each model.
    """
    # Initialize the dictionary
    historical_data = {}

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in model_dict:
        # Print the model name
        print("processing model: ", model['model_name'])

        # Initialize the member dictionary
        member_dict = {}

        # Set up the base dir
        # base_dir = str(dic.canari_base_path_historical)

        # set up the model
        model = model['model_name']

        # print the types of all of the variables
        # to find out which is dict
        print("type of var", type(var))
        print("type of model", type(model))
        print("type of region", type(region))
        
        # Set up the directory path
        # Where all of the members (each unique r?i?p?f? combination) are stored
        # We want the regridded files
        regrid_files = dic.canari_base_path_historical + '/' + var + '/' + model + '/regrid' + '/' + var + '_' + 'Amon' + '_' + model + '_' + 'historical' + '_' + 'r*i?p?f?' + '_*' + region + '_regrid.nc'

        print("regrid_files:", regrid_files)
        
        # Check that the regrid files exist
        if len(glob.glob(regrid_files)) == 0:
            print("Error, regrid files do not exist")
            return None
        
        # Count the number of files
        num_files = len(glob.glob(regrid_files))

        # Print the number of files
        print("number of files for model ", model, ": ", num_files)

        # Set up the member counter
        member = 0

        # Load the data for all members using xr.open_mfdataset()
        data = xr.open_mfdataset(regrid_files, combine='nested', concat_dim='member', parallel=True, chunks={'time': 50})

        # Loop over the members
        for member in range(num_files):
            # Get the data for this member
            member_data = data.isel(member=member)

            # Check if the data is full of NaNs
            if np.isnan(member_data[var]).all():
                print("Error, data is full of NaNs")
                return None

            # Add this data to the member dictionary
            member_dict[member] = member_data

        # Add the member dictionary to the historical data dictionary
        historical_data[model] = member_dict

    # Print the historical data dictionary
    # print("historical_data:", historical_data)
    print("type of historical_data", type(historical_data))
    print("shape of historical_data", np.shape(historical_data))

    # Return the historical data dictionary
    return historical_data

# Function to constrain the years of the historical data
# Define a function to constrain the years to the years that are in all of the model members
def constrain_years_processed_hist(model_data, models):
    """
    Constrains the years to the years that are in all of the models.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    constrained_data (dict): The model data with years constrained to the years that are in all of the models.
    """
    # Initialize a list to store the years for each model
    years_list = []

    # Print the models being proces
    # print("models:", models)
    
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            
            # Extract the years
            years = member.time.dt.year.values

            # If years is less than 48, then don't use this member
            if len(years) < 48:
                print("years less than 48")
                print("not including this member in common years for this model: ", model)
                continue

            # Append the years to the list of years
            years_list.append(years)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))

    # Print the common years for debugging
    # print("Common years:", common_years)
    # print("Common years type:", type(common_years))
    # print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # Print the years extracted from the model
            # print('model years', years)
            # print('model years shape', np.shape(years))

            # If years is less than 48, then don't use this member
            if len(years) < 48:
                print("years less than 48")
                print("not using this member for this model: ", model)
                continue
            
            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(years, common_years)

            # print("years in both shape", np.shape(years_in_both))
            # print("years in both", years_in_both)
            
            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_data:
                constrained_data[model] = []
            constrained_data[model].append(member)

    # # Print the constrained data for debugging
    # print("Constrained data:", constrained_data)

    return constrained_data

# Function to remove years with Nans
# checking for Nans in observed data
def remove_years_with_nans(observed_data, ensemble_mean, variable):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        ensemble_mean (xarray.Dataset): The ensemble mean (model data).
        variable (str): the variable name.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
    """

    # # Set the obs_var_name == variable
    obs_var_name = variable

    # Set up the obs_var_name
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    else:
        print("Invalid variable name")
        sys.exit()

    print("var name for obs", obs_var_name)
    
    for year in observed_data.time.dt.year.values[::-1]:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        
        # If there are any Nan values in the data
        if np.isnan(data.values).any():
            # Print the year
            # print(year)

            # Select the year from the observed data
            observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

            # for the model data
            ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year != year)

        # if there are no Nan values in the data for a year
        # then print the year
        # and "no nan for this year"
        # and continue the script
        else:
            # print(year, "no nan for this year")

            # exit the loop
            break

    return observed_data, ensemble_mean

# Function for calculating the spatial correlations
def calculate_correlations(observed_data, model_data, obs_lat, obs_lon):
    """
    Calculates the spatial correlations between the observed and model data.

    Parameters:
    observed_data (numpy.ndarray): The processed observed data.
    model_data (numpy.ndarray): The processed model data.
    obs_lat (numpy.ndarray): The latitude values of the observed data.
    obs_lon (numpy.ndarray): The longitude values of the observed data.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    try:
        # Initialize empty arrays for the spatial correlations and p-values
        rfield = np.empty([len(obs_lat), len(obs_lon)])
        pfield = np.empty([len(obs_lat), len(obs_lon)])

        # Print the dimensions of the observed and model data
        # print("observed data shape", np.shape(observed_data))
        # print("model data shape", np.shape(model_data))

        # Loop over the latitudes and longitudes
        for y in range(len(obs_lat)):
            for x in range(len(obs_lon)):
                # set up the obs and model data
                obs = observed_data[:, y, x]
                mod = model_data[:, y, x]

                # print the obs and model data
                # print("observed data", obs)
                # print("model data", mod)

                # Calculate the correlation coefficient and p-value
                r, p = stats.pearsonr(obs, mod)

                # If the correlation coefficient is negative, set the p-value to NaN
                if r < 0:
                    p = np.nan

                # Append the correlation coefficient and p-value to the arrays
                rfield[y, x], pfield[y, x] = r, p

        # Print the range of the correlation coefficients and p-values
        # to 3 decimal places
        print(f"Correlation coefficients range from {rfield.min():.3f} to {rfield.max():.3f}")
        print(f"P-values range from {pfield.min():.3f} to {pfield.max():.3f}")

        # Return the correlation coefficients and p-values
        return rfield, pfield

    except Exception as e:
        print(f"Error calculating correlations: {e}")
        sys.exit()

# function for processing the model data for plotting
def process_model_data_for_plot(model_data, models):
    """
    Processes the model data and calculates the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    ensemble_mean (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    """
    # Initialize a list for the ensemble members
    ensemble_members = []

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years_processed_hist(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Print
        print("extracting data for model:", model)

        # Set the ensemble members count to zero
        # if the model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0
        
        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Try to print values for each member
            # print("trying to print values for each member for debugging")
            # print("values for model:", model)
            # print("values for members:", member)
            # print("member values:", member.values)

            # Extract the lat and lon values
            lat = member.lat.values
            lon = member.lon.values

            # Extract the years
            years = member.time.dt.year.values

            # Print statements for debugging
            # print('shape of years', np.shape(years))
            # # print('years', years)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    # Convert the list of all ensemble members to a numpy array
    ensemble_members = np.array(ensemble_members)

    # Print the dimensions of the ensemble members
    # print("ensemble members shape", np.shape(ensemble_members))


    # Take the equally weighted ensemble mean
    ensemble_mean = ensemble_members.mean(axis=0)

    # Print the dimensions of the ensemble mean
    # print(np.shape(ensemble_mean))
    # print(type(ensemble_mean))
    # print(ensemble_mean)
        
    # Convert ensemble_mean to an xarray DataArray
    ensemble_mean = xr.DataArray(ensemble_mean, coords=member.coords, dims=member.dims)

    return ensemble_mean, lat, lon, years, ensemble_members_count

# Function for calculating the spatial correlations
def calculate_spatial_correlations(observed_data, model_data, models, variable):
    """
    Ensures that the observed and model data have the same dimensions, format and shape. Before calculating the spatial correlations between the two datasets.
    
    Parameters:
    observed_data (xarray.core.dataset.Dataset): The processed observed data.
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    rfield (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    pfield (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """
    # try:
    # Process the model data and calculate the ensemble mean
    ensemble_mean, lat, lon, years, ensemble_members_count = process_model_data_for_plot(model_data, models)

    # Debug the model data
    # print("ensemble mean within spatial correlation function:", ensemble_mean)
    # print("shape of ensemble mean within spatial correlation function:", np.shape(ensemble_mean))
    
    # Extract the lat and lon values
    obs_lat = observed_data.lat.values
    obs_lon = observed_data.lon.values
    # And the years
    obs_years = observed_data.time.dt.year.values

    # Initialize lists for the converted lons
    obs_lons_converted, lons_converted = [], []

    # Transform the obs lons
    obs_lons_converted = np.where(obs_lon > 180, obs_lon - 360, obs_lon)
    # add 180 to the obs_lons_converted
    obs_lons_converted = obs_lons_converted + 180

    # For the model lons
    lons_converted = np.where(lon > 180, lon - 360, lon)
    # # add 180 to the lons_converted
    lons_converted = lons_converted + 180

    # Print the observed and model years
    # print('observed years', obs_years)
    # print('model years', years)
    
    # Find the years that are in both the observed and model data
    years_in_both = np.intersect1d(obs_years, years)

    # print('years in both', years_in_both)

    # Select only the years that are in both the observed and model data
    observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))
    ensemble_mean = ensemble_mean.sel(time=ensemble_mean.time.dt.year.isin(years_in_both))

    # Remove years with NaNs
    observed_data, ensemble_mean = remove_years_with_nans(observed_data, ensemble_mean, variable)

    # Print the ensemble mean values
    # print("ensemble mean value after removing nans:", ensemble_mean.values)

    # # set the obs_var_name
    # obs_var_name = variable
    
    # # choose the variable name for the observed data
    # # Translate the variable name to the name used in the obs dataset
    # if obs_var_name == "psl":
    #     obs_var_name = "msl"
    # elif obs_var_name == "tas":
    #     obs_var_name = "t2m"
    # elif obs_var_name == "sfcWind":
    #     obs_var_name = "si10"
    # elif obs_var_name == "rsds":
    #     obs_var_name = "ssrd"
    # elif obs_var_name == "tos":
    #     obs_var_name = "sst"
    # else:
    #     print("Invalid variable name")
    #     sys.exit()

    # variable extracted already
    # Convert both the observed and model data to numpy arrays
    observed_data_array = observed_data.values / 100
    ensemble_mean_array = ensemble_mean.values / 100

    # Print the values and shapes of the observed and model data
    # print("observed data shape", np.shape(observed_data_array))
    # print("model data shape", np.shape(ensemble_mean_array))
    # print("observed data", observed_data_array)
    # print("model data", ensemble_mean_array)

    # Check that the observed data and ensemble mean have the same shape
    if observed_data_array.shape != ensemble_mean_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")

    # Calculate the correlations between the observed and model data
    rfield, pfield = calculate_correlations(observed_data_array, ensemble_mean_array, obs_lat, obs_lon)

    return rfield, pfield, obs_lons_converted, lons_converted, observed_data, ensemble_mean, ensemble_members_count

# Now we want to define a function which will constrain the historical data
# to given years
# this function takes as arguments: the historical data dictionary, the start year and the end year, the season, the model and the member index
def constrain_historical_data_season(historical_data, start_year, end_year, season, model, member):
    """
    Constrains the historical data to given years and season.
    """

    # Extract the data for this model and member
    data = historical_data[model][member].psl

    # print the type of dat
    print("type of data in constrain historical data season", type(data))

    # # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None

    # Extract the months from the season string
    months = dic.season_months[season]

    # Check that the months are not empty
    # and are a list of integers
    if len(months) == 0:
        print("Error, months are empty")
        return None
    elif not isinstance(months, list):
        print("Error, months are not a list")
        return None
    elif not all(isinstance(item, int) for item in months):
        print("Error, months are not all integers")
        return None

    # Format this as a try except block
    try:
        # Constrain the data to the given years
        data = data.sel(time=slice(str(start_year), str(end_year)))

        # Select the months from the dataset
        data = data.sel(time=data['time.month'].isin(months))

        # Return the data
        return data
    except Exception as e:
        print("Error, failed to constrain data: ", e)
        return None
    

def constrain_historical_data_season_dask(historical_data, start_year, end_year, season, model, member):
    """
    Constrains the historical data to given years and season.
    """

    # Extract the data for this model and member
    data = historical_data[model][member].psl

    # Convert the data to a dask array
    data = da.from_array(data)

    # Extract the months from the season string
    months = dic.season_months[season]

    # Check that the months are not empty
    # and are a list of integers
    if len(months) == 0:
        print("Error, months are empty")
        return None
    elif not isinstance(months, list):
        print("Error, months are not a list")
        return None
    elif not all(isinstance(item, int) for item in months):
        print("Error, months are not all integers")
        return None

    # Format this as a try except block
    try:
        # Constrain the data to the given years
        data = data[(data['time.year'] >= start_year) & (data['time.year'] <= end_year)]

        # Select the months from the dataset
        data = data[(da.isin(data['time.month'], months))]

        # Return the data
        return data
    except Exception as e:
        print("Error, failed to constrain data: ", e)
        return None
    

# Now we want to define a function which will calculate anomalies for the historical data
# This function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the model name and the member index
def calculate_historical_anomalies_season(constrained_data, historical_data, model, member):
    """
    Calculates the anomalies for the historical data.
    """

    # Extract the data for this member
    member_data = constrained_data

    # print the dimensions of the member data
    print("member_data dimensions: ", member_data.dims)

    # find all of the members for this model
    datasets = [historical_data[model][member] for member in historical_data[model]]

    # Concatenate the datasets along the member axis into an ensemble
    model_members_ensemble = xr.concat(datasets, dim='member')

    # Print the values of the data
    #print("data values: ", data.values)

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if member_data.isnull().all():
        print("Data is null when calculating anomalies")
        print("Error, data contains only NaN values")
        #return None

    try:
        # print that we are calculating the anomalies
        print("Calculating anomalies")

        # Calculate ensemble mean along the member axis
        model_members_ensemble_mean = model_members_ensemble.mean(dim='member')

        # take the time mean of the ensemble mean
        model_climatology = model_members_ensemble_mean.mean(dim='time')
        
        # print the data climatology
        # print("data_climatology: ", data_climatology)
        # print the shape of the data climatology
        print("dimensions of data_climatology: ", model_climatology.dims)

        # Calculate the anomalies
        data_anomalies = member_data - model_climatology

        # Return the anomalies
        return data_anomalies
    except e as err:
        print("Error, failed to calculate anomalies: ", err)
        return None
    
# Define a new function to calculate the annual mean anomalies
# From the monthly anomalies
# this function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the model name and the member index
def calculate_annual_mean_anomalies(constrained_data_anoms, season):
    """
    Calculates the annual mean anomalies for the historical data.
    """

    # Extract the data for this model and member
    member_data = constrained_data_anoms.psl

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if member_data.isnull().all():
        print("Error, data contains only NaN values")
        return None
    
    # print the type of the dic.season_timeshift
    print("type of dic.season_timeshift: ", type(dic.season_timeshift))
    
    # Set up the season from the season_timeshift dictionary
    season_index = [d['season'] for d in dic.season_timeshift].index(season)
    season = dic.season_timeshift[season_index]['timeshift']

    # print the season
    print("season: ", season)

    # If season is defined as 'season' within the dictionary
    # then we shift the time axis by the 'timeshift' value
    # and then calculate the annual mean
    if season == 'season':

        # print the timeshift value for this season
        print("timeshift value for season: ", season['timeshift'])

        try:

            print("shifting time axis by timeshift value: ", season['timeshift'])

            # Shift the time axis by the timeshift value
            member_data = member_data.shift(time=season['timeshift'])

            # Calculate the annual mean
            member_data = member_data.resample(time='Y').mean(dim='time')
        
        except Exception as err:
            print("Error, failed to shift time axis: ", err)
            return None
        
    else:
        # If season is not defined as 'season' within the dictionary
        # then we calculate the annual mean
        try:
            # Calculate the annual mean
            member_data = member_data.resample(time='Y').mean(dim='time')

        except Exception as err:
            print("Error, failed to calculate annual mean: ", err)
            return None
        
    # Return the annual mean anomalies
    return member_data

# We want to define a function which will calculate the running mean
# of the annual mean anomalies
# this function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the model name and the member index
# and the forecast range - e.g years 2-9

def calculate_running_mean(constrained_data_anoms_annual, forecast_range):
    """
    Calculates the running mean for the historical data.
    """

    # Extract the data for this model and member
    member_data = constrained_data_anoms_annual

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if member_data.isnull().all():
        # Print a message to say that the data contains only NaN values
        print("Error, data contains only NaN values")
        return None
    
    # set up the start and end years
    start_year = int(forecast_range.split('-')[0])
    end_year = int(forecast_range.split('-')[1])

    # Print the forecast range
    print("forecast range: ", start_year, "-", end_year)

    # Calculate the rolling mean value
    rolling_mean_value = end_year - start_year + 1

    # Print the rolling mean value
    print("rolling mean value: ", rolling_mean_value)

    # If the rolling mean value is 1, then we don't need to calculate the rolling mean
    if rolling_mean_value == 1:
        print("rolling mean value is 1, no need to calculate rolling mean")
        return member_data
    # If the rolling mean value is greater than 1, then we need to calculate the rolling mean
    else:
        try:
            
            # Calculate the rolling mean
            member_data = member_data.rolling(time=rolling_mean_value, center=True).mean()

            # Get rid of the data for the years which are now NaN
            member_data = member_data.dropna(dim='time', how='all')

            # Verify that the data is not empty
            if member_data.sizes['time'] == 0:
                print("Error, data is empty")
                return None

            # Return the data
            return member_data
        
        except e as err:
            print("Error, failed to calculate rolling mean and drop Nans: ", err)
            return None


# Now we want to define a function which will loop over the models and members
# to calculate the annual mean anomalies where the rolling mean has been taken
# this function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the season and forecast range
def process_historical_data(historical_data, season, forecast_range, start_year, end_year):
    """
    Loops over the models and members to calculate the annual mean anomalies where the rolling mean has been taken.
    
    Arguments:
    historical_data -- the historical data dictionary
    season -- the season
    forecast_range -- the forecast range
    start_year -- the start year
    end_year -- the end year

    Returns:
    historical_data_processed -- the processed historical data dictionary
    
    """

    # Initialize the dictionary
    historical_data_processed = {}

    # Set up the test model case
    # test_model = [ "BCC-CSM2-MR" ]

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in historical_data:
        # Print the model name
        print("processing model: ", model)

        # Initialize the member dictionary
        member_dict = {}

        # Loop over the members
        # these are defined by the member index key in the dictionary
        for member in historical_data[model]:

            # Print the member index
            print("processing member: ", member)

            # Constrain the data to the given year and season
            constrained_data = constrain_historical_data_season(historical_data, start_year, end_year, season, model, member)

            # Check that the data is not empty
            if constrained_data is None:
                print("Error, data is empty post year and season constraint")
                return None
            
            # Check that this data exists by printing the dimensions
            # print("constrained_data dimensions: ", constrained_data.dims)

            # Calculate the anomalies
            constrained_data_anoms = calculate_historical_anomalies_season(constrained_data, historical_data, model, member)

            # Check that the data is not empty
            if constrained_data_anoms is None:
                print("Error, data is empty post anoms")
                return None

            # print the values of the data
            # print("constraints_data_anoms values: ", constrained_data_anoms.psl.values)

            # Calculate the annual mean anomalies
            constrained_data_anoms_annual = calculate_annual_mean_anomalies(constrained_data_anoms, season)

            # Check that the data is not empty
            if constrained_data_anoms_annual is None:
                print("Error, data is empty post annual mean")
                return None

            # Calculate the running mean
            constrained_data_anoms_annual_rm = calculate_running_mean(constrained_data_anoms_annual, forecast_range)

            # Check that the data is not empty
            if constrained_data_anoms_annual_rm is None:
                print("Error, data is empty post running mean")
                return None

            # Add the data to the member dictionary
            member_dict[member] = constrained_data_anoms_annual_rm

        # Add the member dictionary to the historical data dictionary
        historical_data_processed[model] = member_dict

    # Return the historical data dictionary
    return historical_data_processed

# Try this using dask delayed for parallel processing
def process_historical_data_dask(historical_data, season, forecast_range, start_year, end_year):
    """
    Loops over the models and members to calculate the annual mean anomalies where the rolling mean has been taken.
    
    Arguments:
    historical_data -- the historical data dictionary
    season -- the season
    forecast_range -- the forecast range
    start_year -- the start year
    end_year -- the end year

    Returns:
    historical_data_processed -- the processed historical data dictionary
    
    """

    # Initialize the dictionary
    historical_data_processed = {}

    # Set up the test model case
    test_model = [ "BCC-CSM2-MR" ]

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in test_model:
        # Print the model name
        print("processing model: ", model)

        # Initialize the member dictionary
        member_dict = {}

        # Loop over the members
        # these are defined by the member index key in the dictionary
        delayed_members = []
        for member in historical_data[model]:

            # Print the member index
            print("processing member: ", member)

            # Constrain the data to the given year and season
            delayed_constrained_data = dask.delayed(constrain_historical_data_season)(historical_data, start_year, end_year, season, model, member)

            # Calculate the anomalies
            delayed_constrained_data_anoms = dask.delayed(calculate_historical_anomalies_season)(delayed_constrained_data, historical_data, model, member)

            # Calculate the annual mean anomalies
            delayed_constrained_data_anoms_annual = dask.delayed(calculate_annual_mean_anomalies)(delayed_constrained_data_anoms, season)

            # Calculate the running mean
            delayed_constrained_data_anoms_annual_rm = dask.delayed(calculate_running_mean)(delayed_constrained_data_anoms_annual, forecast_range)

            # Add the data to the member dictionary
            delayed_member = dask.delayed(member_dict.__setitem__)(member, delayed_constrained_data_anoms_annual_rm)
            delayed_members.append(delayed_member)

        # Compute the delayed members using dask.compute()
        dask.compute(*delayed_members)

        # Add the member dictionary to the historical data dictionary
        historical_data_processed[model] = member_dict

    # Return the historical data dictionary
    return historical_data_processed


# try to run this process in parallel for the members of each mode
# to speed up the processing
def process_member_data(model, member, historical_data, season, forecast_range, start_year, end_year):
    """
    Processes the data for a single member of a given model.
    """
    # Open the netCDF files using dask
    file_pattern = historical_data[model][member]
    ds = dd.open_mfdataset(file_pattern, chunks={'time': 'auto'})

    # Constrain the data to the given year and season
    constrained_data = ds.sel(time=slice(f'{start_year}-{season}', f'{end_year}-{season}'))

    # Check that the data is not empty
    if constrained_data.time.size == 0:
        print("Error, data is empty post year and season constraint")
        return None

    # Calculate the anomalies
    climatology = constrained_data.groupby('time.dayofyear').mean('time')
    anomalies = constrained_data.groupby('time.dayofyear') - climatology

    # Check that the data is not empty
    if anomalies.time.size == 0:
        print("Error, data is empty post anoms")
        return None

    # Calculate the annual mean anomalies
    annual_mean_anomalies = anomalies.resample(time='AS').mean('time')

    # Check that the data is not empty
    if annual_mean_anomalies.time.size == 0:
        print("Error, data is empty post annual mean")
        return None

    # Calculate the running mean
    running_mean = annual_mean_anomalies.rolling(time=forecast_range, center=True).mean()

    # Check that the data is not empty
    if running_mean.time.size == 0:
        print("Error, data is empty post running mean")
        return None

    # Compute the running mean and return the processed data
    return (model, member, running_mean.compute())


# dask version
def process_historical_data_parallel(historical_data, season, forecast_range, start_year, end_year):
    """
    Loops over the models and members to calculate the annual mean anomalies where the rolling mean has been taken.
    Processes the members for each model in parallel using dask.

    Arguments:
    historical_data -- the historical data dictionary
    season -- the season
    forecast_range -- the forecast range
    start_year -- the start year
    end_year -- the end year

    Returns:
    historical_data_processed -- the processed historical data dictionary

    """

    # Initialize the dictionary
    historical_data_processed = {}

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in historical_data:
        # Print the model name
        print("processing model: ", model)

        # Initialize the member dictionary
        member_dict = {}

        # Get the list of members for this model
        members = list(historical_data[model].keys())

        # Create a distributed computing cluster
        client = dd.Client()

        # Process the members in parallel using dask
        futures = []
        for member in members:
            future = client.submit(process_member_data, model, member, historical_data, season, forecast_range, start_year, end_year)
            futures.append(future)

        # Get the results from the worker processes
        processed_data = client.gather(futures)

        # Close the distributed computing cluster
        client.close()

        # Add the processed data to the member dictionary
        for model_name, member_name, member_data in processed_data:
            member_dict[member_name] = member_data

        # Add the member dictionary to the historical data dictionary
        historical_data_processed[model] = member_dict

    # Return the historical data dictionary
    return historical_data_processed

# Define a function which will ensure that all of the members
# for all of the models have the same number of years
# this function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
def constrain_years(historical_data):
    """
    Ensures that all of the members for all of the models have the same number of years.
    """

    # Initialize a list to store the years for each model
    years_list = []

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in historical_data:
        # extract the model data
        model_data = historical_data[model]

        # Loop over the members in the model data
        for member in model_data:
            # extract the member data
            member_data = model_data[member]

            # extract the years from the member data
            years = member_data.time.dt.year.values

            # append the years to the years list
            years_list.append(years)

    # Check that the years list is not empty
    if len(years_list) == 0:
        print("Error, years list is empty")
        return None
    
    # Find the years that are common to all of the members
    common_years = set.intersection(*map(set, years_list))

    # Initialize the dictionary for the constrained year data
    historical_data_constrained = {}

    # Loop over the models
    # these are define by the model_name key in the dictionary
    for model in historical_data:
        # Extract the model data
        model_data = historical_data[model]

        # Loop over the members in the model data
        for member in model_data:
            # Extract the member data
            member_data = model_data[member]

            # Extract the years from the member data
            years = member_data.time.dt.year.values

            # Find the years which are common in the member data
            # and the common years
            years_shared = np.intersect1d(years, list(common_years))

            # Constrain the data to the common years
            member_data = member_data.sel(time=member_data.time.dt.year.isin(years_shared))

            # Add the data to the member dictionary
            if model not in historical_data_constrained:
                historical_data_constrained[model] = {}

            historical_data_constrained[model][member] = member_data

    # Return the historical data dictionary
    return historical_data_constrained
    
# Define a function which processes the historical data in preperation for calculating 
# the spatial correlations
# this function takes as arguments: the historical data dictionary
def process_historical_data_spatial_correlations(historical_data):
    """
    Processes the historical data in preperation for calculating the spatial correlations.
    """

    # Initialize a list for the ensemble members
    ensemble_members = []

    # Initialize a dictionary to store the number of ensemble members for each model
    ensemble_members_count = {}

    # Constrain the years for each model
    historical_data = constrain_years(historical_data)

    # Loop over the models
    for model in historical_data:
        # Extract the model data
        model_data = historical_data[model]

        # Extract the ensemble members for this model
        members = list(model_data.keys())

        # Add the members to the ensemble members list
        ensemble_members += members

        # Add the number of members to the ensemble members count dictionary
        ensemble_members_count[model] = len(members)

        # Loop over the ensemble members in the model data
        for member in model_data:
            # Extract the member data
            member_data = model_data[member]

            # Append the data to the ensemble members list
            ensemble_members.append(member_data)

            # Extract the lat and lon values
            lat = member_data.lat.values
            lon = member_data.lon.values

            # Check that the lat and lon values are not empty
            if len(lat) == 0:
                print("Error, lat values are empty")
                return None
            
            if len(lon) == 0:
                print("Error, lon values are empty")
                return None
            
            # Check that the lat and lon values are the same
            # for all ensemble members
            if not np.array_equal(lat, ensemble_members[0].lat.values):
                print("Error, lat values are not the same for all ensemble members")
                return None
            
            if not np.array_equal(lon, ensemble_members[0].lon.values):
                print("Error, lon values are not the same for all ensemble members")
                return None
            
            # extract the year values
            years = member_data.time.dt.year.values

            # Check that the years are not empty
            if len(years) == 0:
                print("Error, years are empty")
                return None
            
            # Check that the years are the same for all ensemble members
            if not np.array_equal(years, ensemble_members[0].time.dt.year.values):
                print("Error, years are not the same for all ensemble members")
                return None
            
            # Check that the ensemble members are not empty
            if len(ensemble_members) == 0:
                print("Error, ensemble members are empty")
                return None
            
    # convert the ensemble members list to a numpy array
    ensemble_members = np.array(ensemble_members)

    # take the equal weighted mean over the ensemble members
    ensemble_mean = ensemble_members.mean(axis=0)

    # Check that the ensemble mean is not empty
    if len(ensemble_mean) == 0:
        print("Error, ensemble mean is empty")
        return None
    
    # Check that the ensemble mean is not NaN
    if ensemble_mean.isnull().all():
        print("Error, ensemble mean contains only NaN values")
        return None
    
    # Convert ensemble mean to an xarray dataset
    ensemble_mean = xr.DataArray(ensemble_mean, coords=member_data.coords, dims=member_data.dims)

    # Return the ensemble mean
    return ensemble_mean

# Using cdo to do the regridding and selecting the region
def regrid_and_select_region(observations_path, region, obs_var_name):
    """
    Uses CDO remapbil and a gridspec file to regrid and select the correct region for the obs dataset. Loads for the specified variable.
    
    Parameters:
    observations_path (str): The path to the observations dataset.
    region (str): The region to select.

    Returns:
    xarray.Dataset: The regridded and selected observations dataset.
    """
    
    # First choose the gridspec file based on the region
    gridspec_path = "/home/users/benhutch/gridspec"

    # select the correct gridspec file
    if region == "north-atlantic":
        gridspec = gridspec_path + "/" + "gridspec-north-atlantic.txt"
    elif region == "global":
        gridspec = gridspec_path + "/" + "gridspec-global.txt"
    elif region == "azores":
        gridspec = gridspec_path + "/" + "gridspec-azores.txt"
    elif region == "iceland":
        gridspec = gridspec_path + "/" + "gridspec-iceland.txt"
    else:
        print("Invalid region")
        sys.exit()

    # Check that the gridspec file exists
    if not os.path.exists(gridspec):
        print("Gridspec file does not exist")
        sys.exit()

    # Create the output file path
    regrid_sel_region_file = "/home/users/benhutch/ERA5/" + region + "_" + "regrid_sel_region.nc"

    # Check if the output file already exists
    # If it does, then exit the program
    if os.path.exists(regrid_sel_region_file):
        print("File already exists")
        print("Loading ERA5 data")
    else:
        print("File does not exist")
        print("Processing ERA5 data using CDO")

        # Regrid and select the region using cdo 
        cdo.remapbil(gridspec, input=observations_path, output=regrid_sel_region_file)

    # Load the regridded and selected region dataset
    # for the provided variable
    # check whether the variable name is valid
    if obs_var_name not in ["psl", "tas", "sfcWind", "rsds", "tos"]:
        print("Invalid variable name")
        sys.exit()

    # Translate the variable name to the name used in the obs dataset
    if obs_var_name == "psl":
        obs_var_name = "msl"
    elif obs_var_name == "tas":
        obs_var_name = "t2m"
    elif obs_var_name == "sfcWind":
        obs_var_name = "si10"
    elif obs_var_name == "rsds":
        obs_var_name = "ssrd"
    elif obs_var_name == "tos":
        obs_var_name = "sst"
    else:
        print("Invalid variable name")
        sys.exit()

    # Load the regridded and selected region dataset
    # for the provided variable
    try:
        # Load the dataset for the selected variable
        regrid_sel_region_dataset = xr.open_mfdataset(regrid_sel_region_file, combine='by_coords', parallel=True, chunks={"time": 100, 'lat': 100, 'lon': 100})[obs_var_name]

        # Combine the two expver variables
        regrid_sel_region_dataset_combine = regrid_sel_region_dataset.sel(expver=1).combine_first(regrid_sel_region_dataset.sel(expver=5))

        return regrid_sel_region_dataset_combine

    except Exception as e:
        print(f"Error loading regridded and selected region dataset: {e}")
        sys.exit()

def select_season(regridded_obs_dataset_region, season):
    """
    Selects a season from a regridded observation dataset based on the given season string.

    Parameters:
    regridded_obs_dataset_region (xarray.Dataset): The regridded observation dataset for the selected region.
    season (str): A string representing the season to select. Valid values are "DJF", "MAM", "JJA", "SON", "SOND", "NDJF", and "DJFM".

    Returns:
    xarray.Dataset: The regridded observation dataset for the selected season.

    Raises:
    ValueError: If an invalid season string is provided.
    """

    try:
        # Extract the months from the season string
        if season == "DJF":
            months = [12, 1, 2]
        elif season == "MAM":
            months = [3, 4, 5]
        elif season == "JJA":
            months = [6, 7, 8]
        elif season == "JJAS":
            months = [6, 7, 8, 9]
        elif season == "SON":
            months = [9, 10, 11]
        elif season == "SOND":
            months = [9, 10, 11, 12]
        elif season == "NDJF":
            months = [11, 12, 1, 2]
        elif season == "DJFM":
            months = [12, 1, 2, 3]
        else:
            raise ValueError("Invalid season")

        # Select the months from the dataset
        regridded_obs_dataset_region_season = regridded_obs_dataset_region.sel(
            time=regridded_obs_dataset_region["time.month"].isin(months)
        )

        return regridded_obs_dataset_region_season
    except:
        print("Error selecting season")
        sys.exit()

def calculate_anomalies(regridded_obs_dataset_region_season):
    """
    Calculates the anomalies for a given regridded observation dataset for a specific season.

    Parameters:
    regridded_obs_dataset_region_season (xarray.Dataset): The regridded observation dataset for the selected region and season.

    Returns:
    xarray.Dataset: The anomalies for the given regridded observation dataset.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        obs_climatology = regridded_obs_dataset_region_season.mean("time")
        obs_anomalies = regridded_obs_dataset_region_season - obs_climatology
        return obs_anomalies
    except:
        print("Error calculating anomalies for observations")
        sys.exit()    


def calculate_annual_mean_anomalies_obs(obs_anomalies, season):
    """
    Calculates the annual mean anomalies for a given observation dataset and season.

    Parameters:
    obs_anomalies (xarray.Dataset): The observation dataset containing anomalies.
    season (str): The season for which to calculate the annual mean anomalies.

    Returns:
    xarray.Dataset: The annual mean anomalies for the given observation dataset and season.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-1)
        else:
            obs_anomalies_shifted = obs_anomalies

        # Calculate the annual mean anomalies
        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

def select_forecast_range(obs_anomalies_annual, forecast_range):
    """
    Selects the forecast range for a given observation dataset.

    Parameters:
    obs_anomalies_annual (xarray.Dataset): The observation dataset containing annual mean anomalies.
    forecast_range (str): The forecast range to select.

    Returns:
    xarray.Dataset: The observation dataset containing annual mean anomalies for the selected forecast range.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        
        forecast_range_start, forecast_range_end = map(int, forecast_range.split("-"))
        print("Forecast range:", forecast_range_start, "-", forecast_range_end)
        
        rolling_mean_range = forecast_range_end - forecast_range_start + 1
        print("Rolling mean range:", rolling_mean_range)

        # if rolling mean range is 1, then we don't need to calculate the rolling mean
        if rolling_mean_range == 1:
            print("rolling mean range is 1, no need to calculate rolling mean")
            return obs_anomalies_annual
        
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time=rolling_mean_range, center = True).mean()
        
        return obs_anomalies_annual_forecast_range
    except Exception as e:
        print("Error selecting forecast range:", e)
        sys.exit()

# Call the functions to process the observations
def process_observations(variable, region, region_grid, forecast_range, season, observations_path, obs_var_name):
    """
    Processes the observations dataset by regridding it to the model grid, selecting a region and season,
    calculating anomalies, calculating annual mean anomalies, selecting the forecast range, and returning
    the processed observations.

    Args:
        variable (str): The variable to process.
        region (str): The region to select.
        region_grid (str): The grid to regrid the observations to.
        forecast_range (str): The forecast range to select.
        season (str): The season to select.
        observations_path (str): The path to the observations dataset.
        obs_var_name (str): The name of the variable in the observations dataset.

    Returns:
        xarray.Dataset: The processed observations dataset.
    """

    # Check if the observations file exists
    if not os.path.exists(observations_path):
        print("Error, observations file does not exist")
        return None

    try:
        # Regrid using CDO, select region and load observation dataset
        # for given variable
        obs_dataset = regrid_and_select_region(observations_path, region, obs_var_name)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_dataset")
        # check_for_nan_values(obs_dataset)

        # Select the season
        # --- Although will already be in DJFM format, so don't need to do this ---
        regridded_obs_dataset_region_season = select_season(obs_dataset, season)

        # Print the dimensions of the regridded and selected region dataset
        print("Regridded and selected region dataset:", regridded_obs_dataset_region_season.time)

        # # Check for NaN values in the observations dataset
        # print("Checking for NaN values in regridded_obs_dataset_region_season")
        # check_for_nan_values(regridded_obs_dataset_region_season)
        
        # Calculate anomalies
        obs_anomalies = calculate_anomalies(regridded_obs_dataset_region_season)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_anomalies")
        # check_for_nan_values(obs_anomalies)

        # Calculate annual mean anomalies
        obs_annual_mean_anomalies = calculate_annual_mean_anomalies_obs(obs_anomalies, season)

        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_annual_mean_anomalies")
        # check_for_nan_values(obs_annual_mean_anomalies)

        # Select the forecast range
        obs_anomalies_annual_forecast_range = select_forecast_range(obs_annual_mean_anomalies, forecast_range)
        # Check for NaN values in the observations dataset
        # print("Checking for NaN values in obs_anomalies_annual_forecast_range")
        # check_for_nan_values(obs_anomalies_annual_forecast_range)

        # if the forecast range is "2-2" i.e. a year ahead forecast
        # then we need to shift the dataset by 1 year
        # where the model would show the DJFM average as Jan 1963 (s1961)
        # the observations would show the DJFM average as Dec 1962
        # so we need to shift the observations to the following year
        if forecast_range == "2-2":
            obs_anomalies_annual_forecast_range = obs_anomalies_annual_forecast_range.shift(time=1)


        return obs_anomalies_annual_forecast_range

    except Exception as e:
        print(f"Error processing observations dataset: {e}")
        sys.exit()



