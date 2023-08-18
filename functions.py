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
            
# Now we want to define a function which will constrain the historical data
# to given years
# this function takes as arguments: the historical data dictionary, the start year and the end year, the season, the model and the member index
def constrain_historical_data_season(historical_data, start_year, end_year, season, model, member):
    """
    Constrains the historical data to given years and season.
    """

    # Extract the data for this model and member
    data = historical_data[model][member]

    # print the type of dat
    print("type of data in constrain historical data season", type(data))

    # Verify that the data is an xarray dataset
    if not isinstance(data, xr.Dataset):
        print("Error, data is not an xarray dataset")
        return None

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
    except e as err:
        print("Error, failed to constrain data: ", err)
        return None
    

# Now we want to define a function which will calculate anomalies for the historical data
# This function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the model name and the member index
def calculate_historical_anomalies_season(historical_data, model, member):
    """
    Calculates the anomalies for the historical data.
    """

    # Extract the data for this model and member
    data = historical_data[model][member].psl

    # Print the values of the data
    print("data values: ", data.values)

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if data.isnull().all():
        print("Data is null when calculating anomalies")
        print("Error, data contains only NaN values")
        return None

    try:
        # print that we are calculating the anomalies
        print("Calculating anomalies")

        # Calculate the mean over the time axis
        data_climatology = data.mean(dim='time')

        # Calculate the anomalies
        data_anomalies = data - data_climatology

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
def calculate_annual_mean_anomalies(historical_data, model, member, season):
    """
    Calculates the annual mean anomalies for the historical data.
    """

    # Extract the data for this model and member
    data = historical_data[model][member].psl

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if data.isnull().all():
        print("Error, data contains only NaN values")
        return None
    
    # print the type of the dic.season_timeshift
    print("type of dic.season_timeshift: ", type(dic.season_timeshift))
    
    # Set up the season from the season_timeshift dictionary
    #     season_timeshift = [
    #     {'season': 'DJF', 'timeshift': -2},
    #     {'season': 'NDJF', 'timeshift': -2},
    #     {'season': 'DJFM', 'timeshift': -3},
    #     {'season': 'NDJFM', 'timeshift': -3},
    #     {'season': 'NDJ', 'timeshift': -1},
    #     {'season': 'ONDJ', 'timeshift': -1},
    # ]
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

            # Shift the time axis by the timeshift value
            data = data.shift(time=season['timeshift'])

            # Calculate the annual mean
            data = data.resample(time='Y').mean(dim='time')
        
        except e as err:
            print("Error, failed to shift time axis: ", err)
            return None
        
    else:
        # If season is not defined as 'season' within the dictionary
        # then we calculate the annual mean
        try:
            # Calculate the annual mean
            data = data.resample(time='Y').mean(dim='time')

        except e as err:
            print("Error, failed to calculate annual mean: ", err)
            return None
        
    # Return the annual mean anomalies
    return data

# We want to define a function which will calculate the running mean
# of the annual mean anomalies
# this function takes as arguments: the historical data dictionary
# which contains the data for the selected years and season
# the model name and the member index
# and the forecast range - e.g years 2-9

def calculate_running_mean(historical_data, model, member, forecast_range):
    """
    Calculates the running mean for the historical data.
    """

    # Extract the data for this model and member
    data = historical_data[model][member].psl

    # Verify that the data is an xarray dataset
    # if not isinstance(data, xr.Dataset):
    #     print("Error, data is not an xarray dataset")
    #     return None
    
    # Check that the xarray dataset contains values other than NaN
    if data.isnull().all():
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
        return data
    # If the rolling mean value is greater than 1, then we need to calculate the rolling mean
    else:
        try:
            
            # Calculate the rolling mean
            data = data.rolling(time=rolling_mean_value, center=True).mean()

            # Get rid of the data for the years which are now NaN
            data = data.dropna(dim='time', how='all')

            # Verify that the data is not empty
            if data.sizes['time'] == 0:
                print("Error, data is empty")
                return None

            # Return the data
            return data
        
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
            data = constrain_historical_data_season(historical_data, start_year, end_year, season, model, member)

            # Check that the data is not empty
            if data is None:
                print("Error, data is empty post year and season constraint")
                return None
            
            # Check that this data exists by printing the data
            print("data: ", historical_data[model][member])

            # Calculate the anomalies
            data = calculate_historical_anomalies_season(historical_data, model, member)

            # Check that the data is not empty
            if data is None:
                print("Error, data is empty post anoms")
                return None

            # Calculate the annual mean anomalies
            data = calculate_annual_mean_anomalies(historical_data, model, member, season)

            # Check that the data is not empty
            if data is None:
                print("Error, data is empty post annual mean")
                return None

            # Calculate the running mean
            data = calculate_running_mean(historical_data, model, member, forecast_range)

            # Check that the data is not empty
            if data is None:
                print("Error, data is empty post running mean")
                return None

            # Add the data to the member dictionary
            member_dict[member] = data

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
        # sys.exit()

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
        regrid_sel_region_dataset = xr.open_mfdataset(regrid_sel_region_file, combine='by_coords', chunks={"time": 50})[obs_var_name]

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



