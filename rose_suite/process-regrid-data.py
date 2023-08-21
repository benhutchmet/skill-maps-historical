#!/usr/bin/env python

"""
process-regrid-data.py
======================

A script which takes the arguments for model, variable, region, season,
forecast_range and start_year and end_year and then processes the merged 
and regridded data in a series of steps.

These steps are:
----------------
1. Select the season (e.g. DJFM) and years (e.g. 1960-2014) from the merged and regridded data.
2. Calculate and remove from members the model climatology for the selected season and years.
3. Take the seasonal mean by shifting the time axis back (in the case of DJFM) and then taking
the annual mean. Or just take the annual mean if the season does not cross the year boundary.
4. Calculate the running mean (e.g. 8 year running mean for years 2-9 forecast) for each member.
(If the forecast range is not 2-2)
----------------

Usage:
------

    python process-regrid-data.py <model> <variable> <region> <season> <forecast_range> <start_year> <end_year>

    e.g. python process-regrid-data.py HadGEM3-GC31-MM tas global DJFM 2-9 1960 2014

    model: Model name (e.g. HadGEM3-GC31-MM) or '5'.
    variable: Variable name (e.g. tas).
    region: Region name (e.g. global).
    season: Season name (e.g. DJFM).
    forecast_range: Forecast range (e.g. 2-9).
    start_year: Start year (e.g. 1960).
    end_year: End year (e.g. 2014).
----------------
"""

# Imports
import os
import sys
import glob
import re
import argparse
import time

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import scipy.stats as stats

# Import CDO
from cdo import *
cdo = Cdo()

# Set up the location of the dictionaries
dict_dir = '/home/users/benhutch/skill-maps-historical/'
sys.path.append(dict_dir)

# Import dictionaries
import dictionaries as dic

# Set up a function to load all of the data for a specified model, variable and region
def load_data(model, variable, region):
    """
    Loads the historical data for a specified model, variable and region.
    
    Parameters
    ----------
    model : str
        Model name (e.g. HadGEM3-GC31-MM).
    variable : str
        Variable name (e.g. tas).
    region : str
        Region name (e.g. global).
        
    Returns
    -------
    historical_data : xarray DataArray
        Historical data for the specified model, variable and region.
    """

    # Initialize a dictionary to store the data
    historical_data = {}

    # Print a message to the screen
    print('Loading data for ' + model + ' ' + variable + ' ' + region + '...')

    # Set up the path to the data for the regridded files
    regrid_files = dic.canari_base_path_historical + '/' + variable + '/' + model + '/regrid' + '/' + variable + '_' + 'Amon' + '_' + model + '_' + 'historical' + '_' + 'r*i?p?f?' + '_*' + region + '_regrid.nc'

    # print the path to the data
    print("Reggrided files path: ", regrid_files)

    # Check if the regridded files exist
    if len(glob.glob(regrid_files)) == 0:
        print("No regridded files found. Exiting...")
        sys.exit()

    # Count the number of regridded files
    num_regrid_files = len(glob.glob(regrid_files))

    # Print the number of regridded files
    print("Number of regridded files for " + model + ": " + str(num_regrid_files))

    # Set up the member counter
    member_counter = 0

    # Try an alternative method of loading the data
    # Loop over the regridded files
    for regrid_file in glob.glob(regrid_files):
        # Check if the file is a .nc file
        print("regrid_file: ", regrid_file)
        if regrid_file.endswith('.nc'):
            # Open the file using xarray
            data = xr.open_dataset(regrid_file, chunks={'time': 100, 'lat': 45, 'lon': 45})

            # Extract the variant_label
            variant_label = data.attrs['variant_label']

            # Print the variant_label
            print("the variant_label for " + regrid_file + " is: ", variant_label)

            # Add the data to the dictionary
            # using the variant_label as the key
            historical_data[variant_label] = data

            # Increment the member counter
            member_counter += 1

    # Print the number of members
    print("Number of members for " + model + ": " + str(member_counter))

    # # Loop over the members
    # for member in historical_data:
    #     print("member: ", member)
    #     # Look at the data for the first member
    #     print("historical_data for the member: ", historical_data[member])

    return historical_data

# Next we want to define a function to select the season and years from the data
def select_season_years(historical_data, season, start_year, end_year):
    """
    Selects the season and years from the data.
    """

    # Print a message to the screen
    print('Selecting the season: ' + season + ' and years: ' + start_year + '-' + end_year + '...')

    # Extract the month numbers for the season
    months = dic.season_months[season]

    # Check the months
    if len(months) == 0:
        print("Error, months are empty")
        return None
    elif not isinstance(months, list):
        print("Error, months are not a list")
        return None
    elif not all(isinstance(item, int) for item in months):
        print("Error, months are not all integers")
        return None

    # Print the months
    print("months: ", months)

    try:
        # Loop over the members
        for member in historical_data:
            # First select the years
            historical_data[member] = historical_data[member].sel(time=slice(str(start_year), str(end_year)))

            # Select the months
            historical_data[member] = historical_data[member].sel(time=historical_data[member]['time.month'].isin(months))

        # print the dimensions of the data for the first member
        # print("data post processing: ", historical_data)

        # Return the data
        return historical_data
    except Exception as Error:
        print(Error)
        print("Unable to select the season: " + season + " and years: " + start_year + "-" + end_year + "...")
        return None

# Now define a function to calculate and remove the model climatology for the selected season and years
def calculate_remove_model_climatology(historical_data_constrained):
    """
    For the selected season and years, calculate and remove the model climatology from each member.
    """

    # Concatenate the data along the variant_label dimension
    historical_data_constrained_ensemble = xr.concat(historical_data_constrained.values(), dim='variant_label')

    # Print the dimensions of the data
    print("Dimensions of the data after psl selected: ", historical_data_constrained_ensemble)

    # Print a message to the screen
    print('Calculating and removing the model climatology...')

    # Calculate the model climatology
    try:

        # Take the mean over all of the members
        members_ensemble_mean = historical_data_constrained_ensemble.mean(dim='variant_label')

        # Print the dimensions of the members_ensemble_mean
        print("Dimensions of the members_ensemble_mean: ", members_ensemble_mean.dims)

        # Take the time mean of this ensemble mean
        model_climatology = members_ensemble_mean.mean(dim='time')

        # print the values of the model_climatology
        # TODO: Make flexible for different variables (e.g. tas, psl, rsds etc.)
        print("Psl Values of the model_climatology: ", model_climatology.psl.values)

        # Print the shape of the psl values of the model_climatology
        # TODO: Make flexible for different variables (e.g. tas, psl, rsds etc.)
        print("Shape of the psl values of the model_climatology: ", model_climatology.psl.values.shape)

        # Print the dimensions of the model_climatology
        print("Dimensions of the model_climatology: ", model_climatology.dims)
    except Exception as error:
        print(error)
        print("Unable to calculate the model climatology...")
        return None
    
    # Now remove the model climatology from each member
    try:
        # Loop over the members
        for member in historical_data_constrained:
            # Print the type of the member
            print("Type of member: ", type(member))

            # Print the dimensions of the member
            print("Dimensions of the member: ", historical_data_constrained[member].dims)

            # Print the member
            # print("Member: ", historical_data_constrained[member])

            # # Print the values of the member
            # print("Values of the member: ", historical_data_constrained[member].values)


            # Remove the model climatology from the member
            # TODO: Make flexible for different variables (e.g. tas, psl, rsds etc.)
            historical_data_constrained[member].psl.values = historical_data_constrained[member].psl.values - model_climatology.psl.values
    except Exception as error:
        print(error)
        print("Unable to remove the model climatology...")
        return None
    
    # Print a message to the screen
    print('Removed the model climatology for the psl variable from model members...')

    # Loop over the members and print the values
    # for member in historical_data_constrained:
    #     # Print the values of the member
    #     print("Values of the member: ", historical_data_constrained[member].psl.values)

    # Return the data
    return historical_data_constrained


# Set up the main function for testing purposes
def main():
    """
    Main function which calls the functions to process the regridde data.
    """

    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name (e.g. HadGEM3-GC31-MM) or 5.')
    parser.add_argument('variable', type=str, help='Variable name (e.g. tas).')
    parser.add_argument('region', type=str, help='Region name (e.g. global).')
    parser.add_argument('season', type=str, help='Season name (e.g. DJFM).')
    parser.add_argument('forecast_range', type=str, help='Forecast range (e.g. 2-9).')
    parser.add_argument('start_year', type=str, help='Start year (e.g. 1960).')
    parser.add_argument('end_year', type=str, help='End year (e.g. 2014).')

    # Extract the arguments
    args = parser.parse_args()

    # Set up the model, variable, region, season, forecast_range, start_year and end_year
    model = args.model
    variable = args.variable
    region = args.region
    season = args.season
    forecast_range = args.forecast_range
    start_year = args.start_year
    end_year = args.end_year

    # Print a message to the screen
    print('Processing data for ' + model + ' ' + variable + ' ' + region + ' ' + season + ' ' + forecast_range + ' ' + start_year + ' ' + end_year + '...')

    # If the model is a number then convert it to a string
    if model.isdigit():
        # Print the model number
        print("Model number: ", model)

        # Extract the numbered element from the list of models
        model = dic.models[int(model) - 1]

        # Print the model name
        print("Model name: ", model)
        print("Processing data for " + model + " " + variable + " " + region + " " + season + " " + forecast_range + " " + start_year + " " + end_year + "...")
    else:
        # Print the model name
        print("Model name: ", model)
        print("Processing data for " + model + " " + variable + " " + region + " " + season + " " + forecast_range + " " + start_year + " " + end_year + "...")

    # Start the timer
    start_time = time.time()

    try:
        # Load the data
        historical_data = load_data(model, variable, region)

        # Print a message to the screen
        print('Loaded data for ' + model + ' ' + variable + ' ' + region + ' ' + season + ' ' + forecast_range + ' ' + start_year + ' ' + end_year + '...')

        # Print the type of the data
        print("Type of historical_data: ", type(historical_data))

        # Print the time taken
        print("Time taken to load data: ", time.time() - start_time, " seconds")

    except Exception as error:
        print(error)
        print("Unable to load data for " + model + " " + variable + " " + region + " " + season + " " + forecast_range + " " + start_year + " " + end_year + "...")
        sys.exit()

    # Select the season and years from the data
    try:
        # Select the season and years from the data
        historical_data_constrained = select_season_years(historical_data, season, start_year, end_year)

        # Print a message to the screen
        print('Selected the season: ' + season + ' and years: ' + start_year + '-' + end_year + '...')

        # Print the dimensions of the data
        # print("Dimensions of the data: ", historical_data.dims)

        # Print the time taken
        print("Time taken to select season and years: ", time.time() - start_time, " seconds")
    except Exception as error:
        print(error)
        print("Unable to select the season: " + season + " and years: " + start_year + "-" + end_year + "...")
        sys.exit()

    # Calculate and remove the model climatology for the selected season and years
    try:
        # Calculate and remove the model climatology for the selected season and years
        historical_data_constrained_anoms = calculate_remove_model_climatology(historical_data_constrained)

        # Print a message to the screen
        print('Calculated and removed the model climatology...')

        # Print the values of the data
        print("Values of the data: ", historical_data_constrained_anoms.values)

        # Print the time taken
        print("Time taken to calculate and remove the model climatology: ", time.time() - start_time, " seconds")
    except Exception as error:
        print(error)
        print("Unable to calculate and remove the model climatology in main...")
        sys.exit()

# Call the main function
# If we are running this script interactively
if __name__ == '__main__':
    main()

