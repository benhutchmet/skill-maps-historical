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
def calculate_remove_model_climatology(historical_data_constrained, variable):
    """
    For the selected season and years, calculate and remove the model climatology from each member.
    """

    # Concatenate the data along the variant_label dimension
    historical_data_constrained_ensemble = xr.concat(historical_data_constrained.values(), dim='variant_label')

    # Print the dimensions of the data
    print("Dimensions of the data after psl selected: ", historical_data_constrained_ensemble)

    # Print a message to the screen
    print('Calculating and removing the model climatology...')

    # Check that the variable is valid
    # if the variable is contained within dic.variables, then it is valid
    if variable in dic.variables:
        # Print a message to the screen
        print("Variable is valid")
    else:
        # Print a message to the screen
        print("Variable is not valid, exiting script")
        sys.exit()

    # Print the variable
    print("Variable: ", variable)

    # Calculate the model climatology
    try:

        # Take the mean over all of the members
        members_ensemble_mean = historical_data_constrained_ensemble.mean(dim='variant_label')

        # Print the dimensions of the members_ensemble_mean
        print("Dimensions of the members_ensemble_mean: ", members_ensemble_mean.dims)

        # Take the time mean of this ensemble mean
        model_climatology = members_ensemble_mean.mean(dim='time')

        # print the values of the model_climatology
        print("Psl Values of the model_climatology: ", model_climatology[variable].values)

        # Print the shape of the psl values of the model_climatology
        print("Shape of the psl values of the model_climatology: ", model_climatology[variable].values.shape)

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
            historical_data_constrained[member][variable].values = historical_data_constrained[member][variable].values - model_climatology[variable].values
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

# Now define a function to take the seasonal mean by shifting the time axis back (in the case of DJFM) 
# and then taking the annual mean
# Or just take the annual mean if the season does not cross the year boundary
def annual_mean_anoms(historical_data_constrained_anoms, season):
    """
    If the season crosses the year boundary then take the seasonal mean by shifting the time axis back.
    Then take the annual mean.
    
    If the season does not cross the year boundary then just take the annual mean.
    """

    # Print a message to the screen
    print('Calculating the annual mean anomalies...')

    # if the season crosses the year boundary
    # i.e. contains December and January ('DJ')
    if 'D' in season and 'J' in season:

        # Print a message to the screen
        print('Season crosses the year boundary, so shifting the time axis back')

        # Set up the season from the season timeshift dictionary
        season_index = [d['season'] for d in dic.season_timeshift].index(season)
        season = dic.season_timeshift[season_index]

        # Print the season
        print("season: ", season)

        # Shift the time axis back
        print("Shifting the time axis back by: ", season['timeshift'])

        # For brevity
        data = historical_data_constrained_anoms

        try:
            # Loop over the members
            for member in data:
                # Shift the time axis back
                data[member] = data[member].shift(time=season['timeshift'])

                # Calculate the annual mean
                data[member] = data[member].resample(time='Y').mean(dim='time')

        except Exception as error:
            print(error)
            print("Unable to shift the time axis back and calculate the annual mean for the season: ", season)
            return None

    else:
        # Print a message to the screen
        print('Season does not cross the year boundary')

        # Print that we are just taking the annual mean
        print('Just taking the annual mean')

        try:
            # Loop over the members
            data = historical_data_constrained_anoms
            for member in data:
                # Calculate the annual mean
                data[member] = data[member].resample(time='Y').mean(dim='time')

        except Exception as error:
            print(error)
            print("Unable to calculate the annual mean for the season: ", season)
            return None
        
    # Print a message to the screen
    print('Calculated the annual mean anomalies for the season: ', season)

    # Return the data
    return data

# Now set up the function for caluclating the running mean
def calculate_running_mean(historical_data_constrained_anoms_annual_mean, forecast_range):
    """
    Calculate the running mean for each member for the historical data which has
    been constrained to the specified season and years, the model climatology has
    been removed and the annual mean anomalies have been calculated (shifted back if necessary).
    """

    # Print a message to the screen
    print('Calculating the running mean for the forecast range: ' + forecast_range)

    # if the forecast range is 2-2
    if forecast_range == '2-2':
        print("Forecast range is 2-2, so running mean does not need to be calculated")
        return historical_data_constrained_anoms_annual_mean
    else:
        print("Forecast range is not 2-2, so running mean needs to be calculated")

        # Extract the start and end years from the forecast_range
        start_year = forecast_range.split('-')[0]
        end_year = forecast_range.split('-')[1]

        # Print the start and end years
        print("start_year: ", start_year)
        print("end_year: ", end_year)

        # Print the forecast range
        print("forecast_range: ", start_year + "-" + end_year)

        # Calculate the running mean value
        running_mean_value = int(end_year) - int(start_year) + 1

        # Print the running mean value
        print("running_mean_value: ", running_mean_value)

        # Calculate the running mean
        try:
            # Loop over the members
            data = historical_data_constrained_anoms_annual_mean
            for member in data:
                # Calculate the running mean
                data[member] = data[member].rolling(time=running_mean_value, center=True).mean()

                # Get rid of the NaNs
                # Only for years containing only NaNs
                data[member] = data[member].dropna(dim='time', how='all')

        except Exception as error:
            print(error)
            print("Unable to calculate the running mean for the forecast range: ", forecast_range)
            return None
        
        # Print a message to the screen
        print('Calculated the running mean for the forecast range: ', forecast_range)

        # Return the data
        return data

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
        historical_data_constrained_anoms = calculate_remove_model_climatology(historical_data_constrained, variable)

        # Print a message to the screen
        print('Calculated and removed the model climatology...')

        # # Print the values of the data
        # print("Values of the data: ", historical_data_constrained_anoms.values)

        # Print the time taken
        print("Time taken to calculate and remove the model climatology: ", time.time() - start_time, " seconds")
    except Exception as error:
        print(error)
        print("Unable to calculate and remove the model climatology in main...")
        sys.exit()

    # Calculate the annual mean anomalies
    try:
        # Calculate the annual mean anomalies
        historical_data_constrained_anoms_annual_mean = annual_mean_anoms(historical_data_constrained_anoms, season)

        # Print a message to the screen
        print('Calculated the annual mean anomalies')

        # # Print the data
        # print("Data: ", historical_data_constrained_anoms_annual_mean)

        # Print the time taken
        print("Time taken to calculate the annual mean anomalies: ", time.time() - start_time, " seconds")
    except Exception as error:
        print(error)
        print("Unable to calculate the annual mean anomalies in main")
        sys.exit()

    # Calculate the running mean
    try:
        # Calculate the running mean
        historical_data_constrained_anoms_annual_mean_rm= calculate_running_mean(historical_data_constrained_anoms_annual_mean, forecast_range)

        # Print a message to the screen
        print('Calculated the running mean')

        # # Print the data
        print("Data: ", historical_data_constrained_anoms_annual_mean_rm)

        # Loop over the members
        for member in historical_data_constrained_anoms_annual_mean_rm:
            # Print the member
            print("Member: ", member)

            # Print the psl values of the member
            print("Psl values of the member: ", historical_data_constrained_anoms_annual_mean_rm[member].psl.values)

        # Print the time taken
        print("Time taken to calculate the running mean: ", time.time() - start_time, " seconds")
    except Exception as error:
        print(error)
        print("Unable to calculate the running mean in main")
        sys.exit()

    # TODO: Save the data
    # Print a message to the screen
    print('Data processing complete, saving the data for ' + model + ' ' + variable + ' ' + region + ' ' + season + ' ' + forecast_range + ' ' + start_year + ' ' + end_year)

    # Set up the path for saving the data
    # /home/users/benhutch/skill-maps-processed-data/psl/EC-Earth3/global/years_2-9/DJFM/outputs/mergetime
    save_path = dic.home_dir + 'skill-maps-processed-data' + '/' + 'historical' + '/' + variable + '/' + model + '/' + region + '/' + 'years_' + forecast_range + '/' + season + '/' + 'outputs' + '/' + 'processed'

    # Now loop over the members and save the data
    # for brevity
    data = historical_data_constrained_anoms_annual_mean_rm
    for member in data:
        # Print that we are saving the data
        print("Saving the data for member: ", member, "and model:", model)

        # if variable is tos, then use 'Omon' instead of 'Amon'
        if variable == 'tos':
            file_name = variable + '_' + 'Omon' + '_' + model + '_' + 'historical' + '_' + member + '_' + start_year + '-' + end_year + '_' + region + '_processed.nc'
        else:
            # Set up the file name
            # psl_Amon_HadGEM3-GC31-MM_historical_r1i1p1f3_g?_1850-2014.nc_global_regrid.nc
            file_name = variable + '_' + 'Amon' + '_' + model + '_' + 'historical' + '_' + member + '_' + start_year + '-' + end_year + '_' + region + '_processed.nc'

        # Set up the path to the file
        file_path = save_path + '/' + file_name

        # Print the file path
        print("file_path: ", file_path)

        # Save the data
        try:
            # Save the data
            data[member].to_netcdf(file_path)

            # Print a message to the screen
            print("Saved the data for member: ", member, "and model:", model)
        except Exception as error:
            print(error)
            print("Unable to save the data for member: ", member, "and model:", model)
            sys.exit()

    # Print a message to the screen
    print('Saved the data for ' + model + ' ' + variable + ' ' + region + ' ' + season + ' ' + forecast_range + ' ' + start_year + ' ' + end_year)

    # Print the time taken
    print("Time taken to save the data: ", time.time() - start_time, " seconds")

    # Print a message to the screen
    print('Data processing complete for ' + model + ' ' + variable + ' ' + region + ' ' + season + ' ' + forecast_range + ' ' + start_year + ' ' + end_year)

    # Print the time taken
    print("Time taken to process the data: ", time.time() - start_time, " seconds")

# Call the main function
# If we are running this script interactively
if __name__ == '__main__':
    main()

