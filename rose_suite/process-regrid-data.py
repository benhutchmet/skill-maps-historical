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

    # Load the data for all members using xr.open_mfdataset()
    # in parallel mode
    historical_data = xr.open_mfdataset(regrid_files, combine='nested', concat_dim='variant_label', parallel=True, chunks={'time': 100, 'lat': 45, 'lon': 45})

    # Print a message to the screen
    print('Loaded data for ' + model + ' ' + variable + ' ' + region + '...')

    # Print the type of the data
    print("Type of historical_data: ", type(historical_data))

    # Print the shape of the data
    print("Shape of historical_data: ", historical_data.shape)

    # Print the data
    print("historical_data: ", historical_data)

    return historical_data


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

# Call the main function
# If we are running this script interactively
if __name__ == '__main__':
    main()

