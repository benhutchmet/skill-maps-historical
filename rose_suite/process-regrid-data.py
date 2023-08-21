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