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