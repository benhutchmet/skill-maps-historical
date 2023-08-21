# Processing historical data #

For creating uninitialized skill maps using the 'historical' CMIP6 data.

Batch processing will be used to prepare the data.

Then the correlations between the observations and the processed data will be performed
and plotted in a jupyter notebook (plotting.ipynb).

# Workflow #

1. Run regrid and merge time axis script (for each model)

2. Run process data script for each model:

    * Select season and year range (start year - end year)
    * Calculate anomalies (remove model mean state from each model member)
    * If season crosses year (e.g. DJFM) then shift back data and take annual average, otherwise just take annual average.
    * Calculate the running mean for the given forecast range (if not 2-2)

! Watch out for time alignment - particularly in 2-2 DJFM cases !