# Data

Raw and processed datasets are intentionally excluded from GitHub.

## Expected inputs

The current pipeline expects data paths relative to the repository root:

- `grdc/` — GRDC station files
- `precipitation/` — gridded precipitation NetCDF data
- `precipitation_grid_last/` — NOAA station/grid precipitation inputs used by the station workflow
- `era5_merged/` — ERA5 daily files
- `g/2025_01.grib` — TIGGE/ECMWF GRIB input referenced by the current configuration
- `BasinATLAS_v10_lev08.csv` — HydroATLAS basin attributes and geometry WKT
- `HydroBasin.csv/` — HydroBASINS CSV files used for topology aggregation

Do not commit the raw datasets or generated pipeline outputs to the repository.
