# fireszn_snow
Loading and preprocessing ECCC station data. Retrieving and processing IMS data. Identifying and excluding stations with inadequate data. Visualizing snow on/off timeseries for selected stations. Plotting maps that show snow on/off coverage for selected dates.

---

# Snow On/Off Analysis with IMS and ECCC Data

This project analyzes snow on/off data using Environment and Climate Change Canada (ECCC) station data and NOAA's Interactive Multisensor Snow and Ice Mapping System (IMS) data. The goal for this project is to visualize and compare the snow coverage between ECCC stations and IMS grid data over the years 2020 to 2023.

## Project Overview

This project involves:
- Loading and preprocessing ECCC station data.
- Retrieving and processing IMS data.
- Identifying and excluding stations with inadequate data.
- Visualizing snow on/off timeseries for selected stations.
- Plotting maps that show snow on/off coverage for selected dates.

## Installation

To use this project, you need to have Python 3.x installed. The following Python packages are also required:

pip install pandas numpy xarray glob2 pyproj matplotlib cartopy rioxarray

Ensure that you have the necessary data files:
- **ECCC Data**: `CWFIS_FWIWX2020s_SOG.csv`, contact Justin.Beckers@nrcan-rncan.gc.ca
- **IMS Data**: IMS netCDF files, e.g., `ims*_4km_v1.3.nc` from https://nsidc.org/data/g02156/versions/1

## Usage

1. **Clone the Repository:**

   git clone https://github.com/yourusername/snow-on-off-analysis.git
   cd snow-on-off-analysis


2. **Place Data Files:**

   - Place the `CWFIS_FWIWX2020s_SOG.csv` file in the project directory.
   - Place the IMS netCDF files in a directory and update the `path_data` variable in the script accordingly.

3. **Run the Script:**

   Execute the script in your Python environment:

 
   python snow_on_off_analysis.py


   The script processes the data, performs the analysis, and generates visualizations.

## Data Processing

The script performs several key data processing tasks:

- **Load and Preprocess Data:** ECCC station data is loaded, and IMS data is cropped to the area of interest. Missing data is identified and excluded from further analysis.
  
- **Time Zone Correction:** The script adjusts the time for each station to account for time zone differences.

- **Coordinate Transformation:** The script converts the x and y coordinates of the IMS data to longitude and latitude for mapping.

- **Data Merging:** The script merges ECCC and IMS data to facilitate comparison.

## Visualization

The project includes the following visualizations:

- **Timeseries Plots:** Snow on/off timeseries for selected stations are plotted for each year.
- **Map Plots:** Maps showing snow on/off coverage for selected dates are generated.

Sample output figures are saved in the `IMS_figures` directory.

## File Descriptions

- `snow_on_off_analysis.py`: The main script that performs data processing, analysis, and visualization.
- `CWFIS_FWIWX2020s_SOG.csv`: ECCC station data file (not included in the repo).
- `ims*_4km_v1.3.nc`: IMS netCDF files (not included in the repo).
- `null_count.csv`: Contains information about stations with inadequate data.
- `merged_data.csv`: Merged data file for further analysis.
- `doy_data.csv`: Data file containing day of year information for snow on/off dates.
- `IMS_figures/`: Directory where output figures are saved.

## Contributing


## License

This project is licensed under the MIT License. 

---
