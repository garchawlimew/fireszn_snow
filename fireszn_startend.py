import pandas as pd
import numpy as np
import xarray as xr
import glob
import itertools
from datetime import datetime
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#Finds first index of either 3 days snow off run (default) 
def index_start_truerun(series, min_length=3):
    start = 0
    runs = []
    for key, run in itertools.groupby(series):
        length = sum(1 for _ in run)
        if key == True and length >= min_length:
            runs.append((key, start, start + length - 1))
        start += length
    if len(runs) == 0:
        return np.nan
    result = max(runs, key=lambda x: x[2] - x[1])
    return result[1]

#converts date to ddd format
def doy_start_truerun(i, times):
    if not np.isnan(i) and 0 <= int(i) < len(times):
        startdoy = int(pd.Timestamp(times[int(i)]).strftime('%j'))
        return startdoy
    else:
        return np.nan

#applies the time zone correction to the local time 
def apply_tz_correction(ds, tz_correct):
    ds['time'] = pd.to_datetime(ds['time'].values)
    ds['time'] = ds['time'] + pd.to_timedelta(tz_correct, unit='h')
    return ds

# Load station data
dtype_dict = {'aes': str, 'wmo': str, 'id': str, 'prov': str, 'rep_date': str, 'quality': str, 'wmo-2': str, 'aes-2': str}
# Replace with the name of the CWFIS snow on ground file you are using
df_csv = pd.read_csv('CWFIS_FWIWX2020s_SOG.csv', dtype=dtype_dict)
df_csv['rep_date'] = pd.to_datetime(df_csv['rep_date'])
df_csv = df_csv[df_csv['rep_date'].dt.year.isin([2020, 2021, 2022, 2023])]

# Create new rows for x and y coordinates
df_csv['X'] = np.nan
df_csv['Y'] = np.nan

# cut off all stations north of 70 deg N
df_csv = df_csv[df_csv['lat'] <= 70]

#setting up a dictionary of each time zone correction
unique_stations = df_csv['wmo-2'].unique()
tz_corrections = {station_id: df_csv[df_csv['wmo-2'] == station_id]['tz_correct'].iloc[0] for station_id in unique_stations}

# Get X and Y for each station
x_list = []
y_list = []

# Define the polar stereographic projection (straight from the IMS file)
proj = Proj('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# Use proj to convert lat and lon to x and y, units of 1e6 metres
for station_id in unique_stations:
    station_data = df_csv[df_csv['wmo-2'] == station_id].copy()
    lon = station_data['lon'].iloc[0]
    lat = station_data['lat'].iloc[0]
    x, y = proj(lon, lat)
    x_list.append(x)
    y_list.append(y)
    df_csv.loc[df_csv['wmo-2'] == station_id, 'X'] = x
    df_csv.loc[df_csv['wmo-2'] == station_id, 'Y'] = y

# Load IMS data -- change path to your directory #
path_data = ""
# Export figures -- "" #
figure_path = ""
filenames_IMS = sorted(glob.glob(path_data + "ims*_4km_v1.3.nc"))

IMS = xr.open_mfdataset(filenames_IMS, combine='by_coords')

# Set the clipping box coordinates
# This roughly encapsulates Canada, change to your preferred dimensions 
xmin, xmax, ymin, ymax = -4.2e6, 2e6, -5.5e6, 0

# Writes the projection 
crs = IMS.projection.proj4
IMS.rio.write_crs(crs, inplace=True)

# Crop IMS data to the specified box
IMS_crop = IMS.rio.clip_box(xmin, ymin, xmax, ymax)

# Subsets the MODIS data to YOUR years of interest 
IMS_timeseries = IMS_crop.sel(time=IMS_crop.time.dt.year.isin([2020, 2021, 2022, 2023]))


### Initially load up until here -------------------------------------------------------------------------------

######################################################################################################
# Returns the station IDs of stations which have more than 100 null values in their whole timeseries
# and grid cells which have 1461 days, which is 4 years of no snow
######################################################################################################


table_nulls = []


for station_id, x, y in zip(unique_stations, x_list, y_list):

    x = x_list[unique_stations.tolist().index(station_id)]
    y = y_list[unique_stations.tolist().index(station_id)]

    
    # Filter station data for the specific year (2020 to 2023) and sort by reported date
    station_data_all_years = df_csv[(df_csv['wmo-2'] == station_id) & (df_csv['rep_date'].dt.year.isin([2020,2021,2022,2023]))].copy()
    station_data_all_years = station_data_all_years.sort_values(by='rep_date')

    #Putting in IMS data to be read
    selected_IMS = IMS_timeseries.sel(x=x, y=y, method='nearest')
    tz_correct = tz_corrections[station_id]
    selected_IMS = apply_tz_correction(selected_IMS, tz_correct)
    selected_IMS['IMS_Surface_Values'] = (selected_IMS['IMS_Surface_Values'] == 4) | (selected_IMS['IMS_Surface_Values'] == 3)
    
    # Sort IMS dataframe
    ims_df = selected_IMS.to_dataframe().reset_index()
    ims_df = ims_df.sort_values(by='time')

    # Count number of null values at each station
    station_nulls = station_data_all_years['sog'].isnull().sum()
    
    #Counts number of snow off days at each gridcell
    snow_off_count = (~ims_df['IMS_Surface_Values']).sum()
    

    if (snow_off_count == 1461) or (station_nulls>100):
        table_df = pd.DataFrame({'station_id':station_id,'snow_off_count': snow_off_count,'station_nulls': station_nulls}, index=[0]) 
        table_nulls.append(table_df)

    
# Concatenate the nulls data into a dataframe
final_null = pd.concat(table_nulls, ignore_index=True)
final_null.to_csv('null_count.csv',index=False)
print("Data has been successfully saved to 'null_count.csv'.")

#################################################################################################################
# Making a map which shows stations with too many nulls, too many snow off values, and stations that will be used
#################################################################################################################

# Read CSV and fix data types for merging
final_null = pd.read_csv('null_count.csv')
final_null['station_id'] = final_null['station_id'].astype(str)

IMS_bad_stations = final_null.groupby('station_id')['snow_off_count'].sum().reset_index()
csv_bad_stations = final_null.groupby('station_id')['station_nulls'].sum().reset_index()

# Merge null results into one DataFrame
nulls_results = pd.merge(IMS_bad_stations, csv_bad_stations, on='station_id')

# Merge with the main DataFrame to get coordinates
nulls_results = pd.merge(nulls_results, df_csv[['wmo-2', 'lon', 'lat']], left_on='station_id', right_on='wmo-2', how='left')

# Filter out good stations by excluding the bad stations
good_stations = df_csv[~df_csv['wmo-2'].isin(nulls_results['station_id'])]

# Set up the map
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add geographical features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='dotted')
ax.add_feature(cfeature.LAKES)
provinces_50m = cfeature.NaturalEarthFeature('cultural',
                                             'admin_1_states_provinces_lines',
                                             '50m',
                                             facecolor='none')
ax.add_feature(provinces_50m)

# Plot bad stations (excluded ones)
sc_badstations = ax.scatter(nulls_results['lon'], nulls_results['lat'], color='red', s=10, label='Excluded Stations')

# Plot good stations (included ones)
sc_goodstations = ax.scatter(good_stations['lon'], good_stations['lat'], color='blue', s=10, label='Valid Stations')

# Add a title with increased size and bold font
ax.set_title('Valid and Invalid Station Data', fontsize=16, fontweight='bold')

# Add a legend
ax.legend(loc='upper right')

# Show the plot
plt.show()
plt.close(fig)


########################################################################################
# from this point on load this code in order to omit all stations included in the null_count list

# Update unique_stations to include only good stations
unique_stations = good_stations['wmo-2'].unique()

##########################################################################################
# Displaying snow on off timeseries of one specific station for both IMS and ECCC Stations
##########################################################################################


years = np.arange(2020, 2023 + 1)
def plot_station_and_ims_timeseries_by_year(station_id):
    df_csv['sog'] = df_csv['sog'] > 0
    # Find the station's x, y coordinates
    x = x_list[unique_stations.tolist().index(station_id)]
    y = y_list[unique_stations.tolist().index(station_id)]

    # Retrieve the station data
    station_data_all_years = df_csv[df_csv['wmo-2'] == station_id].copy()
    station_data_all_years = station_data_all_years.sort_values(by='rep_date')

    # Retrieve the IMS data for the corresponding grid cell
    selected_IMS = IMS_timeseries.sel(x=x, y=y, method='nearest')
    tz_correct = tz_corrections[station_id]
    selected_IMS = apply_tz_correction(selected_IMS, tz_correct)
    selected_IMS['IMS_Surface_Values'] = (selected_IMS['IMS_Surface_Values'] == 4) | (selected_IMS['IMS_Surface_Values'] == 3)
    
    # Sort IMS dataframe
    ims_df = selected_IMS.to_dataframe().reset_index()
    ims_df = ims_df.sort_values(by='time')

    # Generate a figure for each year
    for year in years:
        plt.figure(figsize=(12, 6))
        
        # Filter data for the specific year
        station_data_year = station_data_all_years[(station_data_all_years['rep_date'].dt.year == year)]
        station_data_sorted = station_data_year.sort_values(by='rep_date')

        ims_yearly = ims_df[ims_df['time'].dt.year == year]
 
        # Plot station snow on/off time series
        plt.plot(station_data_sorted['rep_date'], station_data_sorted['sog'].astype(int), label=f'Station Year {year}')

        # Plot IMS snow on/off time series
        plt.plot(ims_yearly['time'], ims_yearly['IMS_Surface_Values'].astype(int), label=f'IMS Year {year}', linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Snow On Ground (1 for snow, 0 for no snow)')
        plt.title(f'Snow On/Off Time Series for Station {station_id} - Year {year}')
        plt.legend()
        plt.show(block=False)
        plt.savefig(figure_path+"snow_onoff_" +str(year) + "for" +str(station_id)+ ".png", dpi=600, bbox_inches='tight')


station_id = '71020'  # Replace with the station ID you want to investigate
plot_station_and_ims_timeseries_by_year(station_id)

#####################################
# Checking timeseries of Medicine Hat
#####################################
# Checking to see if for example Medicine hat actually reads snow or not 
# Coordinates near Medicine Hat, AB
x_mh = -4108303.0557654845
y_mh = -1853502.6891557644

# Find the nearest grid cell in the x and y dimensions
x_idx = np.abs(IMS_timeseries.x - x_mh).argmin()
y_idx = np.abs(IMS_timeseries.y - y_mh).argmin()

# Plot the IMS_Surface_Values for the specific grid cell over time
plt.figure(figsize=(10, 6))
IMS_timeseries.IMS_Surface_Values[:, y_idx, x_idx].plot()
plt.title(f'IMS Surface Values Time Series near Medicine Hat, AB\n(x={x_mh}, y={y_mh})')
plt.xlabel('Time')
plt.ylabel('IMS Surface Values')
plt.grid(True)
plt.show(block=False)


##############################
# Merged snow on/off for loop
# returns csv with snow on off
# boolean values for each stn
##############################

# Initialize an empty list to collect data for all stations
data = []

for station_id, x, y in zip(unique_stations, x_list, y_list):
    df_csv['sog'] = df_csv['sog'] > 0
    tz_correct = tz_corrections[station_id]
    
    # Select nearest points in IMS
    selected_IMS = IMS_timeseries.sel(x=x, y=y, method='nearest')
    selected_IMS = apply_tz_correction(selected_IMS, tz_correct) #overwrites the dataset with the corrected times
    
    # Set snow on land and sea ice to true, false otherwise
    selected_IMS['IMS_Surface_Values'] = (selected_IMS['IMS_Surface_Values'] == 4) | (selected_IMS['IMS_Surface_Values'] == 3)
    
    # Convert IMS data to DataFrame and select necessary columns
    ims_df = selected_IMS.to_dataframe().reset_index()
    ims_df = ims_df[['time', 'IMS_Surface_Values']]
    
    # Ensure ims_df is sorted by time 
    ims_df = ims_df.sort_values(by='time')

    ########## Moving on to station data ##########

    # Filter the CSV data for the current station 
    station_csv = df_csv[df_csv['wmo-2'] == station_id].copy()
    # Sort appropriately and select necessary columns 
    station_csv = station_csv[['rep_date', 'sog', 'wmo-2', 'lon', 'lat', 'X', 'Y']].sort_values(by='rep_date')
    
    # Use asof to merge the times from the ims data to the station data
    merged_data = pd.merge_asof(ims_df, station_csv, left_on='time', right_on='rep_date', direction='nearest')
    
    # Add additional columns
    merged_data['station_id'] = station_id
    merged_data['year'] = merged_data['time'].dt.year
    merged_data['snow on - off'] = (merged_data['sog'] == True) & (merged_data['IMS_Surface_Values'] == True)
    merged_data['time'] = merged_data['time'].dt.strftime('%Y-%m-%d')
    # Select final columns of interest
    merged_data = merged_data[['station_id', 'lon', 'lat', 'X', 'Y', 'time', 'year', 'sog', 'IMS_Surface_Values', 'snow on - off']]
    
    data.append(merged_data)

# Concatenate all data into a single DataFrame
final_df = pd.concat(data, ignore_index=True)

# Save the final DataFrame to a CSV file
final_df.to_csv('merged_data.csv', index=False)

print("Data has been successfully merged and saved to 'merged_data.csv'.")


######################################################################################################################
# Second CSV to be created, containing day of year (doy) data 
# Here we also would like to consider the doy from the station data as well as the IMS data
# We can compare their absolute differences and add a start doy/end doy error column, which is the difference between
# station and IMS data for each location's initialization and termination date respectively
######################################################################################################################

data_doy = []

years = np.arange(2020, 2023 + 1)

for station_id, x, y in zip(unique_stations, x_list, y_list):
    
    for year in years:
        df_csv['sog'] = df_csv['sog'] > 0
        # Filter station data for the specific year (2020 to 2023) and sort by reported date
        station_data_year = df_csv[(df_csv['wmo-2'] == station_id) & (df_csv['rep_date'].dt.year == year)]
        station_data_sorted = station_data_year.sort_values(by='rep_date')

        # Filter for snow on calculation (Sept Oct Nov Dec)
        snow_on_stndata = station_data_sorted[station_data_sorted['rep_date'].dt.month.isin([9, 10, 11, 12])]

        # Calculate snow on/off DOY for station data
        snow_on_station = index_start_truerun(snow_on_stndata['sog'], min_length=7)
        snow_off_station = index_start_truerun(~station_data_sorted['sog'], min_length=3)
        snow_on_doy = doy_start_truerun(snow_on_station, snow_on_stndata['rep_date'].values)
        snow_off_doy = doy_start_truerun(snow_off_station, station_data_sorted['rep_date'].values)
    #-----------------------------------------------------------------------------------------------------------------
        # Process IMS data
        tz_correct = tz_corrections[station_id]
        IMS_yearly = IMS_crop.sel(time=IMS_crop.time.dt.year == year)

        selected_IMS = IMS_yearly.sel(x=x, y=y, method='nearest')
        selected_IMS = apply_tz_correction(selected_IMS, tz_correct)  # Apply timezone correction
        selected_IMS['IMS_Surface_Values'] = (selected_IMS['IMS_Surface_Values'] == 4) | (selected_IMS['IMS_Surface_Values'] == 3)

        # Convert the gridded IMS data to DataFrame and sort by time
        ims_df = selected_IMS.to_dataframe().reset_index()
        ims_df = ims_df[['time', 'IMS_Surface_Values']].sort_values(by='time')

        # Filter for snow on calculation (Sept Oct Nov Dec)
        snow_on_IMSdata = ims_df[ims_df['time'].dt.month.isin([9, 10, 11, 12])]

        # Calculate snow on/off DOY for IMS data
        snowon_IMS = index_start_truerun(snow_on_IMSdata['IMS_Surface_Values'], min_length=7)
        snowoff_IMS = index_start_truerun(~ims_df['IMS_Surface_Values'], min_length=3)
        snowon_doy_IMS = doy_start_truerun(snowon_IMS, snow_on_IMSdata['time'].values)
        snowoff_doy_IMS = doy_start_truerun(snowoff_IMS, ims_df['time'].values)

        # Create a DataFrame containing DOY info for the current year and station
        merged_startend = pd.DataFrame({
            'station_id': station_id,
            'X': x,
            'Y': y,
            'year': year,
            'Snow off date (from IMS)': snowoff_doy_IMS,  # IMS snow off DOY
            'Snow on date (from IMS)': snowon_doy_IMS,  # IMS snow on DOY
            'Snow off date (station data)': snow_off_doy,  # Station snow off DOY
            'Snow on date (station data)': snow_on_doy,  # Station snow on DOY
            'Start DOY error': abs(snow_on_doy - snowon_doy_IMS),
            'End DOY error': abs(snow_off_doy - snowoff_doy_IMS)
        }, index=[0])

        # Append the merged data for the current year and station
        data_doy.append(merged_startend)

# Concatenate all DOY data into a single DataFrame
final_doy = pd.concat(data_doy, ignore_index=True)  
final_doy.to_csv('doy_data.csv', index=False)
print("Data has been successfully merged and saved to 'doy_data.csv'.")

######################
# Statistical Box Plot
# for snow on/off doy
# values and abs error
######################

# Load the data
doy_data = pd.read_csv("doy_data.csv")
snow_df = pd.DataFrame(doy_data)
# Filter only the "Snow off date (from IMS)" column
snow_df_filtered = snow_df[snow_df["Snow off date (from IMS)"] <= 300]
# Remove any rows with NaN values in the filtered DataFrame
snow_df_filtered = snow_df_filtered.dropna(subset=["Snow off date (from IMS)", "Snow on date (from IMS)", "Start DOY error", "End DOY error"])
# Define the labels and data for the boxplot
labels = ['Snow Off Dates', 'Snow On Dates', 'Start DOY Error', 'End DOY Error']
fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size for better readability
ax.set_ylabel('days')
# Plot the boxplots for the filtered data
snow_plots = [
    snow_df_filtered["Snow off date (from IMS)"],
    snow_df_filtered["Snow on date (from IMS)"],
    snow_df_filtered["Start DOY error"],
    snow_df_filtered["End DOY error"]

]
colors = ['blue', 'white', 'orange', 'red']
snow_box = ax.boxplot(snow_plots, labels=labels, patch_artist=True, boxprops=dict(facecolor='white'))
# Set the xlim to give space for the legend box
ax.set_xlim(0.5, len(labels) + 1.5)
# Create the legend box content
legend_labels = []
for i, (data, color) in enumerate(zip(snow_plots, colors), start=1):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    # Prepare the text for the legend
    text = (
        f"{labels[i-1]}:\n"
        f"  Median: {median:.0f}\n"
        f"  Q1: {q1:.0f}\n"
        f"  Q3: {q3:.0f}\n"
        f"  IQR: {iqr:.0f}"
    )
    legend_labels.append(text)
    # Color the boxplot elements to match the legend
    for patch in snow_box['boxes'][i-1:i]:
        patch.set(facecolor=color)

# Create the legend box
legend_box = '\n\n'.join(legend_labels)
plt.gcf().text(0.85, 0.5, legend_box, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
# Adjust layout
plt.subplots_adjust(left=0.25)
# Display the plot
plt.show()

#############################################################################
# Plot sample figures of snow on and snow off for certain transitory dates
#############################################################################

# Ensure that the rep_date column only contains the date component
good_stations['rep_date'] = good_stations['rep_date'].dt.floor('D')

# Define the dates you want to filter by
dates_to_filter = pd.to_datetime(['2020-03-31', '2020-04-30', '2020-09-30', '2020-11-30'])

# Define a function to plot the data on a map for a given date using ECCC station data
def plot_map_for_date(date, df):
    # Filter the DataFrame for the specific date
    df_date = df[df['rep_date'] == date]
    
    # Set up the map with Cartopy
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-90, 90))
  
    # Add geographical features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='dotted')
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(provinces_50m)
 
    # Plot the station data for snow on (sog == 1)
    scatter_on = ax.scatter(
        df_date[df_date['sog'] == 1]['lon'], df_date[df_date['sog'] == 1]['lat'], 
        color='blue', s=10, transform=ccrs.PlateCarree(), label='Snow On')
    # Plot the station data for snow off (sog == 0)
    scatter_off = ax.scatter(
        df_date[df_date['sog'] == 0]['lon'], df_date[df_date['sog'] == 0]['lat'], 
        color='red', s=10, transform=ccrs.PlateCarree(), label='Snow Off')
    # Add a title with increased size and bold font
    ax.set_title(f'Station Data for {date.date()}', fontsize=16, fontweight='bold')
    # Add a legend
    ax.legend(loc='upper right')
    # Show the plot
    plt.show()
    plt.close(fig)

# Plot a map for each of the specified dates only using good stations
for date in dates_to_filter:
    plot_map_for_date(date, good_stations)

#-------------------------------------------------------------------------------------------------
# Now creating the same types of maps for the IMS data

dates_to_filter = pd.to_datetime(['2020-03-31', '2020-04-30', '2020-09-30', '2020-11-30'])
# Update unique_stations to include only good stations
unique_stations = good_stations['wmo-2'].unique()

all_ims_data = []

for station_id, x, y in zip(unique_stations, x_list, y_list):
    x = x_list[unique_stations.tolist().index(station_id)]
    y = y_list[unique_stations.tolist().index(station_id)]

    # Retrieve IMS data for the specific station
    selected_IMS = IMS_timeseries.sel(x=x, y=y, method='nearest')
    tz_correct = tz_corrections[station_id]
    selected_IMS = apply_tz_correction(selected_IMS, tz_correct)
    selected_IMS['IMS_Surface_Values'] = (selected_IMS['IMS_Surface_Values'] == 4) | (selected_IMS['IMS_Surface_Values'] == 3)
    
    # Convert IMS data to DataFrame and sort by time
    ims_df = selected_IMS.to_dataframe().reset_index()
    ims_df = ims_df.sort_values(by='time')
    
    # Append the DataFrame for this station to the list
    all_ims_data.append(ims_df)

# Concatenate all DataFrames into a single DataFrame
final_ims_df = pd.concat(all_ims_data, ignore_index=True)
final_ims_df['time'] = final_ims_df['time'].dt.floor('D')

final_ims_df.to_csv('ims_snowonoff.csv', index=False)

proj_stere = Proj('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')

# Apply the inverse transformation to convert x, y to lon, lat
final_ims_df['lon'], final_ims_df['lat'] = proj_stere(final_ims_df['x'].values, final_ims_df['y'].values, inverse=True)

provinces_50m = cfeature.NaturalEarthFeature('cultural',
                                             'admin_1_states_provinces_lines',
                                             '50m',
                                             facecolor='none')

# Define a function to plot the data on a map for a given date
def plot_map_for_ims(date, df):
    # Filter the DataFrame for the specific date
    df_date = df[df['time'] == date]
    
    # Set up the map with Cartopy
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-90, 90))
  
    # Add geographical features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='dotted')
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(provinces_50m)
 
    # Plot the station data for snow on (IMS_Surface_Values == 1)
    scatter_on = ax.scatter(
        df_date[df_date['IMS_Surface_Values'] == 1]['lon'], df_date[df_date['IMS_Surface_Values'] == 1]['lat'], 
        color='blue', s=10, transform=ccrs.PlateCarree(), label='Snow On')
    # Plot the station data for snow off (IMS_Surface_Values == 0)
    scatter_off = ax.scatter(
        df_date[df_date['IMS_Surface_Values'] == 0]['lon'], df_date[df_date['IMS_Surface_Values'] == 0]['lat'], 
        color='red', s=10, transform=ccrs.PlateCarree(), label='Snow Off')
    # Add a title with increased size and bold font
    ax.set_title(f'IMS Data for {date.date()}', fontsize=16, fontweight='bold')
    # Add a legend
    ax.legend(loc='upper right')
    # Show the plot
    plt.show()
    plt.close(fig)

# Plot a map for each of the specified dates only using good stations
for date in dates_to_filter:
    plot_map_for_ims(date, final_ims_df)


