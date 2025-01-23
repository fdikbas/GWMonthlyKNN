# GWMonthlyKNN.py
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import os

# Load the data
data = pd.read_csv('gwl-monthly.grouped.csv')

# Convert MSMT_DATE to datetime
data['MSMT_DATE'] = pd.to_datetime(data['MSMT_DATE'])

# Exclude stations with no WSE data
stations_with_data = data.groupby('STATION')['WSE'].transform('count') > 0
data = data[stations_with_data]

# Remove lines with no WSE values before the first observation and after the last observation for each station
def filter_observation_range(group):
    first_obs_index = group['WSE'].first_valid_index()
    last_obs_index = group['WSE'].last_valid_index()
    return group.loc[first_obs_index:last_obs_index]

data = data.groupby('STATION').apply(filter_observation_range).reset_index(drop=True)

# Sort the data by station code and then by date within each station
data = data.sort_values(by=['STATION', 'MSMT_DATE'])

# Filter out groups with a single well where there is no missing value
group_sizes = data['GROUP'].value_counts()
single_well_groups = group_sizes[group_sizes == 1].index
for group in single_well_groups:
    if data[data['GROUP'] == group]['WSE'].isnull().sum() == 0:
        data = data[data['GROUP'] != group]

# Extract month and year from the date for seasonality
data['MONTH'] = data['MSMT_DATE'].dt.month
data['YEAR'] = data['MSMT_DATE'].dt.year

# Prepare directories for output
output_dir = 'time_series_graphs_KNN'
infilled_data_dir = 'infilled_data_KNN'
estimated_data_dir = 'estimated_data_KNN'
group_output_dir = 'group_time_series_graphs_KNN'

for directory in [output_dir, infilled_data_dir, estimated_data_dir, group_output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to impute missing values using KNN with enhanced features and time decay
def knn_group_impute(group_data, n_neighbors=5):
    # Create lag features and rolling statistics
    group_data['LAG_1'] = group_data.groupby('STATION')['WSE'].shift(1)
    group_data['LAG_2'] = group_data.groupby('STATION')['WSE'].shift(2)
    group_data['ROLLING_MEAN'] = group_data.groupby('STATION')['WSE'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    group_data['ROLLING_STD'] = group_data.groupby('STATION')['WSE'].transform(lambda x: x.rolling(window=3, min_periods=1).std())

    # Pivot the data to have stations as columns and include month for seasonality
    pivot_data = group_data.pivot(index=['MONTH', 'YEAR'], columns='STATION', values=['WSE', 'LAG_1', 'LAG_2', 'ROLLING_MEAN', 'ROLLING_STD'])
    pivot_data.columns = ['_'.join(col).strip() for col in pivot_data.columns.values]
    pivot_data = pivot_data.reset_index()

    # Use KNN imputer with distance weighting
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed_data = imputer.fit_transform(pivot_data)

    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=pivot_data.columns).set_index(['MONTH', 'YEAR'])

    # Extract the imputed WSE values
    imputed_long = imputed_df.filter(like='WSE').reset_index().melt(id_vars=['MONTH', 'YEAR'], var_name='STATION', value_name='WSE_imputed')

    # Add 'was_missing' column
    original_pivot_data = group_data.pivot(index=['MONTH', 'YEAR'], columns='STATION', values='WSE')
    imputed_long['was_missing'] = original_pivot_data.isnull().melt(value_name='was_missing')['was_missing']

    # Merge with original date and group information
    imputed_long['STATION'] = imputed_long['STATION'].str.split('_').str[1]  # Extract station name
    imputed_long = pd.merge(imputed_long, group_data[['MSMT_DATE', 'MONTH', 'YEAR', 'STATION', 'GROUP']], on=['MONTH', 'YEAR', 'STATION'])

    return imputed_long

# Apply KNN imputation to each group
infilled_data_list = []
for group in data['GROUP'].unique():
    group_data = data[data['GROUP'] == group].copy()
    infilled_group_data = knn_group_impute(group_data)
    infilled_group_data['GROUP'] = group
    infilled_data_list.append(infilled_group_data)

# Combine all infilled group data
infilled_data = pd.concat(infilled_data_list)

# Ensure each station has a single, continuous time series
final_data_list = []
for station in infilled_data['STATION'].unique():
    station_data = infilled_data[infilled_data['STATION'] == station].copy()
    original_data = data[data['STATION'] == station].copy()
    
    # Limit to the range of observations
    first_obs_date = original_data['MSMT_DATE'].min()
    last_obs_date = original_data['MSMT_DATE'].max()
    station_data = station_data[(station_data['MSMT_DATE'] >= first_obs_date) & (station_data['MSMT_DATE'] <= last_obs_date)]
    
    station_data = pd.merge(original_data[['MSMT_DATE', 'WSE', 'STATION', 'GROUP']], station_data[['MSMT_DATE', 'WSE_imputed', 'was_missing']], on='MSMT_DATE', how='left')
    station_data['WSE_final'] = station_data['WSE']
    station_data.loc[station_data['was_missing'], 'WSE_final'] = station_data.loc[station_data['was_missing'], 'WSE_imputed']
    final_data_list.append(station_data)

# Combine final station data
final_data = pd.concat(final_data_list)

# Save the infilled dataset
final_data.to_csv('gwl-monthly-imputed-KNN.csv', index=False)

# Function to generate time series graphs for each station
def generate_time_series_graphs(station_data, station_name):
    plt.figure(figsize=(10, 6), dpi=300)  # Increase resolution with dpi=300

    # Plot the entire series with different colors for observed and estimated values
    for i in range(1, len(station_data)):
        if station_data['was_missing'].iloc[i] and station_data['was_missing'].iloc[i-1]:
            color = 'lightsalmon'  # Estimated-Estimated
        elif station_data['was_missing'].iloc[i] or station_data['was_missing'].iloc[i-1]:
            color = 'lightsalmon'  # Observed-Estimated or Estimated-Observed
        else:
            color = 'lightskyblue'  # Observed-Observed
        plt.plot(station_data['MSMT_DATE'].iloc[i-1:i+1], station_data['WSE_final'].iloc[i-1:i+1], color=color)

    # Plot the dots
    plt.plot(station_data['MSMT_DATE'], station_data['WSE_final'], 'o', markersize=2, color='royalblue', label='Observed')
    plt.plot(station_data.loc[station_data['was_missing'], 'MSMT_DATE'], station_data.loc[station_data['was_missing'], 'WSE_final'], 'o', markersize=2, color='firebrick', label='Infilled')

    plt.xlabel('')
    plt.ylabel('Water Surface Elevation (WSE)')
    plt.title(f'Time Series Graph for Station {station_name}')
    plt.legend()
    
    # Add light gray gridlines in the background
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)

    # Save the graph to a file in the output directory
    file_path = os.path.join(output_dir, f'{station_name}.png')
    plt.savefig(file_path)
    plt.close()

# Generate time series graphs for each station and save infilled data for each station separately
for station in final_data['STATION'].unique():
    station_data = final_data[final_data['STATION'] == station].copy()
    generate_time_series_graphs(station_data, station)

    # Save infilled data for each station separately
    station_infilled_file_path = os.path.join(infilled_data_dir, f'{station}.csv')
    station_data.to_csv(station_infilled_file_path, index=False)

    # Save estimated data for each station separately with time info
    estimated_series = station_data[station_data['was_missing']]
    estimated_file_path = os.path.join(estimated_data_dir, f'{station}.csv')

    if not estimated_series.empty:
        estimated_series.to_csv(estimated_file_path, columns=['MSMT_DATE', 'WSE_final'], index=False)

# Function to generate time series graphs for each group with multiple series in different colors
def generate_group_time_series_graphs(group_name, group_stations):
    plt.figure(figsize=(10, 6), dpi=300)  # Increase resolution with dpi=300

    # Exclude tones of red from the colormap
    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(group_stations)))
    colors = [color for color in colors if not (color[0] > 0.8 and color[1] < 0.3 and color[2] < 0.3)]  # Exclude colors close to red

    for i, (station_name, station_data) in enumerate(group_stations.items()):
        for j in range(1, len(station_data)):
            if station_data['was_missing'].iloc[j] and station_data['was_missing'].iloc[j-1]:
                color = 'lightsalmon'  # Estimated-Estimated
            elif station_data['was_missing'].iloc[j] or station_data['was_missing'].iloc[j-1]:
                color = 'lightsalmon'  # Observed-Estimated or Estimated-Observed
            else:
                rgba_color = colors[i % len(colors)].tolist()
                rgba_color[3] = 0.5  # Adjust the alpha value for lighter tone
                color = rgba_color
            plt.plot(station_data['MSMT_DATE'].iloc[j-1:j+1], station_data['WSE_final'].iloc[j-1:j+1], color=color)

        # Plot the dots
        plt.plot(station_data['MSMT_DATE'], station_data['WSE_final'], 'o', markersize=2, color=colors[i % len(colors)], label=f'{station_name} Observed')
        plt.plot(station_data.loc[station_data['was_missing'], 'MSMT_DATE'], station_data.loc[station_data['was_missing'], 'WSE_final'], 'o', markersize=2, color='firebrick')

    plt.xlabel('')
    plt.ylabel('Water Surface Elevation (WSE)')
    plt.title(f'Time Series Graph for Group {group_name}')
    plt.legend()
    
    # Add light gray gridlines in the background
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)

    # Save the graph to a file in the group output directory
    file_path = os.path.join(group_output_dir, f'{group_name}.png')
    plt.savefig(file_path)
    plt.close()

# Generate time series graphs for each group
for group in final_data['GROUP'].unique():
    group_data = final_data[final_data['GROUP'] == group]
    group_stations = {station: group_data[group_data['STATION'] == station] for station in group_data['STATION'].unique()}
    generate_group_time_series_graphs(group, group_stations)

print(f"Time series graphs have been generated and saved in the 'time_series_graphs_KNN' directory.")
print(f"Infilled data files have been generated and saved in the 'infilled_data_KNN' directory.")
print(f"Estimated data files have been generated and saved in the 'estimated_data_KNN' directory.")