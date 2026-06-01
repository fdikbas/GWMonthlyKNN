# GWMonthlyKNN.py
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.


# -----------------------------------------------------------------------------
# BEGINNER-FRIENDLY EXPLANATION / USER GUIDE
# -----------------------------------------------------------------------------
#
# What this script does, in one sentence
# --------------------------------------
# This script reads a monthly groundwater-level dataset with missing values,
# fills the missing monthly Water Surface Elevation (WSE) values using a
# group-wise K-Nearest Neighbors (KNN) imputation workflow, and then saves both
# the completed data tables and diagnostic time-series graphs.
#
# Why this script exists
# ----------------------
# Groundwater monitoring records often contain missing months. These gaps can
# occur because a sensor failed, telemetry stopped, access to the well was not
# possible, a quality-control procedure removed a measurement, or the monthly
# value was simply not reported. Many downstream groundwater analyses need a
# continuous monthly series: drought diagnostics, seasonal minimum/maximum
# analyses, trend calculations, model calibration, forecasting, and dashboards.
#
# This script provides a transparent baseline way to complete those gaps using
# only the groundwater-level records themselves. It does NOT require rainfall,
# pumping, remote sensing, aquifer-test data, lithologic logs, or any external
# covariates. That is useful when the goal is to prepare a large monitoring
# archive in a reproducible and auditable way.
#
# Required input file
# -------------------
# The script expects one CSV file in the same working directory:
#
#     gwl-monthly.grouped.csv
#
# The CSV must contain at least these columns:
#
#     MSMT_DATE   : date of the monthly observation. Example: 2018-12-01
#     STATION     : unique well/station identifier. Example: 10N04E27R002M
#     GROUP       : group identifier used to decide which wells can borrow
#                   information from each other. In the manuscript this is the
#                   PLSS-prefix group, but it can also be any user-defined group.
#     WSE         : Water Surface Elevation. Missing values should be blank or NaN.
#
# Very important: this script uses the GROUP column as the donor-pool boundary.
# A well is imputed only together with wells in the same GROUP. If the GROUP
# column is poorly defined, the imputation may also be poor. For other datasets,
# GROUP can be based on aquifer code, basin ID, management zone, distance-based
# neighborhood, or correlation-based clustering.
#
# Main idea of the method
# -----------------------
# KNN imputation fills a missing value by searching for similar rows and taking a
# weighted average of their observed values. In this script, similarity is not
# based only on the date. The script builds a feature table that includes:
#
#     1. MONTH and YEAR
#        These help encode the seasonal timing and long-term temporal position.
#
#     2. WSE
#        The observed water level itself, where available.
#
#     3. LAG_1
#        The groundwater level one month earlier for the same station.
#
#     4. LAG_2
#        The groundwater level two months earlier for the same station.
#
#     5. ROLLING_MEAN
#        A 3-month rolling mean. This is a compact local baseline.
#
#     6. ROLLING_STD
#        A 3-month rolling standard deviation. This is a compact local variability
#        descriptor.
#
# The KNNImputer then searches in this engineered feature space. Rows that are
# closer in this feature space are treated as more similar. Because the imputer is
# used with weights="distance", closer neighbors have more influence than farther
# neighbors.
#
# What "group-wise" means
# -----------------------
# The script does not run one single KNN model on the entire 515-well network.
# Instead, it loops over each GROUP separately:
#
#     for each GROUP:
#         take all wells in that group
#         create lag and rolling features
#         pivot the data into a wide table
#         run KNNImputer inside that group
#         return completed values for the wells in that group
#
# This reduces the risk that a well in one hydrogeologic setting borrows values
# from a completely unrelated well somewhere else. It also keeps the computation
# smaller and easier to inspect.
#
# What happens to single-well groups
# ----------------------------------
# Some groups contain only one well. If a single-well group has no missing WSE
# values, the script removes it from the imputation run because there is nothing
# to fill. If a single-well group contains missing values, it remains in the
# workflow. In that case, KNN cannot borrow from neighboring wells, so it relies
# only on temporal self-similarity through MONTH/YEAR, lag features, and rolling
# statistics. These estimates should be interpreted more cautiously than estimates
# supported by coherent multiwell groups.
#
# Observation-window trimming
# ---------------------------
# The script removes rows before the first observed WSE and after the last
# observed WSE for each station. This is important because the goal is internal
# gap filling, not extrapolation. In other words:
#
#     OK:   fill missing months between the first and last observed month.
#     NOT:  invent values before monitoring began or after monitoring ended.
#
# How missing values are marked
# -----------------------------
# The script creates a column named:
#
#     was_missing
#
# This is True for months that were missing in the original WSE column and were
# therefore filled by the imputation workflow. It is False for months that were
# originally observed. This column is essential for auditability because it lets a
# user distinguish measured values from estimated values.
#
# Main output table
# -----------------
# The most important output file is:
#
#     gwl-monthly-imputed-KNN.csv
#
# It contains, for each station and month:
#
#     MSMT_DATE    : monthly date
#     WSE          : original value; observed months have numbers, missing months
#                    remain missing
#     STATION      : well identifier
#     GROUP        : donor-pool group identifier
#     WSE_imputed  : KNN-imputed value returned by the imputer
#     was_missing  : True if the original WSE was missing
#     WSE_final    : final completed series. This equals WSE for observed months
#                    and equals WSE_imputed for originally missing months.
#
# The safest column for downstream analyses is usually WSE_final, because it is
# the completed series. However, WSE and was_missing should always be retained so
# users can tell which values were measured and which were estimated.
#
# Output folders
# --------------
# The script creates four output folders if they do not already exist:
#
#     time_series_graphs_KNN/
#         One PNG graph per station. These graphs show observed and infilled
#         values in the completed time series.
#
#     infilled_data_KNN/
#         One CSV file per station containing the completed station-level data.
#
#     estimated_data_KNN/
#         One CSV file per station containing only the months that were actually
#         estimated. Stations without missing values may not produce a file here.
#
#     group_time_series_graphs_KNN/
#         One PNG graph per group, with all wells in that group plotted together.
#         These plots are useful for visually checking whether the group behaves
#         coherently and whether infilled values look plausible.
#
# How the station-level graphs should be read
# -------------------------------------------
# In the station graphs:
#
#     blue markers/segments generally indicate observed portions of the series.
#     red/firebrick markers indicate infilled months.
#     light-salmon connecting lines highlight segments involving at least one
#     imputed value.
#
# These plots are not just decorative. They are diagnostic plots. A hydrologist or
# data manager should inspect them to check whether imputed months follow the
# expected seasonal cycle, long-term trend, and neighboring-well behavior.
#
# How the group-level graphs should be read
# -----------------------------------------
# Group graphs show all stations in the same GROUP together. They are useful for
# quickly identifying:
#
#     - whether wells in the same group move together,
#     - whether one well behaves very differently from the others,
#     - whether missing months are simultaneous across wells,
#     - whether imputation is weak because all wells are missing at the same time,
#     - whether a group should be split or merged before a future rerun.
#
# Important methodological limitation
# -----------------------------------
# This script provides deterministic imputed values. It does not provide formal
# prediction intervals or probabilistic uncertainty. Uncertainty must be assessed
# separately, for example by using the companion full-network validation script
# that masks known observed blocks and calculates RMSE, MAE, bias, NSE, R, and R2.
#
# Long consecutive gaps
# ---------------------
# Lag and rolling features are computed from the original WSE series before KNN
# imputation. During long consecutive missing gaps, lag and rolling features can
# become missing because recent observed WSE values are unavailable. KNNImputer
# can still operate because it uses the subset of jointly available features when
# computing nan-aware Euclidean distances. However, long gaps are intrinsically
# harder, especially when neighboring wells are also missing during the same time.
#
# What this script does NOT do
# ----------------------------
# This script does not:
#
#     - prove that the chosen GROUP definitions are hydrologically perfect;
#     - use pumping, precipitation, land use, lithology, or remote sensing data;
#     - optimize K automatically for each station;
#     - quantify formal uncertainty intervals;
#     - validate performance by itself using masked-gap experiments;
#     - extrapolate outside a station's monitoring period;
#     - guarantee that every imputed value is hydrologically correct.
#
# Instead, it provides a transparent, reproducible, easy-to-inspect baseline.
#
# When to trust the output more
# -----------------------------
# Imputed values are generally more credible when:
#
#     - the station belongs to a multiwell group,
#     - neighboring wells have overlapping observations,
#     - wells in the same group show similar seasonal and long-term behavior,
#     - missing gaps are not extremely long,
#     - missing gaps are not simultaneous across all wells in the group,
#     - the diagnostic graphs look hydrogeologically plausible.
#
# When to be cautious
# -------------------
# Imputed values should be interpreted cautiously when:
#
#     - the well is in a single-well group,
#     - the group contains wells with very different hydrographs,
#     - local pumping causes abrupt or irregular changes,
#     - stratigraphic differences make nearby wells behave differently,
#     - the missing period is long and coincides with missing data in other wells,
#     - the imputed values smooth out important minima or maxima.
#
# Typical workflow for a new user
# -------------------------------
# 1. Put GWMonthlyKNN.py and gwl-monthly.grouped.csv in the same folder.
# 2. Open Spyder or another Python IDE.
# 3. Set the working directory to that folder.
# 4. Run the script.
# 5. Check that gwl-monthly-imputed-KNN.csv was created.
# 6. Inspect several station and group graphs.
# 7. Use WSE_final for downstream analyses, while keeping was_missing for audit.
# 8. For manuscript-level evidence, run the separate validation script to quantify
#    performance under artificial 12- and 24-month gaps.
#
# Common problems and simple fixes
# --------------------------------
# Problem: FileNotFoundError for gwl-monthly.grouped.csv
# Fix:     Put the CSV in the same folder as this script, or change the read_csv
#          path below.
#
# Problem: KeyError for STATION, GROUP, MSMT_DATE, or WSE
# Fix:     Check that the input CSV uses exactly these column names.
#
# Problem: Date conversion error in MSMT_DATE
# Fix:     Make sure dates are in a format pandas can parse, such as YYYY-MM-DD.
#
# Problem: Empty or strange graphs
# Fix:     Check whether the station has enough observed WSE values and whether
#          the GROUP column is correct.
#
# Problem: Imputed values look unrealistic
# Fix:     Inspect the group graph. The group may combine hydrologically different
#          wells. Try redefining GROUP using aquifer, basin, screened interval,
#          distance, or correlation information and rerun the script.
#
# Reproducibility note
# --------------------
# The script is intentionally simple and parameter-light. The main exposed KNN
# parameter is n_neighbors in knn_group_impute(), defaulting to 5. If you change
# this value, document the change and rerun validation before using the outputs in
# a scientific manuscript or operational decision workflow.
#
# License note
# ------------
# This script is released under the GNU General Public License (GPL), as stated
# above. Keep the license notice if you redistribute or modify the file.
#
# -----------------------------------------------------------------------------
# END OF BEGINNER-FRIENDLY EXPLANATION
# -----------------------------------------------------------------------------


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