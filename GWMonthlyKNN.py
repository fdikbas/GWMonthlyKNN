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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Progress bar (safe import)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

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

# --- FIX: pandas DeprecationWarning for GroupBy.apply operating on grouping columns ---
# New pandas prefers include_groups=False; provide fallback for older versions.
try:
    # include_groups=False -> grouping columns (STATION) group frame içine verilmez.
    # Bu yüzden STATION'ı index seviyesinden geri alıyoruz.
    data = (data
            .groupby('STATION', group_keys=True)
            .apply(filter_observation_range, include_groups=False)
            .reset_index(level=0)   # STATION'ı sütun olarak geri getirir
            .reset_index(drop=True))
except TypeError:
    # Older pandas: include_groups not supported
    data = (data
            .groupby('STATION', group_keys=False)
            .apply(filter_observation_range)
            .reset_index(drop=True))

# Güvenlik kontrolü (isterseniz bırakın; sorunu erken yakalar)
if 'STATION' not in data.columns:
    raise KeyError("STATION column missing after filtering; check GroupBy.apply settings.")



# Sort the data by station code and then by date within each station
data = data.sort_values(by=['STATION', 'MSMT_DATE'])

# Filter out groups with a single well where there is no missing value
group_sizes = data['GROUP'].value_counts()
single_well_groups = group_sizes[group_sizes == 1].index.tolist()

for group in tqdm(single_well_groups, desc="Filtering single-well groups", unit="group"):
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
    group_data['ROLLING_MEAN'] = group_data.groupby('STATION')['WSE'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    group_data['ROLLING_STD'] = group_data.groupby('STATION')['WSE'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )

    # Pivot (MONTH, YEAR) index
    pivot = group_data.pivot(
        index=['MONTH', 'YEAR'],
        columns='STATION',
        values=['WSE', 'LAG_1', 'LAG_2', 'ROLLING_MEAN', 'ROLLING_STD']
    )

    # Flatten column names
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]

    # >>> CRITICAL FIX (match 2025.01.18):
    # Put MONTH and YEAR into the feature matrix used by KNNImputer
    pivot = pivot.reset_index()

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_data = imputer.fit_transform(pivot)

    # Back to DataFrame + restore (MONTH, YEAR) index
    imputed_df = pd.DataFrame(imputed_data, columns=pivot.columns).set_index(['MONTH', 'YEAR'])

    return imputed_df


# Perform KNN imputation for each group and generate graphs
final_data_list = []

groups = data['GROUP'].unique().tolist()
for group in tqdm(groups, desc="Imputing groups (KNN)", unit="group"):
    group_data = data[data['GROUP'] == group].copy()

    # Mark missing values
    group_data['was_missing'] = group_data['WSE'].isnull()

    # Impute missing values using KNN
    imputed_df = knn_group_impute(group_data)

    # Extract the imputed WSE values
    wse_columns = [col for col in imputed_df.columns if col.startswith('WSE_')]
    imputed_wse = imputed_df[wse_columns]

    # Map back to original format
    stations_in_group = group_data['STATION'].unique().tolist()
    for station in stations_in_group:
        station_wse_col = f'WSE_{station}'
        station_data = group_data[group_data['STATION'] == station].copy()

        station_data = station_data.merge(
            imputed_wse[station_wse_col].reset_index().rename(columns={station_wse_col: 'WSE_imputed'}),
            on=['MONTH', 'YEAR'],
            how='left'
        )

        # Create final series (observed where available, otherwise imputed)
        station_data['WSE_final'] = station_data['WSE']
        station_data.loc[station_data['was_missing'], 'WSE_final'] = station_data.loc[
            station_data['was_missing'], 'WSE_imputed'
        ]
        final_data_list.append(station_data)

# Combine final station data
final_data = pd.concat(final_data_list)

# Save the infilled dataset
final_data.to_csv('gwl-monthly-imputed-KNN.csv', index=False)


# ----------------- NEW FIGURE 1: Network-scale observed vs imputed map (LEGEND, yearly vertical lines) -----------------
network_fig_dir = "network_diagnostics"
os.makedirs(network_fig_dir, exist_ok=True)

# Station order: by GROUP then STATION (stable ordering improves readability)
station_order = (final_data[['GROUP', 'STATION']]
                 .drop_duplicates()
                 .sort_values(['GROUP', 'STATION'])['STATION']
                 .tolist())

# Pivot: rows=stations, cols=dates, values=was_missing (True=imputed, False=observed)
prov = (final_data
        .pivot_table(index='STATION', columns='MSMT_DATE', values='was_missing', aggfunc='max')
        .reindex(station_order)
        .sort_index(axis=1))  # ensure dates are sorted left-to-right

# Warn if there are months with no record for some stations (NaN in the map)
if prov.isna().any().any():
    print("Warning: Figure 1 provenance map contains NaN values "
          "(some stations have months with no record in the plotted time domain).")

# Convert to 0/1 for plotting (NaN remains NaN)
prov_mat = prov.astype(float).values  # 0=observed, 1=imputed, NaN=no record
nan_mask = np.isnan(prov_mat)

# ---- Layered plotting to control z-order precisely:
#      (0) background gray for NaNs
#      (1) yearly vertical lines (in front of background, behind data)
#      (2) observed/imputed pixels (NaNs transparent)
fig, ax = plt.subplots(figsize=(12, 6), dpi=400)

# (0) Background layer: show ONLY NaNs as light gray; everything else transparent
bg = np.where(nan_mask, 1.0, np.nan)  # 1 where NaN, else NaN (transparent via set_bad)
bg_cmap = ListedColormap(['lightgray'])
bg_cmap.set_bad((0, 0, 0, 0))  # fully transparent for non-NaN
ax.imshow(bg, aspect='auto', interpolation='nearest', cmap=bg_cmap, vmin=0, vmax=1, zorder=0)

# Build year -> first column index mapping
dates = prov.columns.to_list()
year_first_idx = {}
for i, d in enumerate(dates):
    y = d.year
    if y not in year_first_idx:
        year_first_idx[y] = i

# (1) Yearly vertical lines: several gray tones, behind data but in front of background
for y, idx in year_first_idx.items():
    # place line at the left edge of the first month for that year
    x = idx - 0.5

    # "a few gray tones": darker for decades, medium for 5-year marks, light otherwise
    if y % 10 == 0:
        col, lw = '0.55', 0.9
    elif y % 5 == 0:
        col, lw = '0.65', 0.7
    else:
        col, lw = '0.75', 0.5

    ax.axvline(x=x, color=col, linewidth=lw, zorder=1)

# (2) Data layer: observed/imputed; NaNs transparent so background shows through
data_cmap = ListedColormap(['royalblue', 'yellow'])  # 0=Observed, 1=Imputed
data_cmap.set_bad((0, 0, 0, 0))  # NaNs fully transparent in the data layer
norm = BoundaryNorm([-0.5, 0.5, 1.5], data_cmap.N)
ax.imshow(prov_mat, aspect='auto', interpolation='nearest', cmap=data_cmap, norm=norm, zorder=2)

# X ticks: yearly (or every 2 years if too dense)
years = sorted(year_first_idx.keys())
tick_year_step = 2 if len(years) > 15 else 1
xticks, xticklabels = [], []
for y in years:
    if (y - years[0]) % tick_year_step == 0:
        xticks.append(year_first_idx[y])
        xticklabels.append(str(y))

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=0)
ax.set_yticks([])  # too many stations; omit for clarity
ax.set_xlabel("Year")
ax.set_title("Network-Scale Provenance Map (Observed vs Imputed Months)")

# Legend (upper left)
legend_handles = [
    Patch(facecolor='royalblue', edgecolor='k', label='Observed'),
    Patch(facecolor='yellow', edgecolor='k', label='Imputed'),
]
ax.legend(handles=legend_handles, loc='upper left', frameon=True)

fig.tight_layout()
fig.savefig(os.path.join(network_fig_dir, "network_provenance_observed_vs_imputed.png"))
fig.savefig(os.path.join(network_fig_dir, "network_provenance_observed_vs_imputed.pdf"))
plt.close(fig)
# ----------------- END NEW FIGURE 1 ------------------------------------------------------------------------


# ----------------- NEW FIGURE 2: Contiguous gap-length distribution -----------------
def contiguous_true_run_lengths(bool_array):
    """Return lengths of contiguous True runs in a 1D boolean array."""
    runs = []
    run = 0
    for v in bool_array:
        if bool(v):
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    return runs

gap_lengths = []
for st, sdf in final_data.sort_values('MSMT_DATE').groupby('STATION'):
    mask = sdf['was_missing'].to_numpy()
    gap_lengths.extend(contiguous_true_run_lengths(mask))

gap_fig_dir = "network_diagnostics"
os.makedirs(gap_fig_dir, exist_ok=True)

if len(gap_lengths) > 0:
    plt.figure(figsize=(10, 5), dpi=300)
    # Bin up to, say, 60 months; longer gaps go to the tail automatically
    max_len = max(gap_lengths)
    upper = max(60, max_len)
    bins = np.arange(1, upper + 2) - 0.5  # center bins on integers

    plt.hist(gap_lengths, bins=bins, linewidth=0.5)
    plt.axvline(12, linestyle='--', linewidth=1.0)
    plt.axvline(24, linestyle='--', linewidth=1.0)

    plt.xlabel("Contiguous gap length (months)")
    plt.ylabel("Number of gaps")
    plt.title("Distribution of Contiguous Missing-Data Gaps (All Wells)")

    plt.tight_layout()
    plt.savefig(os.path.join(gap_fig_dir, "gap_length_distribution.png"))
    plt.savefig(os.path.join(gap_fig_dir, "gap_length_distribution.pdf"))
    plt.close()



# --- Plotting helpers (dashed segments to missing points, hollow infilled markers,
# --- and optional partitioned plots for long records; PNG + PDF export) -----------------
def _record_span_years(date_series):
    """Return the span of a time series in years (approx.), based on min/max dates."""
    if date_series is None or len(date_series) == 0:
        return 0.0
    dmin = pd.to_datetime(date_series.min())
    dmax = pd.to_datetime(date_series.max())
    if pd.isna(dmin) or pd.isna(dmax):
        return 0.0
    return max(0.0, (dmax - dmin).days / 365.25)


def _partition_date_ranges(date_series, threshold_2yrs=10.0, threshold_3yrs=30.0):
    """Create (start,end,n_parts,part_index) partitions based on record length.
    Full-series plot is handled separately; this returns ONLY additional partitions.
    """
    if date_series is None or len(date_series) == 0:
        return []
    dmin = pd.to_datetime(date_series.min())
    dmax = pd.to_datetime(date_series.max())
    if pd.isna(dmin) or pd.isna(dmax) or dmax <= dmin:
        return []
    span_years = (dmax - dmin).days / 365.25
    if span_years > threshold_3yrs:
        n_parts = 3
    elif span_years > threshold_2yrs:
        n_parts = 2
    else:
        return []

    total_days = (dmax - dmin).days
    edges = [dmin + pd.Timedelta(days=int(round(total_days * k / n_parts))) for k in range(n_parts + 1)]
    edges[-1] = dmax

    ranges = []
    for p in range(n_parts):
        start = edges[p]
        end = edges[p + 1]
        ranges.append((start, end, n_parts, p + 1))
    return ranges


def generate_time_series_graphs(station_data, station_name, file_stem=None, title_suffix=None):
    station_data = station_data.copy()
    station_data = station_data.sort_values('MSMT_DATE')

    plt.figure(figsize=(10, 6), dpi=300)

    # Line segments: dashed if segment touches an infilled point
    for i in range(1, len(station_data)):
        seg_has_missing = bool(station_data['was_missing'].iloc[i]) or bool(station_data['was_missing'].iloc[i-1])
        if seg_has_missing:
            color = 'lightsalmon'
            linestyle = '--'
        else:
            color = 'lightskyblue'
            linestyle = '-'
        plt.plot(
            station_data['MSMT_DATE'].iloc[i-1:i+1],
            station_data['WSE_final'].iloc[i-1:i+1],
            color=color,
            linestyle=linestyle,
            linewidth=0.8
        )

    # Points: observed vs infilled
    obs_mask = ~station_data['was_missing']
    miss_mask = station_data['was_missing']

    plt.plot(
        station_data.loc[obs_mask, 'MSMT_DATE'],
        station_data.loc[obs_mask, 'WSE_final'],
        linestyle='None',
        marker='o',
        markersize=2,
        color='royalblue',
        label='Observed'
    )

    plt.plot(
        station_data.loc[miss_mask, 'MSMT_DATE'],
        station_data.loc[miss_mask, 'WSE_final'],
        linestyle='None',
        marker='o',
        markersize=2,
        markerfacecolor='white',
        markeredgecolor='firebrick',
        markeredgewidth=0.8,
        label='Infilled'
    )

    plt.xlabel('')
    plt.ylabel('Water Surface Elevation (WSE)')

    if title_suffix is None:
        plt.title(f'Time Series Graph for Station {station_name}')
    else:
        plt.title(f'Time Series Graph for Station {station_name}{title_suffix}')

    plt.legend()
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)

    if file_stem is None:
        file_stem = station_name

    file_path_png = os.path.join(output_dir, f'{file_stem}.png')
    file_path_pdf = os.path.join(output_dir, f'{file_stem}.pdf')

    plt.savefig(file_path_png)
    plt.savefig(file_path_pdf)
    plt.close()

# Generate time series graphs for each station and save infilled data for each station separately
stations_all = final_data['STATION'].unique().tolist()
for station in tqdm(stations_all, desc="Plotting stations", unit="station"):
    station_data = final_data[final_data['STATION'] == station].copy()
    generate_time_series_graphs(station_data, station)

    # Additional partitioned plots for long records (full-series plot is kept)
    _ranges = _partition_date_ranges(station_data['MSMT_DATE'])
    for (_start, _end, _n_parts, _pidx) in _ranges:
        _subset = station_data[(station_data['MSMT_DATE'] >= _start) & (station_data['MSMT_DATE'] <= _end)].copy()
        if not _subset.empty:
            generate_time_series_graphs(
                _subset,
                station,
                file_stem=f"{station}_part{_pidx}of{_n_parts}",
                title_suffix=f" (Part {_pidx}/{_n_parts})"
            )

    # Save infilled data for each station separately
    station_infilled_file_path = os.path.join(infilled_data_dir, f'{station}.csv')
    station_data.to_csv(station_infilled_file_path, index=False)

    # Save estimated data for each station separately with time info
    estimated_series = station_data[station_data['was_missing']]
    estimated_file_path = os.path.join(estimated_data_dir, f'{station}.csv')
    if not estimated_series.empty:
        estimated_series.to_csv(estimated_file_path, columns=['MSMT_DATE', 'WSE_final'], index=False)


def generate_group_time_series_graphs(group_name, group_stations, file_stem=None, title_suffix=None):
    plt.figure(figsize=(10, 6), dpi=300)

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(group_stations)))
    colors = [c for c in colors if not (c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3)]

    for i, (station_name, station_data) in enumerate(group_stations.items()):
        station_data = station_data.copy().sort_values('MSMT_DATE')

        rgba_color = colors[i % len(colors)].tolist()
        rgba_color[3] = 0.5

        for j in range(1, len(station_data)):
            seg_has_missing = bool(station_data['was_missing'].iloc[j]) or bool(station_data['was_missing'].iloc[j-1])
            if seg_has_missing:
                color = 'lightsalmon'
                linestyle = '--'
                alpha = 1.0
            else:
                color = rgba_color
                linestyle = '-'
                alpha = rgba_color[3]

            plt.plot(
                station_data['MSMT_DATE'].iloc[j-1:j+1],
                station_data['WSE_final'].iloc[j-1:j+1],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=alpha
            )

        obs_mask = ~station_data['was_missing']
        miss_mask = station_data['was_missing']

        plt.plot(
            station_data.loc[obs_mask, 'MSMT_DATE'],
            station_data.loc[obs_mask, 'WSE_final'],
            linestyle='None',
            marker='o',
            markersize=2,
            color=rgba_color,
            label=station_name
        )

        plt.plot(
            station_data.loc[miss_mask, 'MSMT_DATE'],
            station_data.loc[miss_mask, 'WSE_final'],
            linestyle='None',
            marker='o',
            markersize=2,
            markerfacecolor='white',
            markeredgecolor='firebrick',
            markeredgewidth=0.8
        )

    plt.xlabel('')
    plt.ylabel('Water Surface Elevation (WSE)')

    if title_suffix is None:
        plt.title(f'Time Series Graph for Group {group_name}')
    else:
        plt.title(f'Time Series Graph for Group {group_name}{title_suffix}')

    plt.legend()
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5)

    if file_stem is None:
        file_stem = group_name

    file_path_png = os.path.join(group_output_dir, f'{file_stem}.png')
    file_path_pdf = os.path.join(group_output_dir, f'{file_stem}.pdf')

    plt.savefig(file_path_png)
    plt.savefig(file_path_pdf)
    plt.close()

# Generate time series graphs for each group
groups_all = final_data['GROUP'].unique().tolist()
for group in tqdm(groups_all, desc="Plotting groups", unit="group"):
    group_data = final_data[final_data['GROUP'] == group]
    group_stations = {st: group_data[group_data['STATION'] == st] for st in group_data['STATION'].unique()}
    generate_group_time_series_graphs(group, group_stations)

    _ranges_g = _partition_date_ranges(group_data['MSMT_DATE'])
    for (_start, _end, _n_parts, _pidx) in _ranges_g:
        _subset = group_data[(group_data['MSMT_DATE'] >= _start) & (group_data['MSMT_DATE'] <= _end)]
        _group_stations_subset = {st: _subset[_subset['STATION'] == st] for st in _subset['STATION'].unique()}
        if len(_group_stations_subset) > 0:
            generate_group_time_series_graphs(
                group,
                _group_stations_subset,
                file_stem=f"{group}_part{_pidx}of{_n_parts}",
                title_suffix=f" (Part {_pidx}/{_n_parts})"
            )

print("Time series graphs have been generated and saved in the 'time_series_graphs_KNN' directory.")
print("Infilled data files have been generated and saved in the 'infilled_data_KNN' directory.")
print("Estimated data files have been generated and saved in the 'estimated_data_KNN' directory.")
print("Group time series graphs have been generated and saved in the 'group_time_series_graphs_KNN' directory.")
