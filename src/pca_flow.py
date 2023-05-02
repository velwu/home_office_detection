import pandas as pd
import numpy as np
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from folium.plugins import MarkerCluster

def read_loc_data(file_path:str):
    df = pd.read_csv(file_path)
    df.sort_values(["id", "start_time"],axis = 0, ascending = True,inplace = True)
    df["id"] = df["id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["start_time"])
    # df["time_unix"] = (df["timestamp"].values.astype(float) / 10**9).astype(int)
    df = df.drop(["start_time"], axis='columns')
    df.reset_index(inplace=True)

    return df

# TODO: multiprocessing: list([df1, df2, ~~])

def get_timestamped_latlon(df:pd.DataFrame, select_id:str, day_count:int, fill_method:str):

    # TODO: this also has to work for multiple IDs

    df_sample = df[df["id"] == select_id][['timestamp', 'lat', 'lon']]

    # TODO: is this the best way to dedup?
    df_unique = df_sample.drop_duplicates(subset='timestamp', keep='first')
    
    df_unique.set_index('timestamp', inplace=True)
    # resample the data, 15 minutes per row
    df_resampled = df_unique.resample('15T').asfreq()
    
    if fill_method == "lerp":
        # Method A: "LERP" a.k.a. Linear Interpolation, fills rows using the mean lat-lon of rows before and after.
        df_interpolated = df_resampled.interpolate()

    if fill_method == "ffill":
        # Method B: "F-Fill" a.k.a. Forward-Fill, the empty row's value using the row before
        df_interpolated = df_resampled.fillna(method='ffill')

    # Make sure the 'timestamp' column is set as the index
    if not isinstance(df_interpolated.index, pd.DatetimeIndex):
        df_interpolated['timestamp'] = pd.to_datetime(df['timestamp'])
        df_interpolated.set_index('timestamp', inplace=True)

    # Get the date of the first row.
    first_date = df_interpolated.index[0].date()

    # Create a new DataFrame with a complete date range
    full_range = pd.date_range(start=first_date, end=first_date + pd.Timedelta(days=day_count) - pd.Timedelta(minutes=15), freq='15T')
    df_complete = pd.DataFrame(index=full_range).reset_index().rename(columns={'index': 'timestamp'})

    # Merge the original dataset with the new DataFrame
    df_merged = df_complete.merge(df_interpolated, on='timestamp', how='left')

    # Set the 'timestamp' column as the index
    df_merged.set_index('timestamp', inplace=True)

    # Interpolate missing values. The [:-1] removes the last row which belongs to the next day
    df_parsed = df_merged.interpolate()[:-1]
    # Find the index of the first non-empty row
    first_valid_index = df_parsed.apply(pd.Series.first_valid_index)
    # Find the earliest valid index among all columns
    earliest_valid_index = first_valid_index.min()

    # Backfill NaN values before the first non-empty row
    df_parsed.loc[:earliest_valid_index] = df_parsed.loc[:earliest_valid_index].fillna(method='bfill')

    # Convert the Timestamp index back to regular column so we can apply datetime logics over it
    df_parsed.reset_index(inplace=True)
    df_parsed["date"] = df_parsed["timestamp"].dt.date
    
    return df_parsed

def get_pca_matrix(df:pd.DataFrame):

    result = df.groupby('date').agg({'lat': list, 'lon': list}).reset_index()
    # df_proto_matrix['lat_lon'] = df_proto_matrix['lat'] + df_proto_matrix['lon']

    # Find the maximum length of latitude and longitude lists
    max_len_lat = result['lat'].apply(len).max()
    max_len_lon = result['lon'].apply(len).max()

    # Pad the lists with NaN values to make them the same length
    result['lat_padded'] = result['lat'].apply(lambda x: np.pad(x, (0, max_len_lat - len(x)), mode='constant', constant_values=np.nan))
    result['lon_padded'] = result['lon'].apply(lambda x: np.pad(x, (0, max_len_lon - len(x)), mode='constant', constant_values=np.nan))

    # Create separate columns for each value in the padded lists
    for i in range(max_len_lat):
        result[f'lat_{i}'] = result['lat_padded'].apply(lambda x: x[i])
    for i in range(max_len_lon):
        result[f'lon_{i}'] = result['lon_padded'].apply(lambda x: x[i])

    # Drop the original and padded list columns
    result = result.drop(columns=['lat', 'lon', 'lat_padded', 'lon_padded', 'date'])
    return result

def run_pca_model(df_matrix: pd.DataFrame):
    scaler = StandardScaler(with_std=False)
    scaler.fit(df_matrix)
    data_matrix = scaler.transform(df_matrix)
    pca = PCA(n_components=2)
    pca.fit_transform(data_matrix)
    array_lat_lon = scaler.inverse_transform(pca.components_[0].reshape(1, -1))

    # `array_lat_lon.shape` should always be (1, 192). It is an array that represents 96 lat-lon pairs. The first 96 are latitutdes, and the second are longitudes.
    latitudes = array_lat_lon[:, :96]
    longitudes = array_lat_lon[:, 96:]

    # Transpose the arrays to have shape (96,)
    latitudes = latitudes.flatten()
    longitudes = longitudes.flatten()
    # Create a DataFrame with lat-lon pairs
    result = pd.DataFrame({'lat': latitudes, 'lon': longitudes})
    result['eigen'] = pca.explained_variance_ratio_[0] # alternative: pca.explained_variance_[0] which is raw eigen value
    return result

def plot_latlon(df:pd.DataFrame, clusters:bool):
    # The input `df` must have `lat` and `lon` columns
    # Create a folium Map instance
    map = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)
    # Create a MarkerCluster instance

    if clusters:
        marker_cluster = MarkerCluster()
        # Add markers to the cluster
        for index, row in df.iterrows():
            marker = folium.Marker(location=[row['lat'], row['lon']])
            marker_cluster.add_child(marker)
        # Add the MarkerCluster instance to the map
        map.add_child(marker_cluster)
    else:
        for index, row in df.iterrows():
            marker = folium.Marker(location=[row['lat'], row['lon']])
            marker.add_to(map)
    return map

def plot_latlon_traces(df:pd.DataFrame):

    map = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=13)

    # Iterate through the DataFrame and plot each row as a point on the map
    # Create lines between neighboring rows
    for index, row in df.iterrows():
        lat, lon = row['lat'], row['lon']

        # Set default marker color
        marker_color = 'blue'

        # Highlight the first row with green
        if index == df.index[0]:
            marker_color = 'green'
        # Highlight the last row with dark yellow
        elif index == df.index[-1]:
            marker_color = 'darkred'

        # Create a custom icon with both the marker symbol and the row index
        icon = folium.DivIcon(
            html=f'<div><i class="fa fa-map-marker fa-2x" style="color: {marker_color};"></i></div><div style="color: {marker_color};">{index}</div>',
            icon_size=(30, 30),
        )

        # Create a marker with the specified icon and a popup showing the row index
        marker = folium.Marker(location=[lat, lon], icon=icon, popup=folium.Popup(f'Row Index: {index}'))
        marker.add_to(map)

        # Add a line between the current row and the previous row
        if index > df.index[0]:
            prev_row = df.loc[index - 1]
            folium.PolyLine(
                locations=[[prev_row['lat'], prev_row['lon']], [lat, lon]],
                color='lightgreen',
                weight=1.5
            ).add_to(map)
    return map
