'''
跳過PCA, 只用OPTICS clustering進行停留點定位
THE CHOSEN ONE: 510018242552475
'''

import os
import webbrowser
import sys
import glob
# os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append("src/")
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import folium
from geopy import distance
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from folium.plugins import MarkerCluster

# data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
# data = utils.read_dmp_data("/data/0322-0409_batchE_top.csv")
# fix_map = False
# epsilon = sys.float_info.epsilon

def pipeline(uuid:str, threshold:float):
    data = utils.read_dmp_data("/data/0322-0409_batchE_top.csv")
    fix_map = False
    print(uuid)
    from sklearn.cluster import OPTICS

    painter = utils.footprint_display()
    footprint = data[uuid]
    matrix = utils.footprint2matrix(footprint)
    cluster_input = []
    latlon_list = []

    for idx in range(1, len(footprint)-1):
        T1, lat_t1, lon_t1, dur_t1 = footprint[idx-1]
        T2, lat_t2, lon_t2, dur_t2 = footprint[idx]
        T3, lat_t3, lon_t3, dur_t3 = footprint[idx+1]
        vector1 = np.array([lat_t2-lat_t1, lon_t2-lon_t1])
        vector2 = np.array([lat_t3-lat_t2, lon_t3-lon_t2])
        dot = np.dot(vector1, vector2)/(np.sqrt(np.sum(vector1**2)+epsilon)*(np.sqrt(np.sum(vector2**2)+epsilon)))
        
        if dot <= threshold:
            cluster_input.append([
                lat_t2, 
                lon_t2,
                dot,
                distance.distance((lat_t2, lon_t2), (lat_t1, lon_t1)).m])
            
            latlon_list.append([
                lat_t2, 
                lon_t2])
    
    scaler = StandardScaler(with_mean=False, with_std=True)
    cluster = OPTICS(min_samples=int(len(footprint)/10)).fit(scaler.fit_transform(np.array(cluster_input)))
    group = [str(i) for i in cluster.labels_]
    centers = [[
        np.median([latlon_list[i][0] for i in range(len(cluster.labels_)) if cluster.labels_[i]==level]),
        np.median([latlon_list[i][1] for i in range(len(cluster.labels_)) if cluster.labels_[i]==level])]
        for level in set(cluster.labels_) if level >=0]

    painter.plot_map(latlon_list, group, f"{uuid}_display", centers=centers, fix_map=fix_map)
    painter.plot_gif(matrix, f"{uuid}_footprint", centers=centers, fix_map=fix_map)

def filter_by_cosine(csv_file_path:str, uuid:str, threshold:float, date_chosen:str):
    epsilon = sys.float_info.epsilon
    print(uuid)
    if date_chosen == '':
        data = utils.read_dmp_data(csv_file_path)
    else:
        data = utils.read_loc_merged_data(csv_file_path, date_chosen)
    footprint = data[uuid]
    cluster_input = []
    latlon_list = []

    for idx in range(1, len(footprint)-1):
        T1, lat_t1, lon_t1, dur_t1 = footprint[idx-1]
        T2, lat_t2, lon_t2, dur_t2 = footprint[idx]
        T3, lat_t3, lon_t3, dur_t3 = footprint[idx+1]
        vector1 = np.array([lat_t2-lat_t1, lon_t2-lon_t1])
        vector2 = np.array([lat_t3-lat_t2, lon_t3-lon_t2])
        dot = np.dot(vector1, vector2)/(np.sqrt(np.sum(vector1**2)+epsilon)*(np.sqrt(np.sum(vector2**2)+epsilon)))
        
        if dot <= threshold:
            cluster_input.append([
                lat_t2, 
                lon_t2,
                dot,
                distance.distance((lat_t2, lon_t2), (lat_t1, lon_t1)).m])
            
            latlon_list.append([lat_t2, lon_t2, T2, dur_t2])
    return latlon_list

def jitter(coord):
    # Small amount of lat/long change
    delta = 0.0001
    # Jitter the coordinate value
    new_coord = coord + delta * np.random.uniform(-1, 1)
    return new_coord

def add_home_work_points(m, home_lat, home_lon, work_lat, work_lon, icon_color:str, date:str, plotted_points):
    precision = 7  # for example, this rounds to the nearest 10,000,000th

    home_lat, home_lon = round(home_lat, precision), round(home_lon, precision)
    home_coords = (home_lat, home_lon)

    work_lat, work_lon = round(work_lat, precision), round(work_lon, precision)
    work_coords = (work_lat, work_lon)

    # Check if home coordinates are already plotted, if yes then jitter
    if home_coords in plotted_points:
        home_lat, home_lon = jitter(home_lat), jitter(home_lon)
    else:
        plotted_points.add(home_coords)

    # Add Home location
    folium.Marker(
        location=[home_lat, home_lon],
        popup=folium.Popup(f'{date} Home: {home_coords[0]},{home_coords[1]}', max_width=250),
        tooltip=f'{date} Home',
        icon=folium.Icon(icon='home', color=icon_color)
    ).add_to(m)

    # Check if work coordinates are already plotted, if yes then jitter
    if work_coords in plotted_points:
        work_lat, work_lon = jitter(work_lat), jitter(work_lon)
    else:
        plotted_points.add(work_coords)

    # Add Work location
    folium.Marker(
        location=[work_lat, work_lon],
        popup=folium.Popup(f'{date} Work: {work_coords[0]},{work_coords[1]}', max_width=250),
        tooltip=f'{date} Work',
        icon=folium.Icon(icon='briefcase', color=icon_color)
    ).add_to(m)

    return m, plotted_points

def plot_list_latlon(input_data:list, uuid:str, th_num:float, df_home_work: pd.DataFrame, 
                     color_by_date=False, clustered=False):
    latlon_data = [[item[0], item[1]] for item in input_data]
    avg_coords = [sum(y) / len(y) for y in zip(*latlon_data)]
    m = folium.Map(location=avg_coords, zoom_start=13)

    cluster = MarkerCluster().add_to(m)
    
    if color_by_date:
        # Generate color palette
        unique_dates = sorted(list(set([point[2].strftime('%Y-%m-%d') for point in input_data])))
        colors = sns.color_palette("husl", len(unique_dates)).as_hex()
        
        # Create a dictionary to map dates to colors
        date_to_color = dict(zip(unique_dates, colors))
 
    color1 = [0, 255, 0]  # RGB for green
    color2 = [0, 0, 255]  # RGB for blue

    for i, point in enumerate(input_data):
        # If color by date is True, map date to color
        if color_by_date:
            # Extract date from timestamp
            date = point[2].strftime('%Y-%m-%d')

            # Map date to color
            color_code = date_to_color[date]
        else:
            # Extract hour from timestamp and scale to 255 for grayscale
            timestamp = point[2]
            time_fraction = (timestamp.hour * 60 + timestamp.minute) / (24 * 60)

            # Calculate color gradient
            color_gradient = calculate_color_gradient(time_fraction, color1, color2)
            
            # Convert to hex color code
            color_code = '#{:02x}{:02x}{:02x}'.format(*color_gradient)

        # Create a Marker object with a Circle icon and add it to the cluster

        if clustered:
            folium.Marker(
                location=[point[0], point[1]],
                icon=folium.Icon(color=color_code, icon='circle', icon_color=color_code),
                tooltip="Start Time:" + str(point[2]) + ", Duration:" + str(point[3])
            ).add_to(cluster)

        else:
            folium.CircleMarker(
                location=[point[0], point[1]], 
                radius=float(point[3] / 200),
                tooltip="Start Time:" + str(point[2]) + ", Duration:" + str(point[3]),
                color=color_code,
                fill=True,
                fill_color=color_code,
                fill_opacity=1.0
            ).add_to(m)

    # Add Home and Work locations if they exist
    hw_iterations = {
        '2023-06-22' : 'purple',
        '2023-07-02' : 'green',
        '2023-07-09' : 'beige'
    }

    plotted_points = set()
    for date, color in hw_iterations.items():
        hw_data = df_home_work[(df_home_work['id'] == int(uuid)) & (df_home_work['date'] == date)]
        if len(hw_data) > 0:
            home_lat = hw_data['home_lat'].values[0]
            home_lon = hw_data['home_lon'].values[0]
            work_lat = hw_data['work_lat'].values[0]
            work_lon = hw_data['work_lon'].values[0]
            m,plotted_points = add_home_work_points(m, home_lat, home_lon, work_lat, work_lon, color, date, plotted_points)

    return m

def selet_csv_by_date(csv_file_path:str, date_str:str):
    df = pd.read_csv(csv_file_path) # ex: csv_file_path = 'file_name.csv'
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['date'] = df['start_time'].dt.date
    df = df[df['date'] == pd.to_datetime(date_str).date()] # ex: date_str = '2023-04-06'
    df_result = df.drop(['date'], axis=1)
    df_result.to_csv('../data/' + date_str + 'new_file.csv', index=False)
    return df_result

def calculate_color_gradient(time_fraction, color1, color2):
    # This function calculates the color gradient between two colors (RGB format)
    return [int(color1[i] * (1 - time_fraction) + color2[i] * time_fraction) for i in range(3)]

def render_consined_map_multi_days(csv_path:str, uuid, th_num:float, dates, th_dur:int, clustered:bool):
    id_to_use = str(uuid)
    # filter list_latlon by the specified dates
    list_latlon = [point for point in filter_by_cosine(csv_path, id_to_use, th_num, '')
                   if point[2].date() in dates]
    
    list_latlon = [entry for entry in list_latlon if entry[-1] > th_dur]

    # Get a list of all csv files beginning with 'HW_'
    csv_files = glob.glob('../data/HW_*.csv')

    # Read them in and concatenate them
    df_list = [pd.read_csv(f) for f in csv_files]
    df_hw = pd.concat(df_list, ignore_index=True)

    print("POINT COUNT " +  str(len(list_latlon)))
    m = plot_list_latlon(list_latlon, id_to_use, th_num, df_hw, True, clustered)
    file_name = id_to_use + "__" + "ALL_DAYS" + "___" + str(th_num) + ".html"
    m.save(file_name)
    webbrowser.open('file://' + os.path.realpath(file_name))
    return m

def render_cosined_map_choice(csv_path:str, date_chosen, uuid, th_num:float):
    date_to_use = str(date_chosen)
    id_to_use = str(uuid)
    list_latlon = filter_by_cosine(csv_path, id_to_use, th_num, date_to_use)

    df_hw_1 = pd.read_csv('../data/HW_0626.csv')
    df_hw_2 = pd.read_csv('../data/HW_0702.csv')
    df_hw = pd.concat([df_hw_1, df_hw_2], ignore_index=True)
    
    print("POINT COUNT " +  str(len(list_latlon)))
    m = plot_list_latlon(list_latlon, id_to_use, th_num, df_hw, False)
    file_name = id_to_use + "__" + date_to_use + "___" + str(th_num) + ".html"
    m.save(file_name)
    webbrowser.open('file://' + os.path.realpath(file_name))
    return m

#with Pool() as pool:
#    pool.map(pipeline, list(data))