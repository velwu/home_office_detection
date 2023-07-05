'''
跳過PCA, 只用OPTICS clustering進行停留點定位
THE CHOSEN ONE: 510018242552475
'''

import os
import webbrowser
import sys
# os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# sys.path.append("src/")
import utils
import numpy as np
import pandas as pd
import folium
from geopy import distance
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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

def plot_list_latlon(input_data:list, uuid:str, th_num:float, df_home_work: pd.DataFrame):
    latlon_data = [[item[0], item[1]] for item in input_data]
    avg_coords = [sum(y) / len(y) for y in zip(*latlon_data)]
    m = folium.Map(location=avg_coords, zoom_start=13)
    
    color1 = [0, 255, 0]  # RGB for green
    color2 = [0, 0, 255]  # RGB for blue

    for i, point in enumerate(input_data):
        # Extract hour from timestamp and scale to 255 for grayscale
        timestamp = datetime.strptime(str(point[2]), '%Y-%m-%d %H:%M:%S')
        time_fraction = (timestamp.hour * 60 + timestamp.minute) / (24 * 60)
        
        # Calculate color gradient
        color_gradient = calculate_color_gradient(time_fraction, color1, color2)
        
        # Convert to hex color code
        color_code = '#{:02x}{:02x}{:02x}'.format(*color_gradient)

        folium.CircleMarker(
            location=[point[0], point[1]], 
            radius=float(point[3] / 200),
            tooltip="Start Time:" + str(point[2]) + ", Duration:" + str(point[3]),
            color=color_code,
            fill=True,
            fill_color=color_code,
            fill_opacity=1.0
        ).add_to(m)

        # Draw lines between points
        if th_num == 1.0 and i > 0:  # don't try to draw a line for the first point
            folium.PolyLine(
                locations=[input_data[i-1][:2], point[:2]],
                color=color_code
            ).add_to(m)

    # Get Home and Work locations for the specified uuid
    hw_old = df_home_work[(df_home_work['id'] == int(uuid)) & (df_home_work['date'] == '2023-06-22')]
    if len(hw_old) > 0:
        print("HOME-WORK exists: " + str(hw_old))
        home_lat = hw_old['home_lat'].values[0]
        home_lon = hw_old['home_lon'].values[0]
        work_lat = hw_old['work_lat'].values[0]
        work_lon = hw_old['work_lon'].values[0]
        # Add Home and Work locations to the map
        m = add_home_work_points(m, home_lat, home_lon, work_lat, work_lon, 'info-sign')
    else:
        print("NO HOME or WORK location for ID: " + uuid)

    hw_new = df_home_work[(df_home_work['id'] == int(uuid)) & (df_home_work['date'] == '2023-07-02')]
    if len(hw_new) > 0:
        print("HOME-WORK exists: " + str(hw_new))
        home_lat = hw_new['home_lat'].values[0]
        home_lon = hw_new['home_lon'].values[0]
        work_lat = hw_new['work_lat'].values[0]
        work_lon = hw_new['work_lon'].values[0]
        # Add Home and Work locations to the map
        m = add_home_work_points(m, home_lat, home_lon, work_lat, work_lon, 'star')
    else:
        print("NO HOME or WORK location for ID: " + uuid)

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

def render_cosined_map(csv_path, uuid, output_id, th_num):
    list_latlon = filter_by_cosine(csv_path, uuid, th_num, '')
    print("POINT COUNT " +  str(len(list_latlon)))
    m = plot_list_latlon(list_latlon, uuid + "___" + str(th_num), th_num)
    m.save(output_id + "___" + str(th_num) + ".html")
    return m

def render_cosined_map_choice(csv_path:str, date_chosen, uuid, th_num:float):
    date_to_use = str(date_chosen)
    id_to_use = str(uuid)
    list_latlon = filter_by_cosine(csv_path, id_to_use, th_num, date_to_use)

    df_hw_1 = pd.read_csv('../data/HW_0626-0702_before.csv')
    df_hw_2 = pd.read_csv('../data/HW_0626-0702_after.csv')
    df_hw = pd.concat([df_hw_1, df_hw_2], ignore_index=True)
    
    print("POINT COUNT " +  str(len(list_latlon)))
    m = plot_list_latlon(list_latlon, id_to_use, th_num, df_hw)
    file_name = id_to_use + "__" + date_to_use + "___" + str(th_num) + ".html"
    m.save(file_name)
    webbrowser.open('file://' + os.path.realpath(file_name))
    return m

def add_home_work_points(m, home_lat, home_lon, work_lat, work_lon, icon_type):
    # Add Home location to the map as a green star
    folium.Marker(
        location=[home_lat, home_lon],
        popup='Home: ' + str(home_lat) + "," + str(home_lon),
        icon=folium.Icon(icon=icon_type, color='green')
    ).add_to(m)

    # Add Work location to the map as a dark blue star
    folium.Marker(
        location=[work_lat, work_lon],
        popup='Work: ' + str(work_lat) + "," + str(work_lon),
        icon=folium.Icon(icon=icon_type, color='darkblue')
    ).add_to(m)

    return m


#with Pool() as pool:
#    pool.map(pipeline, list(data))