'''
跳過PCA, 只用OPTICS clustering進行停留點定位
THE CHOSEN ONE: 510018242552475
'''

import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
import folium
from geopy import distance
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

# data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
data = utils.read_dmp_data("data/0322-0409_batchE_top.csv")
fix_map = False
epsilon = sys.float_info.epsilon

def pipeline(uuid):
    print(uuid)
    from sklearn.cluster import OPTICS

    painter = utils.footprint_display()
    footprint = data[uuid]
    matrix = utils.footprint2matrix(footprint)
    cluster_input = []
    latlon_list = []

    for idx in range(1, len(footprint)-1):
        T1, lat_t1, lon_t1 = footprint[idx-1]
        T2, lat_t2, lon_t2 = footprint[idx]
        T3, lat_t3, lon_t3 = footprint[idx+1]
        vector1 = np.array([lat_t2-lat_t1, lon_t2-lon_t1])
        vector2 = np.array([lat_t3-lat_t2, lon_t3-lon_t2])
        dot = np.dot(vector1, vector2)/(np.sqrt(np.sum(vector1**2)+epsilon)*(np.sqrt(np.sum(vector2**2)+epsilon)))
        
        if dot <= -0.8:
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

def pipeline_labbing(uuid:str, threshold:float):
    print(uuid)
    from sklearn.cluster import OPTICS

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
            
            latlon_list.append([
                lat_t2, 
                lon_t2])
    return latlon_list

def plot_list_latlon(latlon_data:list, pic_name:str):
    avg_coords = [sum(y) / len(y) for y in zip(*latlon_data)]
    m = folium.Map(location=avg_coords, zoom_start=13)
    for point in latlon_data:
        folium.Marker(location=point).add_to(m)
          
    m.save(pic_name + '_map.html')
    return m

#with Pool() as pool:
#    pool.map(pipeline, list(data))