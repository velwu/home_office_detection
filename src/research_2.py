import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
from geopy import distance
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
fix_map = False
epi = sys.float_info.epsilon

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
        
        if np.dot([lat_t2-lat_t1, lon_t2-lon_t1], [lat_t3-lat_t2, lon_t3-lon_t2]) <= 0:
            cluster_input.append([
                lat_t2, 
                lon_t2,
                np.dot([lat_t2-lat_t1, lon_t2-lon_t1], [lat_t3-lat_t2, lon_t3-lon_t2]),
                (epi+distance.distance((lat_t2, lon_t2), (lat_t1, lon_t1)).m)])
            
            
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

with Pool() as pool:
    pool.map(pipeline, list(data))