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
    cluster_input = []
    latlon_list = []

    for idx in range(1, len(footprint)):
        T1, lat_t1, lon_t1 = footprint[idx-1]
        T2, lat_t2, lon_t2 = footprint[idx]
        
        cluster_input.append([
            lat_t2, 
            lon_t2, 
            (T2-T1).total_seconds()/(epi+distance.distance((lat_t2, lon_t2), (lat_t1, lon_t1)).m)])
        
        latlon_list.append([
            lat_t2, 
            lon_t2])
    
    scaler = StandardScaler(with_mean=False, with_std=True)
    cluster = OPTICS(min_samples=int(len(footprint)/10)).fit(scaler.fit_transform(np.array(cluster_input)))
    group = [str(i) for i in cluster.labels_]
    painter.plot_map(latlon_list, group, f"{uuid}_display", fix_map=fix_map)

with Pool() as pool:
    pool.map(pipeline, list(data))