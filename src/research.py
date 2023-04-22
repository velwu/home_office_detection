import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
from geopy import distance
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
fix_map = False

def pipeline(uuid):
    print(uuid)
    from sklearn.cluster import OPTICS
    
    painter = utils.footprint_display()
    footprint = data[uuid]

    # conduct PCA
    matrix = utils.footprint2matrix(footprint)
    pca = TruncatedSVD(n_components=2, n_iter=10)
    
    W = pca.fit_transform(matrix)
    H = pca.components_

    # weight clustering, and get average weights of each cluster
    cluster = OPTICS().fit(W)
    labels_array = cluster.labels_
    labels_set = set([label for label in labels_array if label >= 0])
    PC_weight_mean_array = np.array([np.mean(W[np.where(labels_array==label)],axis=0) for label in labels_set])
    
    cluster_data = {
        'x':W[:,0].tolist(), 
        'y':W[:,1].tolist(), 
        'label':labels_array.tolist()}

    utils.weight_plot(cluster_data, uuid, f"eigen value: {pca.explained_variance_ratio_}")

    # each cluster represents a potential routing track
    result = np.dot(PC_weight_mean_array, H)
    painter.plot_gif(matrix, f"{uuid}_footprint", fix_map=fix_map)
    painter.plot_gif(result, f"{uuid}_PC1", fix_map=fix_map)

    # figure out where home/office is
    '''
    latlon_list = []
    cluster_input = []
    for row in range(result.shape[0]):
        for col in range(1, 96):
            previous_lat, previous_lon = result[row, col-1], result[row, col+95]
            lat, lon = result[row, col], result[row, col+96]
            latlon_list.append([lat, lon])
            cluster_input.append([lat, lon, distance.distance((lat, lon), (previous_lat, previous_lon)).m])


    cluster = OPTICS().fit(np.array(cluster_input))
    group = cluster.labels_
    painter.plot_map(latlon_list, group, f"{uuid}_display", fix_map=fix_map)
    '''
    
with Pool() as pool:
    pool.map(pipeline, list(data))


