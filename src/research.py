import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
from geopy import distance
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

    # utils.weight_plot(cluster_data, uuid, f"eigen value: {pca.explained_variance_ratio_}")

    # each cluster represents a potential routing track
    result = np.dot(PC_weight_mean_array, H)

    # figure out where home/office is
    # '''
    # https://www.nature.com/articles/s41598-021-88822-3
    latlon_list = []
    cluster_input = []
    for row in range(result.shape[0]):
        for col in range(1, 96-1):
            lat, lon = result[row, col], result[row, col+96]
            lat_prev, lon_prev = result[row, col-1], result[row, col+95]
            lat_next, lon_next = result[row, col+1], result[row, col+97]
            vector1 = [lat-lat_prev, lon-lon_prev]
            vector2 = [lat_next-lat, lon_next-lon]

            latlon_list.append([lat, lon])
            cluster_input.append([
                lat, 
                lon,
                np.dot(vector1, vector2),
                distance.distance((lat, lon), (lat_prev, lon_prev)).m]) 
            # cluster_input.append([
            #     lat, 
            #     lon, 
            #     np.dot(vector1, vector2),
            #     distance.distance((lat, lon), (lat_prev, lon_prev)).m])

    # distance_list = [i[2] for i in cluster_input]
    # plt.plot()
    # sns.kdeplot(data={'distance':distance_list}, x='distance')
    # plt.savefig(f"display/footprint/{uuid}_kde.png")
    # plt.close()

    scaler = StandardScaler(with_mean=False, with_std=True)
    cluster = OPTICS(min_samples=25).fit(scaler.fit_transform(np.array(cluster_input)))
    group = [str(i) for i in cluster.labels_]
    centers = [[
        np.mean([latlon_list[i][0] for i in range(len(cluster.labels_)) if cluster.labels_[i]==level]),
        np.mean([latlon_list[i][1] for i in range(len(cluster.labels_)) if cluster.labels_[i]==level])]
        for level in set(cluster.labels_) if level >=0]
    
    # '''
    # painter.plot_gif(matrix, f"{uuid}_footprint", fix_map=fix_map)
    painter.plot_gif(result, f"{uuid}_PC1", centers=centers, fix_map=fix_map)
    painter.plot_map(latlon_list, group, f"{uuid}_display", centers=centers, fix_map=fix_map)
    
with Pool() as pool:
    pool.map(pipeline, list(data))


