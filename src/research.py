import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")

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
    painter.plot_map(matrix, f"{uuid}_footprint", fix_map=False)
    painter.plot_map(result, f"{uuid}_PC1", fix_map=False)


with Pool() as pool:
    pool.map(pipeline, list(data))

