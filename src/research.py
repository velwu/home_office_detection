import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler


data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")

def pipeline(uuid):
    print(uuid)
    from sklearn.cluster import OPTICS
    
    painter = utils.footprint_display()
    footprint = data[uuid]

    matrix = utils.footprint2matrix(footprint)
    pca = TruncatedSVD(n_components=matrix.shape[0]-1, n_iter=10)
    
    W = pca.fit_transform(matrix)
    H = pca.components_

    cluster = OPTICS().fit(W)
    labels_array = cluster.labels_
    PC_weight_mean_array = np.array([np.mean(W[np.where(labels_array==label)],axis=0) for label in set(labels_array.tolist())])
    
    cluster_data = {
        'x':W[:,0].tolist(), 
        'y':W[:,1].tolist(), 
        'label':labels_array.tolist()}

    utils.weight_plot(cluster_data, uuid, f"eigen value: {pca.explained_variance_ratio_}")

    result = np.dot(PC_weight_mean_array, H)
    painter.plot_map(matrix, f"{uuid}_footprint", fix_map=False)
    painter.plot_map(result, f"{uuid}_PC1", fix_map=False)


with Pool() as pool:
    pool.map(pipeline, list(data))

