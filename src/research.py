import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD

data_footprint = utils.read_dmp_data("data/0322-0409_batchE_top.csv")
data_home_and_work = utils.read_home_work_data("data/0322-0409_batchE_HW.csv")

def pipeline(uuid):
    print(uuid)
    from sklearn.cluster import OPTICS
    
    painter = utils.footprint_display()
    footprint = data_footprint[uuid]
    home_work_node = data_home_and_work.query(f'id == "{uuid}"').to_dict(orient='records')[0]

    print("HOME WORK NODE EXISTS FOR : " + uuid)
    print(home_work_node)

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
    painter.plot_map(matrix, f"{uuid}_footprint", fix_map=False, home_work_data=home_work_node)
    painter.plot_map(result, f"{uuid}_PC1", fix_map=False, home_work_data=home_work_node)


with Pool(processes=8) as pool:
    pool.map(pipeline, list(data_footprint))
