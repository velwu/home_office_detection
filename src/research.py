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
    from sklearn.cluster import KMeans
    
    painter = utils.footprint_display()
    footprint = data[uuid]

    pca = TruncatedSVD(n_components=len(footprint)-1, n_iter=10)
    matrix = utils.footprint2matrix(footprint)

    W = pca.fit_transform(matrix)
    H = pca.components_
    qualified = True

    cluster = KMeans(n_clusters=2, n_init=20).fit(W)
    labels_array = cluster.labels_
    label_to_pc1_weight_dict = {label:np.sum(W[np.where(labels_array==label),0]) for label in set(labels_array.tolist())}
    label_pc1_weight_list = [[label, weight_sum] for label, weight_sum in label_to_pc1_weight_dict.items()]
    label_pc1_weight_list.sort(reverse=True, key=lambda x:x[1])
    label = label_pc1_weight_list[0][0]

    label_list = labels_array==label
    cluster_data = {
        'x':W[:,0].tolist(), 
        'y':W[:,1].tolist(), 
        'label':label_list.tolist()}

    utils.weight_plot(cluster_data, uuid, f"eigen value: {pca.explained_variance_ratio_}", qualified)

    # result = np.dot(np.mean(W[labels_array==label],axis=0).reshape(1,-1), H)
    painter.plot_map(matrix, f"{uuid}_footprint", fix_map=False, qualified=qualified)
    # painter.plot_map(result, f"{uuid}_PC1", fix_map=False, qualified=qualified)


with Pool() as pool:
    pool.map(pipeline, list(data))

