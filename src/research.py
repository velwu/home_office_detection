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
from sklearn.cluster import KMeans

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")

def pipeline(uuid):
    pca = TruncatedSVD(n_components=2, n_iter=10)
    # pca = PCA()
    painter = utils.footprint_display()
    
    footprint = data[uuid]
    matrix = utils.footprint2matrix(footprint)

    W = pca.fit_transform(matrix)
    H = pca.components_

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

    plt.plot()
    sns.scatterplot(data=cluster_data,x='x',y='y',hue='label')
    plt.text(
        min(W[:,0].tolist()),
        min(W[:,1].tolist()),
        f"eigen value: {pca.explained_variance_ratio_[:5]}",
        fontdict={'size':10, 'color':'red'})
    plt.savefig(f"display/weight/{uuid}.png")
    plt.close()

    # painter.plot_map(matrix, f"{uuid}_footprint", fix_map=False)
    # painter.plot_map(pc1, f"{uuid}_PC1", fix_map=False)


with Pool() as pool:
    pool.map(pipeline, list(data))

