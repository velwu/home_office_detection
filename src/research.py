import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
scaler = StandardScaler(with_std=False)
pca = TruncatedSVD(n_components=1, n_iter=10)
painter = utils.footprint_display()


def pipeline(uuid):
    footprint = data[uuid]
    matrix = utils.footprint2matrix(footprint)

    std_matrix = scaler.fit_transform(matrix)
    pca.fit(std_matrix)
    pc1 = pca.components_[0] + scaler.mean_

    painter.plot_map(footprint, f"{uuid}_footprint", 3, fix_map=False)
    painter.plot_map(utils.matrix2foorprint(pc1), f"{uuid}_PC1", 3, fix_map=False)


with Pool() as pool:
    pool.map(pipeline, list(data))


    