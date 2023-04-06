import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
scaler = StandardScaler(with_std=False)
pca = TruncatedSVD(n_components=1, n_iter=10)


for uuid, footprint in data.items():
    matrix = utils.convert2matrix(footprint)

    std_matrix = scaler.fit_transform(matrix)
    pca.fit(std_matrix)
    pc1 = pca.components_[0] + scaler.mean_

    