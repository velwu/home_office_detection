import random
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.preprocessing import StandardScaler

# parameters
mode = 'TruncatedSVD'


data = np.array(
    [[i+random.gauss(mu=0.0, sigma=0.1) for i in range(5,-6,-1)]+[j/10+random.gauss(mu=0.0, sigma=0.01) for j in range(5,-6,-1)]]+
    [[i+random.gauss(mu=0.0, sigma=0.1) for i in range(-2,9)]+
    [j/10+random.gauss(mu=0.0, sigma=0.01) for j in range(-5,6)] for k in range(3)])


if mode == 'TruncatedSVD': 
    pca = TruncatedSVD(n_components=1, n_iter=10)
    W = pca.fit_transform(data)
    H = pca.components_
    result = np.dot(W, H)

elif mode == 'pca':
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(data)

    pca = PCA(n_components=2, n_iter=10)
    W = pca.fit_transform(data)
    H = pca.components_
    result = scaler.inverse_transform(np.dot(W, H))

else:
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(data)

    pca = NMF(n_components=2)
    W = pca.fit_transform(data+10)
    H = pca.components_
    result = H

print('original data\n', data, '\n')
print('result\n', result, '\n')
