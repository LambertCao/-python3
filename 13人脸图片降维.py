import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from IPython.display import Image
from scipy.io import loadmat

def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
def pca(X):
    X = np.matrix(X)
    m,n = X.shape
    X = (X-X.mean())/X.std()
    sigma = (X.T*X)/m
    U,S,V = np.linalg.svd(sigma)

    return U,S,V
def project_data(X,U,K):
    X = np.matrix(X)
    U = np.matrix(U)
    Z = X*U[:,0:K]
    return Z
def recover_data(Z,U,K):
    return Z*(U[:,0:K].T)

data = loadmat('ex7faces.mat')
X = data['X']
X1 = np.reshape(X[0,:],(32,32))
plt.imshow(X1)
# plt.show()
# T = X.reshape((5000,32,32))
# plt.imshow(T[2])
# plt.show()
U,S,V = pca(X)
# print(U.shape)
Z = project_data(X,U,K=100)
X_recover = recover_data(Z,U,100)
# print(X_recover.shape)
face = np.reshape(X[0,:],(32,32))
plt.imshow(face)

plot_n_image(X,5000)
plt.show()