import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from IPython.display import Image


def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V
def projectData(X,U,K):
    X = np.matrix(X)
    U = np.matrix(U)
    Ureduce = U[:,0:K]
    result = X*Ureduce
    return result
def recover_data(Z,U,K):
    U_reduced = U[:,:K]
    return Z*U_reduced.T
data = loadmat('ex7data1.mat')
# print(data)
X = data['X']
# print(data)
plt.scatter(X[:,0],X[:,1],s=30)


# print(X)
U,S,V = pca(X)
# print(X.shape)
# print(U.shape)
# print(S.shape)
# print(V.shape)

result = projectData(X=X,U=U,K=1)
# print(result)
recover = recover_data(Z=result,U=U,K=1)
print(recover)
plt.scatter(list(recover[:,0]),list(recover[:,1]))
plt.show()