import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy import stats

def estimateGaussian(X):
    # X = np.matrix(X)
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu ,sigma


def selectThreshold(pval,yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step = (pval.max()-pval.min())/1000
    for epsilon in np.arange(pval.min(),pval.max(),step):
        preds = pval < epsilon
        tp = np.sum(np.logical_and(preds == 1,yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1,yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0,yval == 1)).astype(float)
        precision = tp/(fp+tp)
        recall = tp/(tp+fn)
        f1 = (2*precision*recall)/(precision+recall)
        if f1>best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_f1,best_epsilon

data = loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
# print(Xval.shape,yval.shape)
mu,sigma = estimateGaussian(X)
# print(mu,sigma)
#对训练集中的数据进行概率计算
p = np.zeros((X.shape[0],X.shape[1]))
p[:,0] = stats.norm(mu[0],sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1],sigma[1]).pdf(X[:,1])
p = np.reshape(p[:,0]*p[:,1],(p.shape[0],1))
# print(p.shape)

#对交叉验证集中的数据进行概率计算
pval = np.zeros((Xval.shape[0],Xval.shape[1]))
pval[:,0] = stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])
# print(pval.shape)
# print(pval.max()-pval.min())

pval = np.reshape(pval[:,0] * pval[:,1],(yval.shape))
# print(pval.shape)
# print(yval.shape)
f1,epsilon = selectThreshold(pval,yval)
# print(f1,epsilon)

print(f1,epsilon)
out = np.where(p<epsilon)
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(X[:,0],X[:,1])
ax.scatter(X[out[0],0],X[out[0],1],c = 'red',s = 50)
plt.show()