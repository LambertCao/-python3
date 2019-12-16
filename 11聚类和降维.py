#k均值聚类算法
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy.io import loadmat
from IPython.display import Image
from skimage import io
from sklearn.cluster import KMeans


#对聚类中心初始化
def init_centroids(X,k):
    m,n = X.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0,m,k)
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    return centroids

#对数据进行最近点分类
def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx

def computeCentroids(X,centroids,K):
    n = X.shape[1]
    new_centroids = []
    idx = find_closest_centroids(X,centroids)
    for k in range(K):
        idx_k = np.where(idx == k)
        new_centroids.append(sum((X[idx_k]))/len(X[idx_k]))
    new_centroids = np.reshape(new_centroids,(K,n))
    return idx,new_centroids

def run_k_mean(X,centroids,K,max_iter):
    for i in range(max_iter):
       idx,centroids = computeCentroids(X,centroids,K)
    return idx,centroids

#用sklearn进行k均值聚类
def sk_kmeans(K,X):
    X =  np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))/255
    Kmea = KMeans(n_clusters=16,n_init=100,n_jobs=-1)
    kmea = Kmea.fit(X)
    centroids = kmea.cluster_centers_
    C= kmea.predict(X)
    compresses_pic = centroids[C].reshape((128,128,3))
    plt.imshow(X)
    plt.imshow(compresses_pic)
    plt.show()
    # fig,ax = plt.subplots(2,1)
    # ax[0].imshow(X)
    # ax[1].imshow(compresses_pic)
    # plt.show()
#以下为对数据进行聚类操作
# data = loadmat('ex7data2.mat')
# # print(data)
# X = data['X']
# centroids = np.array([[3,3],[6,2],[8,5]])
# # idx = find_closest_centroids(data['X'],centroids)
# # print(idx)
# # tt,cc = computeCentroids(data['X'],centroids,3)
# # print(tt,cc)
# idx,centroids = run_k_mean(data['X'],centroids,3,10)
# cluster1 = X[np.where(idx==0)]
# cluster2 = X[np.where(idx==1)]
# cluster3 = X[np.where(idx==2)]
# # print(cluster3)
# # print(cluster1)
# data = pd.DataFrame(data['X'],columns=['X1','X2'])
# # print(data)
# sb.set(context='notebook',style = 'white')
# sb.lmplot('X1','X2',data=data,fit_reg=False)
# plt.show()
# plt.scatter(cluster1[:,0],cluster1[:,1],s=30,color = 'r',label = 'Cluster 1')
# plt.scatter(cluster2[:,0],cluster2[:,1],s = 30,color = 'g',label = 'Cluster 2')
# plt.scatter(cluster3[:,0],cluster3[:,1],s=30,color = 'b',label = 'Cluster 3')
# plt.legend(loc = 'best')
# plt.show()

#以下为对图像进行压缩操作
image = Image(filename='bird_small.png')
image_data = loadmat('bird_small.mat')
# plt.imshow(image_data['A'])
# plt.show()
# print(image_data)
# print(image_data['A'].shape)
A = image_data['A']
# print(A.shape)
A = A/255
# print(A)
# print("----------------------------------")
T = A.shape
A = np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
centroids = init_centroids(A,16)
# print(centroids)
idx,centroids = run_k_mean(X=A,centroids=centroids,K=16,max_iter=10)
# print(idx.shape)
# print(centroids)
X_recovered = centroids[idx.astype(int),:]


X_recovered = np.reshape(X_recovered,(T[0],T[1],T[2]))
# print(X_recovered.shape)
# plt.imshow(X_recovered)
# plt.show()


#调用sklearn对图片进行降维
sk_kmeans(K=16,X=image_data['A'])
