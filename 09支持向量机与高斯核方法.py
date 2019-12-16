import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from scipy.io import loadmat


def gaussian_kernel(X1,X2,sigma):
    fist_term = np.sum(np.power((X1-X2),2))
    second_term = -2*np.power(sigma,2)
    return np.exp(fist_term/second_term)

#对高斯核进行测试
# X1 = np.array([1.0,2.0,1.0])
# X2 = np.array([0.0,4.0,-1.0])
# sigma = 2
# print(gaussian_kernel(X1,X2,sigma))

#导入数据
# row_data = loadmat('ex6data2.mat')
# # print(data)
# data = pd.DataFrame(row_data['X'],columns=['X1','X2'])
# data['y'] = row_data['y']
# positive = data[data['y'].isin([1])]
# negative = data[data['y'].isin([0])]
# plt.scatter(positive['X1'],positive['X2'],marker='+',label = 'positive')
# plt.scatter(negative['X1'],negative['X2'],marker='o',label = 'negative')
# plt.legend(loc = 'best')
# # plt.show()
#
# svc = svm.SVC(C=100,gamma=50,probability=True)
# svc.fit(data[['X1','X2']],data['y'])
# print(svc.score(data[['X1','X2']],data['y']))
# data['SVM 1 Confidence'] = svc.decision_function(data[['X1','X2']])
# plt.scatter(data['X1'],data['X2'],s = 30,c = data['SVM 1 Confidence'],cmap='seismic')
# plt.show()
#
# data['Predict']= svc.predict_proba(data[['X1','X2']])[:,0]
# plt.scatter(data['X1'],data['X2'],c=data['Predict'],cmap='Reds')
# plt.show()

row_data = loadmat('ex6data3.mat')
# print(row_data)
X = row_data['X']
Xval = row_data['Xval']
y = row_data['y'].ravel()
yval = row_data['yval'].ravel()
C = [0.01,0.03,0.1,0.3,1,3,10,30,100]
gamma = [0.01,0.03,0.1,0.3,1,3,10,30,100]
best_score = 0
best_params = {'C':None,'gamma':None}
for c in C:
    for g in gamma:
        svc = svm.SVC(C=c,gamma=g,probability=True)
        svc.fit(X,y)
        score = svc.score(Xval,yval)
        if score > best_score:
            best_score = score
            best_params['C'] = c
            best_params['gamma']  = g
print(best_score,best_params)

