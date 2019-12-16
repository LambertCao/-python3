import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

#没有正则化的损失函数
def cost(params,Y,R,num_features,learningRate):

    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies*num_features],(num_movies,num_features)))
    theta = np.matrix(np.reshape(params[num_users*num_features:],(num_users,num_features)))
    first1 = np.multiply(X*theta.T-Y,R)
    second1 = np.power(first1,2)
    result = np.sum(second1)/2 + (learningRate/2)*(np.sum(np.power(theta,2))+np.sum(np.power(X,2)))
    return result

def cofiCostFunc(params,Y,R,num_features,learningRate):

    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies*num_features],(num_movies,num_features)))
    theta = np.matrix(np.reshape(params[num_movies*num_features:],(num_users,num_features)))
    first1 = np.multiply(X * theta.T - Y, R)
    second1 = np.power(first1, 2)
    result = np.sum(second1) / 2 + (learningRate / 2) * (np.sum(np.power(theta, 2)) + np.sum(np.power(X, 2)))
    first = np.multiply(X*theta.T-Y,R)
    second = first*theta
    grad_x = second + learningRate*X
    tree = first.T*X
    grad_theta = tree + learningRate*theta
    grad = np.concatenate((np.ravel(grad_x),np.ravel(grad_theta)))
    return result,grad

movies_data = loadmat('ex8_movies.mat')
# print(movies_data)
Y = movies_data['Y']
R = movies_data['R']

movie_idx = {}
with open('movie_ids.txt','r',encoding = 'gbk') as f:
    for line in f:
        tokens = line.split(' ')
        # print(tokens[1:])
        movie_idx[int(tokens[0])-1] = ' '.join(tokens[1:])
        # print(movie_idx)
        # break
    # print(movie_idx[0])

ratings = np.zeros((1682,1))
R = np.append(R,ratings != 0,axis=1)
Y = np.append(Y,ratings,axis=1)
# print(R)
features = 10
learningRate = 10
movies = Y.shape[0]
users = Y.shape[1]
X = np.random.random(size = (movies,features))
theta = np.random.random(size = (users,features))
params = np.concatenate((np.ravel(X),np.ravel(theta)))
print(X.shape,theta.shape,params.shape)
Ymeans = np.zeros((movies,1))
Ynorm = np.zeros((movies,users))
for i in range(movies):
    idx = np.where(R[i,:]==1)[0]
    Ymeans[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymeans[i]
# print(Ynorm.mean())
fmin = minimize(fun=cofiCostFunc,x0=params,args=(Ynorm,R,features,learningRate),method='CG',jac=True,options={'maxiter':1000})
# print(fmin)
X = np.reshape(fmin.x[:movies*features],(movies,features))
theta = np.reshape(fmin.x[movies*features:],(users,features))
print(X.shape,theta.shape)
X = np.matrix(X)
theta = np.matrix(theta)
# print(Y)
predictions = X*theta.T
mypredict = predictions[:,-1] + Ymeans
# print(mypredict)
idx = np.argsort(mypredict,axis=0)[::-1]
# print(idx)
print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(mypredict[j])), movie_idx[j]))