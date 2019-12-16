import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def computeCost(X,y,theta):
    inner = np.power(((X*theta.T)- y ),2)
    return np.sum(inner)/(2*len(X))
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T) - y

        for j in range(parameters):
            term =  np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (alpha/len(X))*np.sum(term)
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta ,cost


if __name__ == '__main__':
    alpha = 0.01
    iters = 1000
    data = pd.read_csv('ex1data1.txt', header=None, names=['Polution', 'Profit'])
    data.insert(0, 'ones', 1)
    # print(data.head())
    # print(data.describe())

    # plt.scatter(x=data['Polution'], y=data['Profit'],label = 'Traing Data')
    # plt.xlabel('Population of City in 10,000s')
    # plt.ylabel('Profit in $10,000s')
    # plt.title('Figure 1: Scatter plot of training data')

    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]
    # print(X.head())

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    # print(X)
    theta = np.matrix(np.array([0, 0]))
    # print(theta.ravel())
    g,cost = gradientDescent(X,y,theta,alpha,iters)
    # print(g)
    # print(cost)
    # plt.subplot(211)
    # x = np.linspace(data['Polution'].min(),data['Polution'].max(),100)
    # f = g[0,0] + g[0,1]*x
    # plt.plot(x,f,c = 'coral',label = 'Prediction')
    # plt.scatter(data['Polution'],data['Profit'],label = 'Traning Data')
    # plt.xlabel('Population')
    # plt.ylabel('Profit')
    # plt.title('Predicted Profit vs. Population Size')
    # plt.legend(loc = 'best')

    # plt.subplot(212)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(np.arange(iters),cost,c= 'r',label = r'损失函数值')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')
    plt.legend(loc = 'best')
    plt.show()
