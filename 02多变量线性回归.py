import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#定义一个损失函数
def Cost(X,y,theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

def Gradient_Descent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (alpha*np.sum(term))/len(X)
        theta = temp
        cost[i] = Cost(X,y,theta)
    return theta,cost



if __name__ == '__main__':
    alpha = 0.01
    iters = 1000

    #读取文件
    data = pd.read_csv('ex1data2.txt',header = None,names=['Size','Bedrooms','Price'])
    #特征放缩
    #插入第一列数据，定值为1
    ma = np.matrix(data)

    data = (data-data.mean())/data.std()
    print(data)
    data.insert(0,'ones',1)

    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    theta = np.matrix(np.array([0,0,0]))

    g,cost = Gradient_Descent(X,y,theta,alpha,iters)
    print(Cost(X,y,g))


    plt.plot(np.arange(iters),cost,c = 'r')
    plt.show()


    print(g)
    #假设要预测一个房子的价格值，其房子面积为2014，卧室为4间
    h = g[0,0]*1 + g[0,1]*-0.000857 + g[0,2]*-0.223675
    print(h)