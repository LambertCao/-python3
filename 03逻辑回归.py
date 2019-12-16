import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

def sigmoid(z):
    return 1/(1+np.exp(-z))

# x = np.arange(-10,10,step = 1)
# plt.plot(x,sigmoid(x),c = 'r')
# plt.show()

# def cost(theta,X,y):
#     z = X.dot(theta.T)
#     h = sigmoid(z)
#     inner1 = np.multiply(y,np.log(h))
#     inner2 = np.multiply(1-y,np.log(1-h))
#     return (-1/len(X))*np.sum(inner1+inner2)
def cost(theta, X, y):
    theta = np.matrix(theta)

    first = np.multiply(-y, np.log(sigmoid(X.dot(theta.T))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta.T))))
    return np.sum(first - second) / (len(X))

def gradient(theta,X,y):
    theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X.dot(theta.T)) - y

    for i in range(parameters):
        inner = np.multiply(error,X[:,i])
        grad[i] = np.sum(inner) / len(X)
    return grad

def predict(theta,X):
    t = []
    inner = sigmoid(X.dot(theta.T))
    for i in inner:
        if i >=0.5:
            t.append(1)
        else:
            t.append(0)
    return t


data = pd.read_csv('ex2data1.txt',names = ['Exam1','Exam2','Admitted'],header=None)
# print(data.head())

data.insert(0,'ones',1)
#分别选出Admitted为0或为1的点
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

plt.scatter(positive['Exam1'],positive['Exam2'],c='r',label = '1',marker = 'o')
plt.scatter(negative['Exam1'],negative['Exam2'],c='b',label = '0',marker = 'x')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend(loc = 'best')
# plt.show()


cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)
# print(X)
# print(y)

X = np.matrix(X)
y = np.matrix(y)
theta = np.zeros(3)
theta = np.matrix(theta)
# print(theta)
# print(theta)
# print(X)
# print(y)
# result = costfunction(theta,X,y)
# print(result)
result = opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))
print(result)

#准确率分析
corrent = []
theta_min = np.matrix(result[0])


plt.plot(data['Exam1'],(-1*theta_min[0,0]-theta_min[0,1]*data['Exam1'])/theta_min[0,2])
plt.show()
# predictions = predict(theta_min,X)
# for (a,b) in zip(predictions,y):
#     if a == b:
#         corrent.append(1)
#     else:
#         corrent.append(0)
# accuracy = (sum(map(int, corrent)) % len(corrent))
# print('accuracy:{0}%'.format(accuracy))
#
#
# #实行预测
# a = [1,80,70]
# print(predict(theta=theta_min,X=np.matrix(a)))