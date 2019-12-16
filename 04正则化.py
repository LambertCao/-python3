import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def Cost(theta,X,y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    fist = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    three = (learingRate/(2*len(X)))*(np.sum(np.power(theta[:,1:theta.shape[1]],2)))
    return (1/len(X))*np.sum(fist-second)+three

def gradientReg(theta,X,y,learningRate):
     theta = np.matrix(theta)
     X = np.matrix(X)
     y = np.matrix(y)
     parameters = int(theta.ravel().shape[1])
     grad = np.zeros(parameters)
     error = sigmoid(X*theta.T) - y
     for i in range(parameters):
        temp = np.multiply(error,X[:,i])
        if i == 0:
            grad[i] = (1/(len(X))) * np.sum(temp)
        else:
            grad[i] = np.sum(temp)/len(X)+(learningRate/len(X))*theta[:,i]
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

data2 = pd.read_csv('ex2data2.txt',header=None,names=['Test1','Test2','result'])
# print(data2.head())
suc = data2[data2['result'].isin([1])]
fail = data2[data2['result'].isin([0])]

plt.scatter(suc['Test1'],suc['Test2'],c='black',marker='+',label = 'y=1')
plt.scatter(fail['Test1'],fail['Test2'],c = 'b',marker = 'o',label = 'y=0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(loc = 'best')
plt.title('Figure 3: Plot of training data')
plt.show()

data2.insert(3,'ones',1)
# print(data2.head())
x1 = data2['Test1']
x2 = data2['Test2']
for i in range(1,5):
    for j in range(0,i):
        data2['F'+str(i)+str(j)] = np.power(x1,i-j)*np.power(x2,j)
data2.drop('Test1',axis=1,inplace=True)
data2.drop('Test2',axis=1,inplace=True)
print(data2.head())

cols = data2.shape[1]
X2 = data2.iloc[:,1:cols]
y = data2.iloc[:,0:1]
X2 = np.array(X2.values)
y = np.array(y.values)
theta = np.zeros(11)
learingRate = 1

# print(Cost(theta=theta,X=X2,y=y,lengd=learingRate))
# print(gradientReg(theta,X2,y,learingRate))
#
result = opt.fmin_tnc(func=Cost,x0=theta,fprime=gradientReg,args=(X2,y,learingRate))
print(result)

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))