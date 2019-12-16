import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

def linerRegCostFunction(theta,X,y,learningrate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    X = np.insert(X,0,values=np.ones(m),axis=1)
    fist_term = np.multiply(1/(2*m),np.sum(np.power((X*theta.T - y),2)))
    second_term = np.multiply(learningrate/(2*m),np.sum(np.power(theta[:,1:],2)))
    cost = fist_term - second_term

    #计算损失梯度
    params = int(theta.shape[1])
    grad = np.zeros(params)
    error = X*theta.T -y
    # temp = []
    for i in range(params):

        temp = np.multiply(error,X[:,i])
        # print(temp.shape)
        if i == 0:
            grad[i] = (np.multiply(1/m,np.sum(temp)))
        else:
            grad[i] = (np.multiply(1/m,np.sum(temp)) + (learningrate * theta[:,i])/m)


    return cost,grad

def trainLinearReg(theta,X,y,learningrate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    min = minimize(fun=linerRegCostFunction,x0=theta,args=(X,y,learningrate),method='TNC',jac=True,options={'disp':True})
    return min
def learningCurve(X,y,Xval,yval,theta,learningrate):
    Jtrain = []
    Jval = []
    for i in range(1,13):
        min = trainLinearReg(theta=theta,X=X[:i,:],y=y[:i],learningrate=learningrate)
        cost,grad = linerRegCostFunction(theta=min.x,X = X[:i,:],y = y[:i],learningrate=learningrate)
        Jtrain.append(cost)
        costval,gradval = linerRegCostFunction(theta=min.x,X = Xval,y = yval,learningrate=learningrate)
        Jval.append(costval)
    plt.plot(np.arange(1,13),Jtrain,label = 'Train')
    plt.plot(np.arange(1,13),Jval,label = 'Cross Validation')
    plt.xlabel('Number of training example')
    plt.ylabel('Error')
    plt.title('Learning curve for linear regression')
    plt.legend(loc = 'best')
    plt.show()

def polyFeatures(X,Xval,Xtest,p):
    X = np.matrix(X)
    # y = np.matrix(y)
    Xval = np.matrix(Xval)
    # yval = np.matrix(yval)
    Xtest = np.matrix(Xtest)
    # ytest = np.matrix(ytest)
    for i in range(2,p+1):
        X = np.column_stack((X,np.power(X[:,0],i)))
        # y = np.column_stack((y,np.power(y[:,0],i)))
        Xval = np.column_stack((Xval,np.power(Xval[:,0],i)))
        # yval = np.column_stack((yval,np.power(yval[:,0],i)))
        Xtest = np.column_stack((Xtest,np.power(Xtest[:,0],i)))
        # ytest = np.column_stack((ytest,np.power(ytest[:,0],i)))
    return X,Xval,Xtest



data = loadmat('ex5data1.mat')
# print(data)
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# cost,grad = linerRegCostFunction(theta=[1,1],X = X,y = y,learningrate=1)
# print(cost)
# print(grad)
# min = trainLinearReg(theta=[1,1],X = X,y = y,learningrate=1)
# print(min)

# plt.scatter(X,y,c = 'red',marker='x')
# plt.plot(X,min.x[0]+X*min.x[1])
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
# plt.title('Figure 1: Data')
# plt.show()
# print(X.shape,y.shape,Xval.shape,yval.shape,Xtest.shape,ytest.shape)
X1,Xval1,Xtest1 = polyFeatures(X = X,Xval=Xval,Xtest=Xtest,p=8)
theta = np.random.random(size=X1.shape[1]+1)
learningCurve(X = X,y = y,Xval=Xval,yval = yval,theta=np.random.random(size=X.shape[1]+1),learningrate=1)



# min = trainLinearReg(theta=theta,X=X1,y=y1,learningrate=0)


# plt.scatter(X,y,c = 'red',marker='x')
# X2 = np.insert(X1,0,values=np.ones(len(X1)),axis=1)
# X2 = np.matrix(X2)
# # print(X2*(theta_min).T)
# # plt.plot(X,min.x[0]+min.x[1]*np.power(X,1)+min.x[2]*np.power(X,2)+min.x[3]*np.power(X,3)+min.x[4]*np.power(X,4)+min.x[5]*np.power(X,5)+min.x[6]*np.power(X,6)+min.x[7]*np.power(X,7)+min.x[8]*np.power(X,8))
# plt.xlabel('Change in water level (x)')
# plt.ylabel('Water flowing out of the dam (y)')
learningCurve(X = X1,y=y,Xval=Xval1,yval=yval,theta=theta,learningrate=1)
# print(X)
# print(X1[:3,1])






