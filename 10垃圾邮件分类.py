import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn import svm


dataTest = loadmat('spamTest.mat')
dataTrain = loadmat('spamTrain.mat')
# print(dataTrain)
Xtest = dataTest['Xtest']
ytest = dataTest['ytest']
Xtrain = dataTrain['X']
ytrain = dataTrain['y']
# print(Xtest.shape,ytest.shape,Xtrain.shape,ytrain.shape)

svc = svm.SVC()
svc.fit(Xtrain,ytrain)
score = svc.score(Xtest,ytest)
print('Training accuracy = {0}%'.format(np.round(score*100,2)))