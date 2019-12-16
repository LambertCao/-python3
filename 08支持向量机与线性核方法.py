import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm


row_data = loadmat('ex6data1.mat')
# print(data)
data = pd.DataFrame(row_data['X'],columns=['X1','X2'])
data['y'] = row_data['y']
# print(data)
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
plt.scatter(positive['X1'],positive['X2'],marker='+',c='black',label = 'Positive')
plt.scatter(negative['X1'],negative['X2'],marker='o',c='red',label = 'Negative')
plt.legend(loc = 'best')
# plt.show()

svc = svm.LinearSVC(C = 1,loss='hinge',max_iter=1000)
# print(svc)
svc.fit(data[['X1','X2']],data['y'])
w = svc.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(0,5)
yy = a*xx-(svc.intercept_[0])/w[1]
# print(svc.score(data[['X1','X2']],data['y']))
data['SVM 1 Confidence'] = svc.decision_function(data[['X1','X2']])
# print(data['SVM 1 Confidence'])

plt.plot(xx,yy)
plt.scatter(data['X1'],data['X2'],s = 50,c = data['SVM 1 Confidence'],cmap='seismic')
plt.title('SVM (C=1) Decision Confidence')
plt.show()


