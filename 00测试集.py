import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import imageio

img = imageio.imread('bird_small.png')
print(img)

# X = [1,1]
# theta = np.ones(len(X))
# print(theta)


# X = np.arange(12)
# y = np.ones(12)
# print(X)
# plt.plot(X,y)
# plt.show()

# t = list(str(bin(6)))
# listt = t[2:]
# intlistt = [int(j) for j in listt]
# print(sum(intlistt))

# print(help(plt.scatter))
# print(np.array([1.0,2.0,1.0]))
s = np.array([[3,2],[6,4],[7,8]])
s = np.matrix(s)
print(sum(s))