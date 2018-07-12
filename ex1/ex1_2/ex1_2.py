import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

path = './ex1data2.txt'
data = pd.read_csv(path,header=None,names=['size','bedrooms','price'])
data = (data - data.mean()) / data.std()
# print(data.head())
# print(len(data))

def computerCost(X,y,theta):
    inner = np.power((X * theta.T - y),2)
    return np.sum(inner) / (2 * len(X))

data.insert(0,'ones',1)
# print(data.head())

cols = data.shape[1]
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols - 1:]
# print(X.head())
# print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
# print(X)
# print(y)

theta = np.matrix(np.array([0,0,0]))

# batch gradient descent 批量梯度下降
def  gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = X * theta.T - y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            # print(len(X))
            temp[0,j] = theta[0,j] - (alpha / len(X)) * np.sum(term)
        theta = temp
        cost[i] = computerCost(X,y,theta)
    return theta,cost

alpha = 0.01
iters = 1000

g,cost = gradientDescent(X,y,theta,alpha,iters)
print(g)

#训练进程（每次迭代得到的cost）
fig = plt.subplot()
fig.plot(np.arange(iters),cost,'r')
fig.set_xlabel('iterations')
fig.set_ylabel('cost')
plt.show()


#正规方程
def normalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@ X.T@y
    return theta

thetaEqn = normalEqn(X,y)
print(thetaEqn)
'''
*，dot的区别
https://www.cnblogs.com/liuhuiwisdom/p/6026369.html
'''