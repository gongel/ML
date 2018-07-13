import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = './ex2data1.txt'
data = pd.read_csv(path,header=None,names=['e1','e2','admitted'])
# print(data.head())

positive = data[data['admitted'].isin([1])] #data的admitted行是否包括1
negative = data[data['admitted'].isin([0])]#

plt.scatter(positive['e1'],positive['e2'],s = 50,c = 'b',marker='x',label='admitted')
plt.scatter(negative['e1'],negative['e2'], s=50, c='r',marker='o',label='not admitted')
plt.xlabel('e1')
plt.ylabel('e2')
# plt.show()

#sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#just for test sigmoid
nums = np.arange(-10,10,step=1)
plt.plot(nums,sigmoid(nums),'r-')
# plt.show()

def cost(theta, X, y):
    X =np.matrix(X)
    y=np.matrix(y)
    theta=np.matrix(theta)
    first = np.multiply(-y,np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    return np.sum(first-second)/len(X)

data.insert(0,'ones',1)
cols=data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:]

#convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
# print(X.shape)
# print(y.shape)
# print(theta.shape)
# print(cost(theta,X,y))

def gradient1(theta,X,y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    return (1/len(X))*X.T@(sigmoid(X@theta.T)-y)#这一步会在本篇博文中以图片形式解释：使用向量化的实现，可以把所有这些 n个参数同时更新
print('gradent1',gradient1(theta,X,y))
# '''
# [[ -0.1       ]
#  [-12.00921659]
#  [-11.26284221]]
#  '''

#或者按照线性回归的梯度
# 来计算梯度下降
def gradient2(theta,X,y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters=int(theta.shape[1])
    grad = np.matrix(np.zeros(parameters))
    error = sigmoid(X@theta.T)-y
    for j in range(parameters):
        term=np.multiply(error,X[:,j])
        grad[0,j]=np.sum(term)/len(X)
    return grad
print('gradient2',gradient2(theta,X,y))
'''
[[ -0.1        -12.00921659 -11.26284221]]
'''


#使用scipy.optimize.minimize去寻找最优参数（拟合参数）
import scipy.optimize as opt
res=opt.fmin_tnc(func=cost,x0=(theta),args=(X,y),fprime=gradient2)
print(res)
print(cost(res[0],X,y))

#进行预测：当 hθ大于等于0.5时，预测 y=1；当 hθ小于0.5时，预测 y=0 。
def predict(theta,X):
    probability=sigmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]

theta_min=np.matrix(res[0])
predictions=predict(theta_min,X)
correct=[1 if (a==1 and b==1) or (a==0  and b==0) else 0 for (a,b) in zip(predictions,y)]
print(sum((map(int,correct))))#map内置高阶函数，sum求和1+1+1+0......
accuracy=(sum(map(int,correct))%len(correct))#len(correct)==len(X)
print('accuracy={0}%'.format(accuracy))


#寻找决策边界
coef = -(res[0]/res[0][2])
x=np.arange(130,step=0.1)
y=coef[0]+coef[1]*x
print(coef)
#绘制边界
plt.scatter(positive['e1'],positive['e2'],s = 50,c = 'b',marker='x',label='admitted')
plt.scatter(negative['e1'],negative['e2'], s=50, c='r',marker='o',label='not admitted')
plt.xlabel('e1')
plt.ylabel('e2')
plt.plot(x,y,'g-')
plt.title('Decision Boundary')
plt.show()