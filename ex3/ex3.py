import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data =loadmat('./ex3data1.mat')
# print(data)
print(data['X'].shape,data['y'].shape)
print(data['y'])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costReg(theta,X,y,alpha):#正则化代价函数
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))#每个对应元素相乘
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg=alpha*(theta*theta.T-theta[0,0]*theta[0,0])/len(X)/2 #reg为正则化函项（不对theat0进行正则化）
    # reg = (alpha / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return (np.sum(first-second)/len(X)+reg).getA()[0][0]#len是维度


def gradientReg(theta,X, y,alpha):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    paramaters = theta.shape[1]
    grad = np.zeros(paramaters)

    error = sigmoid(X * theta.T) - y

    for j in range(paramaters):
        term = np.multiply(error, X[:, j])
        if (j == 0):
            grad[j] = np.sum(term) / len(X)
        else:
            grad[j] = np.sum(term) / len(X) + alpha / len(X) * theta[:, j]
    return grad
def gradient_without_loop(theta,X,y,alpha,lambd=1):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X * theta.T) - y
    grad=(alpha*X.T*error/len(X)).T+alpha*lambd/len(X)*theta
    #theta0没有被正则化
    grad[0,0]=np.sum(np.multiply(error,X[:,0]))/len(X)
    #或者grad[0,0]=np.sum(error)/len(X) 因为X[:,0]全为1
    return np.array(grad).ravel()

from scipy.optimize import minimize

def one_vs_all(X,y,num_labels,alpha):
    rows=X.shape[0]
    params=X.shape[1]
    all_theta=np.zeros((num_labels,params+1))
    X=np.insert(X,0,values=np.ones(rows),axis=1) #插入第一列全1

    for i in range(1,num_labels+1):
        theta=np.zeros(params+1)
        y_i=np.array([1 if label == i else 0 for label in  y])#将y变成两类：0或者1。其实可以用OneHotEncoder代替
        y_i=np.reshape(y_i,(rows,1))
        fmin = minimize(fun=costReg,x0=theta,args=(X,y_i,alpha),method='TNC',jac=gradient_without_loop)
        all_theta[i-1,:]=fmin.x
    return all_theta


print(np.unique(data['y'])) # Find the unique elements of an array.即标签的类别数目
#类别为1-10十个数字

all_theta=one_vs_all(data['X'],data['y'],10,1)
print(all_theta)

def predict_all(X,all_theta):
    rows=X.shape[0]
    params=X.shape[1]
    num_labels=all_theta.shape[0]
    X=np.insert(X,0,values=np.ones(rows),axis=1)
    X = np.matrix(X)
    all_theta=np.matrix(all_theta)
    h=sigmoid(X*all_theta.T)
    h_argmax=np.argmax(h,axis=1)#返回x轴上（axis=1）或y轴上（axis=0）找到的最大值元素所在下标组成的一维数组
    h_argmax+=1#因为数组无论行还是列都是以0开始的，但是我们的标签是从1开始的，所以得加1
    return h_argmax

y_pre=predict_all(data['X'],all_theta)
correct=[1 if a ==b else 0 for (a,b) in zip(y_pre,data['y'])]
accuracy=sum(map(int,correct)) / float(len(correct))#计算1的个数并求百分比即精确度
print(accuracy)

