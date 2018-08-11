import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path='./ex2data2.txt'
data=pd.read_csv(path,header=None,names=['t1','t2','ac'])
# print(data.head())

positive=data[data['ac'].isin([1])]
negative=data[data['ac'].isin([0])]


plt.scatter(positive['t1'],positive['t2'],s=50,c='b',marker='o')
plt.scatter(negative['t1'],negative['t2'],s=50,c='r',marker='x')
plt.legend(['positive','negative'])#图例：positive，negative
plt.xlabel('t1')
plt.ylabel('t2')
# plt.show()

degree=5
x1=data['t1']
x2=data['t2']
data.insert(3,'ones',1)
for i in range(1,degree):
    for j in range(0,i):
        data['F'+str(i)+str(j)]=np.power(x1,i-j)*np.power(x2,j)#创建一组多项式特征
data.drop('t1',axis=1,inplace=True)#就地删除两列t1和t2
data.drop('t2',axis=1,inplace=True)
# print(data.head())

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costReg(theta,X,y,alpha):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg=alpha*(theta*theta.T-theta[0,0]*theta[0,0])/len(X)/2 #reg为正则化函项（不对theat0进行正则化）
    # reg = (alpha / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return (np.sum(first-second)/len(X)+reg).getA()[0][0]#len是维度


def gradientReg(theta,X,y,alpha):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    paramaters=theta.shape[1]#获取theta的列数
    grad=np.zeros(paramaters) #y一维数组

    error=sigmoid(X*theta.T)-y

    for j in range(paramaters):
        term=np.multiply(error,X[:,j])
        if(j==0):
            grad[j]=np.sum(term)/len(X)
        else:
            grad[j]=np.sum(term)/len(X)+alpha/len(X)*theta[:,j]
    return grad

def gradientReg_without_loop(theta,X,y,alpha,lambd=1):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X * theta.T) - y
    grad=(alpha*X.T*error/len(X)).T+alpha*lambd/len(X)*theta
    #theta0没有被正则化
    grad[0,0]=np.sum(np.multiply(error,X[:,0]))/len(X)
    #或者grad[0,0]=np.sum(error)/len(X) 因为X[:,0]全为1
    return np.array(grad).ravel()

cols=data.shape[1]
x=data.iloc[:,1:cols]
y=data.iloc[:,0:1]
X=np.matrix(x.values)
y=np.matrix(y.values)
theta=np.zeros(11)


alpha=1
print(costReg(theta,X,y,alpha))
print(gradientReg(theta,X,y,alpha))

import scipy.optimize as opt
result=opt.fmin_tnc(func=costReg,x0=theta,fprime=gradientReg_without_loop,args=(X,y,alpha))
# result=opt.fmin_tnc(func=costReg,x0=theta,fprime=gradientReg,args=(X,y,alpha))
print(result)