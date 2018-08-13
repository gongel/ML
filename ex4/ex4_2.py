#Back propagation
#先前向传播
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat("ex4data1.mat")#这里的数据和ex3data1相同
X=data['X']
y=data['y']
# print(X.shape,y.shape)#(5000, 400) (5000, 1)

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
y_onehot=encoder.fit_transform(y)
# print(y_onehot.shape)#(5000, 10) 说明数据只有10类

def sigmod(z):
    return 1 / (1+np.exp(-z))

#前向传播函数：（400+1）->（25+1）->（10），这里中间层为一层
def forward_propagate(X,theta1,theta2):
    m=X.shape[0]
    a1=np.insert(X,0,values=np.ones(m),axis=1)
    z2=a1*theta1.T
    a2=np.insert(sigmod(z2),0,np.ones(m),axis=1)
    z3=a2*theta2.T
    h=sigmod(z3)
    return a1,z2,a2,z3,h

def cost(params,input_size,hidden_size,num_labels,X,y,alpha):
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    # 将参数数组揭开为每层的参数矩阵
    theta1=np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2=np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    '''
    𝑎𝑖(𝑗) 代表第第 𝑗 层的第 𝑖 个激活单元或者用sl。
     𝜃(𝑗)代表从第 𝑗 层映射到第 𝑗+1 层时的权重矩,例如 𝜃(1)代表从第一层映射到第二层的权重矩阵。
     其尺寸为：第 𝑗+1 层的激活单元数量为行，以第 𝑗 层的激活单元数加一为列矩阵。
     '''
    a1, z2, a2, z3, h=forward_propagate(X,theta1,theta2)
    J=0
    for i in range(m):
        first_term=np.multiply(-y[i,:],np.log(h[i,:]))
        second_term=np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J+=np.sum(first_term-second_term)
    return J/m

#初始化参数
input_size=400
hidden_size=25
num_labels=10
alpha=1
#随机初始化完成网络参数大小的参数数组
params=(np.random.random(size=hidden_size*(input_size+1) + num_labels*(hidden_size+1)) - 0.5) *0.25
m=X.shape[0]
X=np.matrix(X)
y=np.matrix(y)
#将参数数组揭开为每层的参数矩阵
theta1=np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
theta2=np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
# print(theta1.shape,theta2.shape)# (25, 401) (10, 26)oo
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# print(a1.shape,z2.shape,a2.shape,z3.shape,h.shape) #(5000, 401) (5000, 25) (5000, 26) (5000, 10) (5000, 10)

#输出假设h和真实y的误差之和（总误差）
# print(cost(params,input_size,hidden_size,num_labels,X,y_onehot,alpha))

def costReg(params,input_size,hidden_size,num_labels,X,y,alpha):
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    # 将参数数组揭开为每层的参数矩阵
    theta1=np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2=np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    '''𝑎𝑖(𝑗) 代表第第 𝑗 层的第 𝑖 个激活单元或者用sl。
     𝜃(𝑗)代表从第 𝑗 层映射到第 𝑗+1 层时的权重矩,例如 𝜃(1)代表从第一层映射到第二的层权重矩阵。
     其尺寸为：第𝑗+1层的激活单元数量为行，以第 𝑗 层的激活单元数加一为列矩阵。
     '''
    a1, z2, a2, z3, h=forward_propagate(X,theta1,theta2)
    J=0
    for i in range(m):
        first_term=np.multiply(-y[i,:],np.log(h[i,:]))
        second_term=np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J+=np.sum(first_term-second_term)

    #增加正则项
    reg=float(alpha)/2/m*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
    return J/m+reg


#反向传播开始
def sigmod_gradient(z): #sigmod函数的求导
    return np.multiply(sigmod(z),(1-sigmod(z)))

def back_propagate_without_reg(params,input_size,hidden_size,num_labels,X,y,alpha):
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    # 将参数数组揭开为每层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    #初始化
    J=0
    delta1=np.zeros(theta1.shape)
    delta2=np.zeros(theta2.shape)
    #计算cost
    for i in range(m):
        first_term=np.multiply(-y[i,:],np.log(h[i,:]))
        second_term=np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J+=np.sum(first_term-second_term)
    reg = float(alpha) / 2 / m * (np.sum(np.power(theta1[:, 1 :], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    J=J/m+reg

    #反向传播开始
    for t in range(m): #每行计算一次（也就是每对数据）
        a1t=a1[t,:]#(1,401)
        z2t=z2[t,:]#(1,25)
        a2t=a2[t,:]#(1,26)
        ht=h[t,:]#(1,10)
        yt=y[t,:]#(1,10)
        d3t=ht-yt#(1,10)#
        z2t=np.insert(z2t,0,values=np.ones(1))#(1,26)#其实操作完之后就是a2t
        d2t=np.multiply((theta2.T * d3t.T).T,sigmod_gradient(z2t))#(1,26)
        delta1=delta1+(d2t[:,1:]).T*a1t
        delta2=delta2+d3t.T*a2t

    delta1=delta1/m
    delta2=delta2/m

    grad=np.concatenate((np.ravel(delta1),np.ravel(delta2)))
    return J,grad

J,grad=back_propagate_without_reg(params,input_size,hidden_size,num_labels,X,y_onehot,alpha)
# print(J,grad.shape)

def back_propagate_with_reg(params,input_size,hidden_size,num_labels,X,y,alpha):
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    # 将参数数组揭开为每层的参数矩阵
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    #初始化
    J=0
    delta1=np.zeros(theta1.shape)
    delta2=np.zeros(theta2.shape)
    #计算cost
    for i in range(m):
        first_term=np.multiply(-y[i,:],np.log(h[i,:]))
        second_term=np.multiply((1-y[i,:]),np.log(1-h[i,:]))
        J+=np.sum(first_term-second_term)
    reg = float(alpha) / 2 / m * (np.sum(np.power(theta1[:, 1 :], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    J=J/m+reg

    #反向传播开始
    for t in range(m): #每行计算一次（也就是每对数据）
        a1t=a1[t,:]#(1,401)
        z2t=z2[t,:]#(1,25)
        a2t=a2[t,:]#(1,26)
        ht=h[t,:]#(1,10)
        yt=y[t,:]#(1,10)
        d3t=ht-yt#(1,10)
        z2t=np.insert(z2t,0,values=np.ones(1))#(1,26)#其实操作完之后就是a2t
        d2t=np.multiply((theta2.T * d3t.T).T,sigmod_gradient(z2t))#(1,26)
        delta1=delta1+(d2t[:,1:]).T*a1t
        delta2=delta2+d3t.T*a2t

    delta1=delta1/m
    delta2=delta2/m

    #增加正则项：只会影响grad
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * alpha) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * alpha) / m

    grad=np.concatenate((np.ravel(delta1),np.ravel(delta2)))
    return J,grad
J_Reg,grad_Reg=back_propagate_with_reg(params,input_size,hidden_size,num_labels,X,y_onehot,alpha)
# print(J_Reg,grad_Reg.shape)

#准备训练网络
from scipy.optimize import minimize
fmin=minimize(fun=back_propagate_with_reg,x0=params,args=(input_size,hidden_size,num_labels,X,y_onehot,alpha),method='TNC',jac=True,options={'maxiter':250})
print(fmin)

#开始预测:类似于逻辑回归
X=np.matrix(X)
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
y_pre=np.argmax(h,axis=1)+1
print(y_pre)

#计算精准度
correct=[1 if a ==b else 0 for (a,b) in zip(y_pre,y)]
accuracy=sum(map(int,correct)) / float(len(correct))#计算1的个数并求百分比即精确度
print(accuracy)
