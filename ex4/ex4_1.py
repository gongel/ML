#前馈神经网络FFNN,FP
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
# print(theta1.shape,theta2.shape)# (25, 401) (10, 26)
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# print(a1.shape,z2.shape,a2.shape,z3.shape,h.shape) #(5000, 401) (5000, 25) (5000, 26) (5000, 10) (5000, 10)

#输出假设h和真实y的误差之和（总误差）
print(cost(params,input_size,hidden_size,num_labels,X,y_onehot,alpha))

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

#输出假设h和真实y的误差之和（总误差）
print(costReg(params,input_size,hidden_size,num_labels,X,y_onehot,alpha))

