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


def cost(theta, X, y):
    first = np.multiply(-y,np.log(sigmoid(X * theta)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta)))
    return np.sum(first-second)/len(X)

data.insert(0,'ones',1)
cols=data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

#convert to numpy arrays and initalize the parameter array theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros((3,1))
print(X.shape)
print(y.shape)
print(theta.shape)
print(cost(theta,X,y))

def gradientDescdent(X,y,theta,alpha,iters):
    for i in range(iters):
        temp= (1/len(X))*X.T@(sigmoid(X@theta)-y)
        theta+= -alpha*temp
    return theta
print('gradientDescent',gradientDescdent(X,y,theta,alpha=0.01,iters=1000))

#进行预测：当 hθ大于等于0.5时，预测 y=1；当 hθ小于0.5时，预测 y=0 。
def predict(theta,X):
    probability=sigmoid(X*theta)
    return [1 if x>=0.5 else 0 for x in probability]

theta_min=gradientDescdent(X,y,theta,alpha=0.001,iters=1000)
predictions=predict(theta_min,X)
correct=[1 if (a==1 and b==1) or (a==0  and b==0) else 0 for (a,b) in zip(predictions,y)]
print(sum((map(int,correct))))#map内置高阶函数，sum求和1+1+1+0......
accuracy=(sum(map(int,correct))%len(correct))#len(correct)==len(X)
print('accuracy={0}%'.format(accuracy))


#寻找决策边界
x=np.arange(30,100)
y=(-theta_min[0,0]-theta_min[1,0]*x)/theta_min[2,0]
#绘制边界
plt.scatter(positive['e1'],positive['e2'],s = 50,c = 'b',marker='x',label='1')
plt.scatter(negative['e1'],negative['e2'], s = 50,c='r',marker='o',label='0')
plt.xlabel('e1')
plt.ylabel('e2')
plt.plot(x,y,'g')
plt.title('Decision Boundary')
plt.show()