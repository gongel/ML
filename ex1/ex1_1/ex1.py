import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#DataFrame
path='./ex1data1.txt'
data = pd.read_csv(path,header=None,names=['population','profit'])
print(data.head())#显示数据前五行
print(data.describe())#描述数据的数量，平均值，中位数等等


def computerCost(X,y,theta):# J(θ0,θ1)
    #     """
    #     X: R(m*n), m 样本数, n 特征数
    #     y: R(m)
    #     theta : R(n), 线性回归的参数
    #     """
    inner = np.power((X * theta.T-y),2) #矩阵中每个元素平方
    return np.sum(inner)/(2 * len(X))

data.insert(0,'ones',1)
print(data.head())
'''
arameters:	
loc : (int) Insertion index. Must verify 0 <= loc <= len(columns)  要插入的那一列
column : (string, number, or hashable object) label of the inserted column    要插入那列的标签
value : (int, Series, or array-like) 要插入那列的值
allow_duplicates : (bool) optional    布尔类型，可选择

'''
cols = data.shape[1] #所有列数 shape[0]是行数
X = data.iloc[:,0:cols-1] #X是所有行，然后去掉最后一列
y = data.iloc[:,cols-1:cols]#y是所有行，最后一列
#iloc(行，列)

# print(X.head())
# print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))#theta初始化为1x2的零矩阵

# print(theta)
# print(X.shape)
# print("x[:,1]:",X[:,1])
# print(theta.shape)
# print(y.shape)
# print('.............................................')
# print(y)

print(computerCost(X,y,theta))
print( np.matrix(np.zeros(theta.shape)).shape)

# batch gradient descent 批量梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters) #此时cost为一个一维数组，所以下面用cost[i]进行访问
    for i in range(iters):
        error = (X * theta.T) - y #此时erros为一个mx1的矩阵

        for j in range(parameters):
            term = np.multiply(error,X[:,j]) #矩阵对应位置元素相乘
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term)) #theta第0行第0列或者第1列；sum求矩阵中所有元素的和
        theta = temp
        cost[i] = computerCost(X,y,theta)
    return theta,cost
'''
学习率alpha，迭代次数iters
https://blog.csdn.net/qq_27008079/article/details/70527140 (1.4小节)
'''

alpha = 0.01
iters = 1000

g,cost = gradientDescent(X,y,theta,alpha,iters) #g为学习好的theta
print(g) #[[-3.24140214  1.1272942 ]]

#再来计算代价函数（误差）
print(computerCost(X,y,g)) #4.515955503078912

#最后来绘制线性模型以及数据，直观地看出它的拟合
b = g[0,0]#截距b
k = g[0,1]#斜率k
sub = plt.subplot()
sub.set_xlabel("population")
sub.set_ylabel("profit")
sub.scatter(data.population, data.profit, label="Training data")
sub.plot(data.population, data.population*k + b, label="Prediction")
sub.legend(loc=2)
plt.show()

