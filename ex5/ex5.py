import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data=sio.loadmat('ex5data1.mat')
    return map(np.ravel,[data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])

X,y,Xval,yval,Xtest,ytest=load_data()
df=pd.DataFrame({'water_level':X,'flow':y})
# sns.lmplot('water_level','flow',data=df,fit_reg=False,size=7)
# plt.show()
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
theta=np.ones(X.shape[1])

def cost(theta,X,y):
    '''
    :param theta:R((n+1)*1)
    :param X: R(m*(n+1))
    :param y: R(m*1)
    :return:
    '''
    m=X.shape[0]
    inner=X@theta-y
    sequare_sum=inner.T@ inner #平方和等价于向量内积
    cost=sequare_sum/2/m
    return cost

def costReg(theta,X,y,l=1):
    m=X.shape[0]
    regularized_term=np.power(theta[1:],2).sum()*l/2/m
    return cost(theta,X,y)+regularized_term

def gradient(theta,X,y):
    m=X.shape[0]
    inner=X.T @ (X@theta - y)
    return inner/m  # R((n+1)*1)

def gradientReg(theta,X,y,l=1):
    m=X.shape[0]
    regularized_term=theta.copy()#与theta保持相同的shape
    regularized_term[0]=0#theta0 不需要正则化
    regularized_term=regularized_term*l/m
    return gradient(theta,X,y)+regularized_term# R((n+1)*1)

def linear_regression(X,y,l=1):
    theta=np.ones(X.shape[1])
    # disp : bool
    # Set to True to print convergence messages.
    res=opt.minimize(fun=costReg,x0=theta,args=(X,y,l),method='TNC',jac=gradientReg,options={'disp':False})
    return res

# print(linear_regression(X,y,l=0).get('x'),linear_regression(X,y,l=0).get('success'))
final_theta=linear_regression(X,y,l=0).get('x')
b=final_theta[0]#截距
m=final_theta[1]
# plt.scatter(X[:,1],y,label='Training data')
# plt.plot(X[:,1],X[:,1]*m+b,label='Prediction')
# plt.show()

def plot_learning_curve1(X,y,Xval,yval,l=0):
    training_cost,cv_cost=[],[]
    m=X.shape[0]
    for i in range(1,m+1):
        res=linear_regression(X[:i,:],y[:i],l=0)#用训练集的子集来拟合模型
        tc=costReg(res.x,X[:i,:],y[:i],l=0)#子集代价
        cv=costReg(res.x,Xval,yval,l=0)#交叉验证代价
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(np.arange(1,m+1),training_cost,label='training cost')
    plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
    plt.legend(loc=1)
    plt.show()#图像见readme

def poly_features(x,power,as_ndarray=False):
    data={'f{}'.format(i):np.power(x,i) for i in range(1,power+1)}
    df=pd.DataFrame(data)#字典来创建DataFrame对象
    return df.as_matrix() if as_ndarray else df

def normalize_feature(df):
    #apply:DataFrame in pandas.core.frame
    return df.apply(lambda column:(column-column.mean())/column.std())#std为标准差

#创建多项式特征
def prepare_poly_data(*args,power):
    def prepare(x):
        df=poly_features(x,power=power)
        #归一化，标准化
        ndarr=normalize_feature(df).as_matrix()
        return np.insert(ndarr,0,np.ones(ndarr.shape[0]),axis=1)
    return [prepare(x) for x in  args]

X,y,Xval,yval,Xtest,ytest=load_data()
X_poly,Xval_poly,Xtest_poly=prepare_poly_data(X,Xval,Xtest,power=8)
# print(X_poly[:3,:])

#不同程度的正则化的多项式特征
def plot_learning_curve2(X,y,Xval,yval,l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1,m+1):
        res=linear_regression(X[:i,:],y[:i],l=l)
        #只是正则化训练参数theta，而计算两种cost均是非正则化的
        tc=cost(res.x,X[:i,:],y[:i])
        cv=cost(res.x,Xval,yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    plt.plot(np.arange(1,m+1),training_cost,label='training cost')
    plt.plot(np.arange(1,m+1),cv_cost,label='cv cost')
    plt.show()

# 无正则化即lambda=0的多项式特征
# plot_learning_curve2(X_poly,y,Xval_poly,yval,l=0)#见readme
#正则化lambda=1的多项式特征
# plot_learning_curve2(X_poly,y,Xval_poly,yval,l=1)#见readme
#正则化lambda=100的多项式特征
# plot_learning_curve2(X_poly,y,Xval_poly,yval,l=100)#见readme

#来找最佳lambad
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
def plot_learning_curve3(X,y,Xval,yval,l_candidate):
    training_cost, cv_cost = [], []
    for l in l_candidate:
        res=linear_regression(X,y,l)
        tc=cost(res.x,X,y)
        cv=cost(res.x,Xval,yval)
        training_cost.append(tc)
        cv_cost.append(cv)
    print(l_candidate[np.argmin(cv_cost)])#输出最小的cv对应的lambda：1
    plt.plot(l_candidate,training_cost,label='training cost')
    plt.plot(l_candidate,cv_cost,label='cv cost')
    plt.show()
# plot_learning_curve3(X_poly,y,Xval_poly,yval,l_candidate)#见readme

#用测试集数据去计算cost
for l in l_candidate:
    theta=linear_regression(X_poly,y,l).x
    print('test cost(lambda={}) = {}'.format(l,cost(theta,Xtest_poly,ytest)))
'''
test cost(lambda=0) = 10.122298845834932
test cost(lambda=0.001) = 11.038925797542761
test cost(lambda=0.003) = 11.263813399776474
test cost(lambda=0.01) = 10.881449681215951
test cost(lambda=0.03) = 10.022502633735408
test cost(lambda=0.1) = 8.632063147125516
test cost(lambda=0.3) = 7.3365130940506145
test cost(lambda=1) = 7.4662885310563025
test cost(lambda=3) = 11.643931095825252
test cost(lambda=10) = 27.715080290691617

调参后，lambda=0.3是最优选择
'''







