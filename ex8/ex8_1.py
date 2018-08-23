import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

data=loadmat('data/ex8data1.mat')
X=data['X'] #shape (307, 2)

def plot1():
    fig,ax=plt.subplots()
    ax.scatter(X[:,0],X[:,1])
    plt.show()

def estimate_gaussian(X):
    mu=X.mean(axis=0)
    sigma=X.var(axis=0)
    return mu,sigma

mu,sigma=estimate_gaussian(X)
Xval=data['Xval']#(307, 2)
yval=data['yval']#(307, 1)

from scipy import stats
dist=stats.norm(mu[0],sigma[0])
p=np.zeros((X.shape[0],X.shape[1]))
#数组传递给概率密度函数，并获得数据集中每个点的正态分布概率
p[:,0]=stats.norm(mu[0],sigma[0]).pdf(X[:,0])#pdf: scipy.stats._distn_infrastructure.rv_frozen
p[:,1]=stats.norm(mu[1],sigma[1]).pdf(X[:,1])
#验证集：使用相同的模型参数
pval=np.zeros((Xval.shape[0],Xval.shape[1]))
pval[:,0]=stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])
pval[:,1]=stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])

def select_threshold(pval,yval):
    best_epsilon=0 #阈值
    best_f1=0
    f1=0
    step=(pval.max()-pval.min())/1000
    for epsilon in np.arange(pval.min(),pval.max(),step):
        preds=pval<epsilon
        tp=np.sum(np.logical_and(preds==1,yval==1)).astype(float)
        fp=np.sum(np.logical_and(preds==1,yval==0)).astype(float)
        fn=np.sum(np.logical_and(preds==0,yval==1)).astype(float)
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        f1=(2*precision*recall)/(precision+recall)
        if f1>best_f1:
            best_f1=f1
            best_epsilon=epsilon
    return best_epsilon,best_f1

epsilon,f1=select_threshold(pval,yval)
#第0列还是第1列有一个的p<epsilon就是异常
outliers=np.where(p<epsilon)#第二个array的元素代表是第0列还是第1列

def plot2(outliers):
    fig,ax=plt.subplots()
    ax.scatter(X[:,0],X[:,1])
    ax.scatter(X[outliers[0],0],X[outliers[0],1],s=50,color='r',marker='x')
    plt.show()

plot2(outliers)


