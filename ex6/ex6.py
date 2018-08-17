import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

raw_data1=loadmat('data/ex6data1.mat')
# print(raw_data)
data=pd.DataFrame(raw_data1['X'],columns=['X1','X2'])
data['y']=raw_data1['y']

def plot_scatter():
    positive=data[data['y'].isin([1])]
    negative=data[data['y'].isin([0])]
    plt.scatter(positive['X1'],positive['X2'],s=50,marker='x',label='Positive')
    plt.scatter(negative['X1'],negative['X2'],s=50,marker='o',label='Negative')
    plt.show()

from sklearn import svm
#线性分类
#C=1
svc=svm.LinearSVC(C=1,loss='hinge',max_iter=1000)
svc.fit(data[['X1','X2']],data['y'])
print(svc.score(data[['X1','X2']],data['y']))#0.980392156863
# data['SVM 1 Confidence']=svc.decision_function(data[['X1','X2']])
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data['X1'],data['X2'],s=50,c=data['SVM 1 Confidence'],cmap='seismic')
# ax.set_title('SVM (C=1) Decision Confidence')
# plt.show()

#C=100
svc2=svm.LinearSVC(C=100,loss='hinge',max_iter=1000)
svc2.fit(data[['X1','X2']],data['y'])
print(svc2.score(data[['X1','X2']],data['y']))#0.941176470588
# data['SVM 2 Confidence']=svc2.decision_function(data[['X1','X2']])
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(data['X1'],data['X2'],s=50,c=data['SVM 2 Confidence'],cmap='seismic')
# ax.set_title('SVM (C=100) Decision Confidence')
# plt.show()


#非线性分类
def gaussian_kernel(x1,x2,sigma):
    return np.exp(-(np.sum((x1-x2)**2)/(2*(sigma**2))))

raw_data2=loadmat('data/ex6data2.mat')
data=pd.DataFrame(raw_data2['X'],columns=['X1','X2'])
data['y']=raw_data2['y']
# plot_scatter()

svc3=svm.SVC(C=100,gamma=10,probability=True)
svc3.fit(data[['X1','X2']],data['y'])
print(svc3.score(data[['X1','X2']],data['y']))
# data['Probability']=svc3.predict_proba(data[['X1','X2']])[:,0]
# fig,ax=plt.subplots()
# ax.scatter(data['X1'],data['X2'],s=30,c=data['Probability'],cmap='Reds')
# plt.show()

raw_data3=loadmat('data/ex6data3.mat')
X=raw_data3['X']
Xval=raw_data3['Xval']
y=raw_data3['y'].ravel()
yval=raw_data3['yval'].ravel()

C_values=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]



def search_best():
    best_score = 0
    best_params = {'C': None, 'gamma': None}
    for C in  C_values:
        for gamma in gamma_values:
            svc=svm.SVC(C=C,gamma=gamma)
            svc.fit(X,y)
            score=svc.score(Xval,yval)
            if score>best_score:
                best_score=score
                best_params['C']=C
                best_params['gamma']=gamma
    return best_score,best_params

spam_train=loadmat('data/spamTrain.mat')
spam_test=loadmat('data/spamTest.mat')
X=spam_train['X']
Xtest=spam_test['Xtest']
y=spam_train['y'].ravel()
ytest=spam_test['ytest'].ravel()
# print(X.shape,y.shape,Xtest.shape,ytest.shape)#(4000, 1899) (4000,) (1000, 1899) (1000,)

svc4=svm.SVC()
svc4.fit(X,y)
print('Training accuracy = {0}%'.format(np.round(svc4.score(X,y)*100,2)))#94.4%
print('Test accuracy = {0}%'.format(np.round(svc4.score(Xtest,ytest)*100,2)))# 95.3%
'''
想要精度更高，调参（笑哭 哭笑 笑出眼泪 破涕为笑 笑死 笑尿 笑cry）
'''
