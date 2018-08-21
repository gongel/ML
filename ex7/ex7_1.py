import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

def find_closest_centroids(X,centroids):
    m=X.shape[0]
    k=centroids.shape[0]
    idx=np.zeros(m)
    for i in range(m):
        min_dist=1000000
        for j in range(k):
            dist=np.sum((X[i,:]-centroids[j,:])**2)
            if dist<min_dist:
                min_dist=dist
                idx[i]=j
    return idx

data=loadmat('data/ex7data2.mat')
X=data['X'] #2个特征
initial_centroids=np.array([[3,3],[6,2],[8,5]])#初始化聚类中心点
# idx=find_closest_centroids(X,initial_centroids)
# print(idx[0:3])#[ 0.  2.  1.],前3个实例的聚类中心分别为initial_centroids的第0，2,1个实例（每个实例有两个特征）
df=pd.DataFrame(data.get('X'),columns=['X1','X2'])
# print(df.head())
# sb.set(context='notebook',style='ticks')
# sb.lmplot('X1','X2',data=df,fit_reg=False)
# plt.show()

def compute_centroids(X,idx,k):
    m,n=X.shape
    centroids=np.zeros((k,n))
    for i in range(k):
        indices=np.where(idx==i)#返回idx里面元素等于i的下标组成的array
        # print(indices[0])#indices的type为tuple
        centroids[i,:]=np.sum(X[indices[0],:],axis=0)/len(indices[0])#求属于同一个中心的所有数据的平均值（属性X1和属性X2）
    return centroids
# print(compute_centroids(X,idx,3))

def run_kmeans(X,initial_centroids,max_iters):
    m,n=X.shape
    k=initial_centroids.shape[0]
    idx=np.zeros(m)
    centroids=initial_centroids
    for _ in range(max_iters):
        idx=find_closest_centroids(X,centroids)
        centroids=compute_centroids(X,idx,k)
    return idx,centroids

def plot_cluster(idx):
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]
    fig, ax = plt.subplots()
    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='cluster1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='cluster2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='cluster3')
    ax.legend()  # 图示
    plt.show()

def init_centroids(X,k):
    m,n=X.shape
    centroids=np.zeros((k,n))
    idx=np.random.randint(0,m,k)#k为个数
    for i in range(k):
        centroids[i,:]=X[idx[i],:]
    return centroids
# print(init_centroids(X,3))

from IPython.display import Image
Image(filename='data/bird_small.png')
image_data=loadmat('data/bird_small.mat')
A=image_data['A']
# print(A.shape)#(128, 128, 3)
A =A / 255 # normalize value ranges
X=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
# print(X.shape)#(16384, 3)

initial_centroids=init_centroids(X,16)
idx,centroids=run_kmeans(X,initial_centroids,10)
idx=find_closest_centroids(X,centroids)#get the closest centroids at last time
X_recovered=centroids[idx.astype(int),:]# map each pixel to the centroid value
X_recovered=np.reshape(X_recovered,(A.shape[0], A.shape[1], A.shape[2]))#128*128*3
plt.imshow(X_recovered)
plt.show()
'''
您可以看到我们对图像进行了压缩，但图像的主要特征仍然存在
'''
