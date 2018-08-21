# PCA
# one target
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('data/ex7data1.mat')
X = data['X']


def plot1():
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()


# plot1()
def PCA(X):
    X = (X - X.mean()) / X.std()
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)  # 奇异阵分解
    return U, S, V  # U即为主成分


def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


# 反向转换来回复原始数据
def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)


def plot2():
    fig, ax = plt.subplots()
    U, S, V = PCA(X)
    Z = project_data(X, U, 1)
    X_recovered = recover_data(Z, U, 1)
    ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
    plt.show()


# another target:on faces
faces = loadmat('data/ex7faces.mat')
X = faces['X']  # (5000,1024)


def plot_n_image(X, n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))
    first_n_images = X[:n, :]
    fi, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True)
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


face = np.reshape(X[3, :], (32, 32))
plt.imshow(face)
plt.show()
U, S, V = PCA(X)
Z = project_data(X, U, 100)
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3, :], (32, 32))
plt.imshow(face)
plt.show()
