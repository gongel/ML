import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('data/ex8_movies.mat')
# Y是包含从1到5的等级的（数量的电影x数量的用户）数组.
# R是包含指示用户是否给电影评分的二进制值的“指示符”数组。
# 两者应该具有相同的维度。
Y = data['Y']  # (1682, 943)
R = data['R']  # (1682, 943)


def cost(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    # compute cost
    J = 0
    error = np.multiply((X * theta.T) - Y, R)
    square_error = np.power(error, 2)
    J = 1. / 2 * np.sum(square_error)
    return J


def evaluate():
    params_data = loadmat('data/ex8_movieParams.mat')
    X = params_data['X']  # (1682, 10)
    theta = params_data['Theta']  # (943, 10)
    users = 4
    movies = 5
    features = 3
    X_sub = X[:movies, :features]
    theta_sub = theta[:users, :features]
    Y_sub = Y[:movies, :users]
    R_sub = R[:movies, :users]

    params = np.concatenate((np.ravel(X_sub), np.ravel(theta_sub)))  # 将theata和X联系起来
    return params, X_sub, Y_sub, R_sub, features


def cost_and_gradient(params, Y, R, num_features):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    error = np.multiply((X * theta.T) - Y, R)
    sequare_error = np.power(error, 2)
    J = 1. / 2 * np.sum(sequare_error)
    # calculate the gradients
    X_grad = error * theta
    theta_grad = error.T * X

    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))
    return J, grad


def cost_and_gradientReg(params, Y, R, num_features, alpha):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    error = np.multiply((X * theta.T) - Y, R)
    sequare_error = np.power(error, 2)
    J = 1. / 2 * np.sum(sequare_error)

    # add the cost  regularization
    J = J + np.sum(np.power(theta, 2)) * alpha / 2
    J = J + np.sum(np.power(X, 2)) * alpha / 2
    # calculate the gradients with regularization
    X_grad = error * theta + alpha * X
    theta_grad = error.T * X + alpha * theta

    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))
    return J, grad

def costReg(params, Y, R, num_features,alpha):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    error = np.multiply((X * theta.T) - Y, R)
    sequare_error = np.power(error, 2)
    J = 1. / 2 * np.sum(sequare_error)
    # add the cost  regularization
    J = J + np.sum(np.power(theta, 2)) * alpha / 2
    J = J + np.sum(np.power(X, 2)) * alpha / 2
    return J
def gradientReg(params, Y, R, num_features, alpha):
    Y = np.matrix(Y)
    R = np.matrix(R)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    J = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    error = np.multiply((X * theta.T) - Y, R)
    # calculate the gradients with regularization
    X_grad = error * theta + alpha * X
    theta_grad = error.T * X + alpha * theta
    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))
    return grad

params, X_sub, Y_sub, R_sub, features = evaluate()
movie_idx = {}
f = open('data/movie_ids.txt', encoding='gbk')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]  # 去除换行'\n'
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])  # 修改下标并往字典里面添加数据


# 将自己的评级向量添加到现有数据集中以包含在模型中
def reproduce_and_prepare_data():
    ratings = np.zeros(1682)
    ratings[0] = 4
    ratings[6] = 3
    ratings[11] = 5
    ratings[53] = 4
    ratings[63] = 5
    ratings[65] = 3
    ratings[68] = 5
    ratings[97] = 2
    ratings[182] = 4
    ratings[225] = 5
    ratings[354] = 5
    R = data['R']
    Y = data['Y']
    Y = np.insert(Y, 0, ratings, axis=1)
    R = np.insert(R, 0, ratings != 0, axis=1)
    movies = Y.shape[0]  # 1682
    users = Y.shape[1]  # 944
    features = 50
    alpha = 10
    X = np.random.random(size=(movies, features))  # (1682, 10)
    theta = np.random.random(size=(users, features))  # (944, 10)
    params = np.concatenate((np.ravel(X), np.ravel(theta)))  # (26260,)
    Ymean = np.zeros((movies, 1))
    Ynorm = np.zeros((movies, users))
    # 归一化
    for i in range(movies):
        idx = np.where(R[i, :] == 1)[0]
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return params, Ynorm, R, features, alpha, movies, features, users, Ymean


params, Ynorm, R, features, alpha, movies, features, users, Ymean = reproduce_and_prepare_data()
print(Ynorm.mean())
# prediction
from scipy.optimize import minimize

# fmin = minimize(fun=costReg, x0=params, args=(Ynorm, R, features, alpha), method='TNC', jac=gradientReg)
fmin = minimize(fun=cost_and_gradientReg, x0=params, args=(Ynorm, R, features, alpha), method='CG', jac=True, options={'maxiter': 100})
print(fmin)

final_X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))  # (1682, 10)
final_theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))  # (944, 10)
predictions = final_X * final_theta.T
my_preds = predictions[:, -1] + Ymean
idx = np.argsort(my_preds, axis=0)[::-1]  # 降序
print(idx)
print('Top 10 movie predictions:')
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))
