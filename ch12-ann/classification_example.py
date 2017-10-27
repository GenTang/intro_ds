# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用神经网络解决分类问题
"""


from mlp import ANN 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  StandardScaler, OneHotEncoder


def generateData(n):
    """
    """
    np.random.seed(12046)
    blobs = make_blobs(n_samples=n, centers = [[-2, -2], [2, 2]])
    circles = make_circles(n_samples=n, factor=.4, noise=.05)
    moons = make_moons(n_samples=n, noise=.05)
    blocks = np.random.rand(n, 2) - 0.5
    y = (blocks[:, 0] * blocks[:, 1] < 0) + 0
    blocks = (blocks, y)
    # 由于神经网络对数据的线性变换不稳定，因此将数据做归一化处理
    scaler = StandardScaler()
    blobs = (scaler.fit_transform(blobs[0]), blobs[1])
    circles = (scaler.fit_transform(circles[0]), circles[1])
    moons = (scaler.fit_transform(moons[0]), moons[1])
    blocks = (scaler.fit_transform(blocks[0]), blocks[1])
    return blobs, circles, moons, blocks


def drawData(ax, data):
    """
    将数据可视化
    """
    X, y = data
    label1 = X[y>0]
    ax.scatter(label1[:, 0], label1[:, 1], marker="o")
    label0 = X[y==0]
    ax.scatter(label0[:, 0], label0[:, 1], marker="^", color="k")
    return ax


def drawModel(ax, model):
    """
    将模型的分离超平面可视化
    """
    x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    x2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Y = model.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:, 1]
    Y = Y.reshape(X1.shape)
    ax.contourf(X1, X2, Y, levels=[0, 0.5], colors=["gray"], alpha=0.4)
    return ax


def trainLogit(data):
    """
    """
    X, y = data
    model = LogisticRegression()
    model.fit(X, y)
    return model


def trainANN(data, logPath):
    """
    """
    X, y = data
    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()
    model = ANN([4, 4, 2], logPath)
    model.fit(X, y)
    return model


def visualize(data):
    """
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(10, 10), dpi=80)
    fig1 = plt.figure(figsize=(10, 10), dpi=80)
    # 在图形框里画四幅图
    for i in range(len(data)):
        ax = fig.add_subplot(2, 2, i+1)
        ax1 = fig1.add_subplot(2, 2, i+1)
        drawData(ax, data[i])
        drawModel(ax, trainANN(data[i], "logs/data_%s" % (i+1)))
        drawData(ax1, data[i])
        drawModel(ax1, trainLogit(data[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    data = generateData(200)
    visualize(data)
