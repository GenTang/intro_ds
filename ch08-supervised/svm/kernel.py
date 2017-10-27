# -*- coding: UTF-8 -*-
"""
此脚本用于展示kernel SVM
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel


def generateData(n):
    """
    生成非线性分类数据
    """
    np.random.seed(2044)
    X = np.c_[np.random.uniform(-1.5, 1.5, size=n).reshape(-1, 1),
        np.random.uniform(-1.5, 1.5, size=n).reshape(-1, 1)]
    Y = ((X ** 2).sum(axis=1, keepdims=True) <= 1) + 0
    data = np.concatenate((Y, X), axis=1)
    data = pd.DataFrame(data, columns=["y", "x1", "x2"])
    return data


def visualizeData(data):
    """
    展示通过空间变换，将非线性问题转换为线性问题
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 在图形框里画两幅图
    ax = fig.add_subplot(1, 2, 1)
    drawData(ax, data)
    x = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x, x)
    r = X1 ** 2 + X2 ** 2
    ax.contour(X1, X2, r, levels=[1])
    ax.contourf(X1, X2, r, levels=[1, 10], colors=["gray"], alpha=0.4)
    ax1 = fig.add_subplot(1, 2, 2)
    # 对数据做空间变换
    df = data.copy()
    df["x1"] = data["x1"] ** 2
    df["x2"] = data["x2"] ** 2
    drawData(ax1, df)
    xx = np.linspace(0, 2.5, 100)
    XX1, XX2 = np.meshgrid(xx, xx)
    l = XX1 + XX2
    ax1.contour(XX1, XX2, l, levels=[1])
    ax1.contourf(XX1, XX2, l, levels=[1, 10], colors=["gray"], alpha=0.4)
    ax1.set_xlim([0, 2.5])
    ax1.set_ylim([0, 2.5])
    plt.show()


def drawData(ax, data):
    """
    将数据可视化
    """
    label1 = data[data["y"]>0]
    ax.scatter(label1[["x1"]], label1[["x2"]], marker="o")
    label0 = data[data["y"]==0]
    ax.scatter(label0[["x1"]], label0[["x2"]], marker="^", color="k")
    return ax


def drawModel(ax, model):
    """
    将模型的分离超平面可视化
    """
    x = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x, x)
    Y = model.decision_function(np.c_[X1.ravel(), X2.ravel()])
    Y = Y.reshape(X1.shape)
    ax.contour(X1, X2, Y, levels=[-1, 0, 1], colors=["r", "r", "r"],
        linestyles=["--", "-", "--"])
    ax.contourf(X1, X2, Y, levels=[-100, 0], colors=["gray"], alpha=0.4)
    return ax


def trainLinearSVM(data):
    """
    训练线性的SVM模型
    """
    model = SVC(kernel="linear")
    model.fit(data[["x1", "x2"]], data["y"])
    return model


def trainModel(data):
    """
    在模型里使用不同的核函数
    """
    kernel = [linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel]
    res = []
    for i in kernel:
        model = SVC(kernel=i, coef0=1)
        model.fit(data[["x1", "x2"]], data["y"])
        res.append({"name": i.__name__, "result": model})
    return res


def visualizeModel(data, res):
    """
    将模型结果可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(10, 10), dpi=80)
    # 在图形框里画四幅图
    for i in range(len(res)):
        ax = fig.add_subplot(2, 2, i+1)
        drawData(ax, data)
        drawModel(ax, res[i]["result"])
        ax.set_title(res[i]["name"])
    plt.show()


if __name__ == "__main__":
    data = generateData(30)
    re = trainLinearSVM(data)
    print "模型的预测结果为：%s" % re.predict(data[["x1", "x2"]])
    visualizeData(data)
    res = trainModel(data)
    visualizeModel(data, res)
