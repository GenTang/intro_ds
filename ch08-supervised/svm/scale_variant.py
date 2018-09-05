# -*- coding: UTF-8 -*-
"""
此脚本用于展示SVM是scale variant
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel


def generate_data(n):
    """
    生成非线性分类数据
    """
    np.random.seed(2044)
    X = np.c_[np.random.uniform(-1.5, 1.5, size=n).reshape(-1, 1),
              np.random.uniform(-1.5, 1.5, size=n).reshape(-1, 1)]
    Y = ((X ** 2).sum(axis=1, keepdims=True) <= 1) + 0
    data = np.concatenate((Y, X), axis=1)
    data = pd.DataFrame(data, columns=["y", "x1", "x2"])
    data["x2"] = data["x2"]
    return data


def train_model(data):
    """
    在模型里使用不同的核函数
    """
    kernel = [polynomial_kernel, rbf_kernel]
    res = []
    for i in kernel:
        model = SVC(kernel=i, coef0=1)
        model.fit(data[["x1", "x2"]], data["y"])
        res.append({"name": i.__name__, "result": model})
    return res


def draw_data(ax, data):
    """
    将数据可视化
    """
    label1 = data[data["y"] > 0]
    ax.scatter(label1[["x1"]], label1[["x2"]], marker="o")
    label0 = data[data["y"] == 0]
    ax.scatter(label0[["x1"]], label0[["x2"]], marker="^", color="k")
    return ax


def draw_model(ax, model):
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


def visualize_model(data):
    """
    将模型结果可视化
    """
    res = train_model(data)
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 在图形框里画四幅图
    for i in range(len(res)):
        ax = fig.add_subplot(1, 2, i+1)
        draw_data(ax, data)
        draw_model(ax, res[i]["result"])
        ax.set_title(res[i]["name"])
    plt.show()


if __name__ == "__main__":
    data = generate_data(30)
    new_data = data.copy()
    new_data["x2"] = data["x2"] / 4
    visualize_model(data)
    visualize_model(new_data)
