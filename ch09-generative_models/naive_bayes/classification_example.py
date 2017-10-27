# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用random forest embedding和伯努利模型解决分类问题
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline


def generateData(n):
    """
    随机生成训练数据
    """
    X, Y = make_moons(n_samples=n, noise=0.05, random_state=2046)
    data = np.concatenate((Y.reshape(-1, 1), X), axis=1)
    data = pd.DataFrame(data, columns=["y", "x1", "x2"])
    return data


def trainModel(data):
    """
    使用random forest embedding+伯努利模型对数据建模
    """
    pipe = Pipeline([("embedding", RandomTreesEmbedding(random_state=1024)),
        ("model", BernoulliNB())])
    pipe.fit(data[["x1", "x2"]], data["y"])
    return pipe


def visualize(data, pipe):
    """
    将数据和模型结果可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里画一幅图
    ax = fig.add_subplot(1, 1, 1)
    label1 = data[data["y"]>0]
    ax.scatter(label1[["x1"]], label1[["x2"]], marker="o")
    label0 = data[data["y"]==0]
    ax.scatter(label0[["x1"]], label0[["x2"]], marker="^", color="k")
    # 将模型的预测结果可视化
    x1 = np.linspace(min(data["x1"]) - 0.2, max(data["x1"]) + 0.2, 100)
    x2 = np.linspace(min(data["x2"]) - 0.2, max(data["x2"]) + 0.2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    prob = pipe.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:, 0]
    prob = prob.reshape(X1.shape)
    ax.contourf(X1, X2, prob, levels=[0.5, 1], colors=["gray"], alpha=0.4)
    plt.show()


if __name__ == "__main__":
    data = generateData(200)
    pipe = trainModel(data)
    visualize(data, pipe)
