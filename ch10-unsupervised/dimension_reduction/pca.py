# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用PCA做数据降维
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_data(n):
    """
    随机生成训练数据
    """
    np.random.seed(1001)
    x = np.linspace(-4, 4, n)
    error = np.random.randn(n)
    y = 1 * x + error
    data = np.c_[x, y]
    return data


def visualize(data, model):
    """
    将模型结果可视化
    """
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 原始数据
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8)
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 模型结果
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8)
    m = model.mean_
    for v, l in zip(model.components_, model.explained_variance_):
        start, end = m, m + 1.5 * np.sqrt(l) * v
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(facecolor="k", width=2.0))
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def train_model(data):
    """
    使用PCA对数据进行降维
    """
    model = PCA(n_components=2)
    model.fit(data)
    return model


if __name__ == "__main__":
    data = generate_data(200)
    model = train_model(data)
    visualize(data, model)
