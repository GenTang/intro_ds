# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用核函数对非线性数据进行降维
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA, KernelPCA


def generate_data(n):
    """
    生成线性和非线性数据
    """
    x = np.linspace(-5, 5, n)
    error = np.random.randn(n)
    y = 1 * x + error
    linear = np.c_[x, y]
    non_linear, _ = make_moons(n_samples=n, noise=0.05)
    return linear, non_linear


def train_PCA(data):
    """
    使用线性主成分分析对数据进行降维
    """
    model = PCA(n_components=2)
    model.fit(data)
    return model


def train_kernelPCA(data):
    """
    使用带有核函数的主成分分析对数据进行降维
    """
    model = KernelPCA(n_components=2, kernel="rbf", gamma=25)
    model.fit(data)
    return model


def visualize(ax, data, model):
    """
    将PCA的降维结果可视化
    """
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8)
    m = model.mean_
    v = model.components_[0]
    l = data[:, 0].max() - data[:, 0].min()
    start, end = m, m + .5 * l * v / np.linalg.norm(v)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(facecolor="k", width=2.0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def visualize_kernelPCA(ax, data, labels):
    """
    将kernel PCA的降维结果可视化
    """
    colors = ["#82CCFC", "k"]
    markers = ["^", "o"]
    for i in range(len(colors)):
        ax.scatter(data[labels == i, 0], data[labels == i, 1],
                   color=colors[i], s=50, marker=markers[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def run_PCA(linear, non_linear):
    """
    使用PCA分别对线性和非线性数据进行降维
    """
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    model = train_PCA(linear)
    visualize(ax, linear, model)
    ax = fig.add_subplot(1, 2, 2)
    model = train_PCA(non_linear)
    visualize(ax, non_linear, model)
    plt.show()


def run_kernelPCA():
    """
    使用kernel PCA对数据降维
    """
    data, labels = make_moons(n_samples=100, noise=0.05)
    fig = plt.figure(figsize=(10, 10), dpi=80)
    # 将原始数据可视化
    ax = fig.add_subplot(2, 2, 1)
    visualize_kernelPCA(ax, data, labels)
    # 使用PCA对数据降维，并将结果可视化
    ax = fig.add_subplot(2, 2, 2)
    model = train_PCA(data)
    x = model.transform(data)[:, 0]
    visualize_kernelPCA(ax, np.c_[x, [0] * len(x)], labels)
    # 使用kernel PCA对数据降维，并将结果可视化
    ax = fig.add_subplot(2, 2, 3)
    model = train_kernelPCA(data)
    x = model.transform(data)[:, 0]
    visualize_kernelPCA(ax, np.c_[x, [0] * len(x)], labels)
    # 展示数据在kernel PCA第一和第二主成分的降维结果
    ax = fig.add_subplot(2, 2, 4)
    visualize_kernelPCA(ax, model.transform(data), labels)
    plt.show()


if __name__ == "__main__":
    np.random.seed(20001)
    linear, non_linear = generate_data(200)
    run_PCA(linear, non_linear)
    run_kernelPCA()
