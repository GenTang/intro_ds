# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用混合高斯对数据进行聚类
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs



def generateCaseOne(n):
    """
    随机生成非均质数据
    """
    mean1 = (2, 3)
    cov1 = [[0.6, 0], [0, 0.6]]
    data1 = np.random.multivariate_normal(mean1, cov1, n)
    mean2 = (0, 0)
    cov2 = [[4, 0], [0, 0.1]]
    data2 = np.random.multivariate_normal(mean2, cov2, n)
    return np.r_[data1, data2]


def generateCaseTwo(n):
    """
    随机生成内部方差不相同的数据
    """
    centers = [[-2, 0], [0, 2], [2, 4]]
    std = [0.1, 1, 0.2]
    data, _ = make_blobs(n_samples=n, centers=centers, cluster_std=std)
    return data


def trainModel(data, clusterNum):
    """
    使用混合高斯对数据进行聚类
    """
    model = GaussianMixture(n_components=clusterNum, covariance_type="full")
    model.fit(data)
    return model


def visualize(ax, data, labels, centers):
    """
    将模型结果可视化
    """
    colors = ["#82CCFC", "k", "#0C5FFA"]
    ax.scatter(data[:, 0], data[:, 1], c=[colors[i] for i in labels],
        marker="o", alpha=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], marker="*", c=colors, edgecolors="white",
        s=700., linewidths=2)
    yLen = data[:, 1].max() - data[:, 1].min()
    xLen = data[:, 0].max() - data[:, 0].min()
    lens = max(yLen+1, xLen+1) / 2.
    ax.set_xlim(data[:, 0].mean()-lens, data[:, 0].mean()+lens)
    ax.set_ylim(data[:, 1].mean()-lens, data[:, 1].mean()+lens)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def run():
    """
    程序的入口
    """
    np.random.seed(12031)
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    data = generateCaseOne(400)
    model = trainModel(data, 2)
    visualize(ax, data, model.predict(data), model.means_)
    ax = fig.add_subplot(1, 2, 2)
    data = generateCaseTwo(1200)
    model = trainModel(data, 3)
    visualize(ax, data, model.predict(data), model.means_)
    plt.show()


if __name__ == "__main__":
    run()
