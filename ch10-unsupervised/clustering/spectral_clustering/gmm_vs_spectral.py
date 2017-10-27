# -*- coding: UTF-8 -*-
"""
此脚本用于展示GMM模型的缺陷以及谱聚类的结果
"""


import numpy as np
import matplotlib.pyplot as plt
from spectral import SpectralClustering
from sklearn.datasets import make_circles, make_moons
from sklearn.mixture import GaussianMixture


def generateCircles(n):
    """
    生成圆圈数据
    """
    data, _ = make_circles(n_samples=n, factor=0.5, noise=0.06)
    return data


def generateMoons(n):
    """
    生成月牙型数据
    """
    data, _ = make_moons(n_samples=n, noise=0.08)
    return data


def trainGMM(data, clusterNum):
    """
    训练混合高斯模型
    """
    model = GaussianMixture(n_components=clusterNum, covariance_type='full')
    model.fit(data)
    return model


def trainSpectralClustering(data, clusterNum):
    """
    训练谱聚类模型
    """
    model = SpectralClustering(n_clusters=clusterNum, affinity="rbf",
        gamma=100, assign_labels="kmeans")
    model.fit(data)
    return model


def visualize(ax, data, labels, centers):
    """
    将模型结果可视化
    """
    colors = ["#0C5FFA", "k"]
    ax.scatter(data[:, 0], data[:, 1], c=[colors[i] for i in labels],
        marker="o", alpha=0.8)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker="*", c=colors,
            edgecolors="white", s=700., linewidths=2)
    else:
        pass
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
    np.random.seed(19991)
    circles = generateCircles(1000)
    moons = generateMoons(500)
    # 展示GMM模型的效果
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    model = trainGMM(circles, 2)
    visualize(ax, circles, model.predict(circles), model.means_)
    ax = fig.add_subplot(1, 2, 2)
    model = trainGMM(moons, 2)
    visualize(ax, moons, model.predict(moons), model.means_)
    # 展示谱聚类的结果
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    model = trainSpectralClustering(circles, 2)
    visualize(ax, circles, model.labels_, None)
    ax = fig.add_subplot(1, 2, 2)
    model = trainSpectralClustering(moons, 2)
    visualize(ax, moons, model.labels_, None)
    plt.show()


if __name__ == "__main__":
    run()
