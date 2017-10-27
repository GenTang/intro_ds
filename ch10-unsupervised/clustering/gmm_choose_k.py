# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用BIC选择混合高斯的聚类个数
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs


def generateData(n):
    """
    随机生成内部方差不相同的数据
    """
    centers = [[-2, 0], [0, 2], [2, 4]]
    std = [0.1, 1, 0.2]
    data, _ = make_blobs(n_samples=n, centers=centers, cluster_std=std)
    return data


def trainModel(data, clusterNum, covType):
    """
    使用混合高斯训练模型
    """
    model = GaussianMixture(n_components=clusterNum, covariance_type=covType)
    model.fit(data)
    return model


def visualizeResult(ax, data, labels, centers):
    """
    将聚类结果可视化
    """
    colors = ["#82CCFC", "k", "#0C5FFA", "#BAE7FC", "#3CAFFA"]
    ax.scatter(data[:, 0], data[:, 1], c=[colors[i] for i in labels],
        marker="o", alpha=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], marker="*", c=colors, edgecolors="white",
        s=600., linewidths=2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def visualizeBIC(ax, re, covTypes, colors):
    """
    将聚类结果的BIC指标可视化
    """
    for i, j in enumerate(covTypes):
        re[j].plot(kind="bar", color=colors[i], ax=ax, position=i-1, width=0.20, label=j)
    legend = plt.legend(loc="best", shadow=True)
    ax.set_xlim([-1, 5])


def run():
    """
    程序的入口
    """
    np.random.seed(12031)
    data = generateData(1200)
    covTypes = ["spherical", "tied", "diag", "full"]
    colors = ["#BAE7FC", "#82CCFC", "#0C5FFA", "k"]
    re = []
    bestBIC = np.infty
    for num in range(1, 6):
        item = {"num": num}
        for cov in covTypes:
            model = trainModel(data, num, cov)
            _bic = model.bic(data)
            item[cov] = _bic
            if _bic < bestBIC:
                bestGMM = model
                bestBIC = _bic
            else:
                pass
        re.append(item)
    re = pd.DataFrame(re)
    re = re.set_index(["num"])
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    visualizeResult(ax, data, bestGMM.predict(data), bestGMM.means_)
    ax = fig.add_subplot(1, 2, 2)
    visualizeBIC(ax, re, covTypes, colors)
    plt.show()


if __name__ == "__main__":
    run()
