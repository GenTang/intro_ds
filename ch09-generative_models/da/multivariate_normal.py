# -*- coding: UTF-8 -*-
"""
此脚本用于展示在不同参数下，正态分布的分布形状
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal


def generateData(n, mean, cov):
    """
    随机生成正态分布数据
    """
    np.random.seed(2033)
    data = np.random.multivariate_normal(mean, cov, size=n)
    return data


def drawData(ax, mu, cov):
    """
    将正太分布的数据可视化
    """
    data = generateData(150, mu, cov)
    ax.scatter(data[:, 0], data[:, 1])
    x = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(x, x)
    Z = bivariate_normal(X, Y, cov[0, 0], cov[1, 1], mu[0], mu[1], cov[0, 1])
    ax.contour(X, Y, Z)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def visualize():
    """
    生成并可视化不同的正太分布
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(10, 10), dpi=80)
    # 在图形框里画四幅图
    ax = fig.add_subplot(2, 2, 1)
    cov = np.array([[1., 0.], [0., 1.]])
    mu = np.array([0., 0.])
    drawData(ax, mu, cov)
    ax = fig.add_subplot(2, 2, 2)
    cov = np.array([[4., 0.], [0., 4.]])
    mu = np.array([0., 0.])
    drawData(ax, mu, cov)
    ax = fig.add_subplot(2, 2, 3)
    cov = np.array([[4., 3.], [3., 4.]])
    mu = np.array([0., 0.])
    drawData(ax, mu, cov)
    ax = fig.add_subplot(2, 2, 4)
    cov = np.array([[4., -3.], [-3., 4.]])
    mu = np.array([0., 0.])
    drawData(ax, mu, cov)
    plt.show()


if __name__ == "__main__":
    visualize()
