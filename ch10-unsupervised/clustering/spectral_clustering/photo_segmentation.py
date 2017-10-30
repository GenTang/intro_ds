# -*- coding: UTF-8 -*-
"""
此脚本用于展示利用谱聚类进行图片分割
"""


import os

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from spectral import spectral_clustering


def readImg(path):
    """
    读取图片，并将其转换为邻接矩阵
    """
    # 对于彩色照片，只使用其中一个维度的色彩
    im = sp.misc.imread(path)[:, :, 2]
    im = im / 255.
    # 若运算速度太慢，可使用如下的语句来缩减图片的大小
    # im = sp.misc.imresize(im, 0.10) / 255.
    # 计算图片的梯度，既相邻像素点之差
    graph = image.img_to_graph(im)
    beta = 20
    # 计算邻接矩阵
    graph.data = np.exp(-beta * graph.data / graph.data.std())
    return im, graph


def visualize(im, labels):
    """
    将模型结果可是化
    """
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(im, cmap=plt.cm.gray, alpha=0.8)
    ax.contour(labels, contours=range(3), colors="k", linewidths=3,
        linestyles="dashdot")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()    


def run(path):
    """
    程序的入口
    """
    im, graph = readImg(path)
    # 使用AMG算法来加速计算，但需要先安装第三方库pyamg
    labels = spectral_clustering(graph, n_clusters=3,
        assign_labels="discretize", eigen_solver="amg")
    labels = labels.reshape(im.shape)
    visualize(im, labels)


if __name__ == "__main__":
    homePath = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        dataPath = "%s\\data\\photo.jpg" % homePath
    else:
        dataPath = "%s/data/photo.jpg" % homePath
    run(dataPath)
