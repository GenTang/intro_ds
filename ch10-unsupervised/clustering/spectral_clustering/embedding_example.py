# -*- coding: UTF-8 -*-
"""
此脚本用于展示spectral embedding的效果
"""


import numpy as np
import matplotlib.pyplot as plt
from spectral_embedding_ import spectral_embedding


def generateData():
    """
    生成邻接矩阵
    """
    data = np.array([
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0]])
    return data


def visualize(data):
    """
    将模型结果可视化
    """
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, 0], data[:, 1], s=200, edgecolors="k")
    plt.show()


def run():
    """
    程序的入口
    """
    data = generateData()
    # 使用spectral embedding将数据转换为2维向量
    re = spectral_embedding(data, n_components=2, drop_first=False)
    visualize(re)


if __name__ == "__main__":
    run()
