# -*- coding: UTF-8 -*-
"""
此脚本用于展示随机变量引起的模型幻觉
"""


import numpy as np
import matplotlib.pyplot as plt


def generateData(seed, num):
    x = 0
    np.random.seed(seed)
    data = []
    for i in range(num):
        x += np.random.normal()
        data.append(x)
    return data


def visualizeData(series1, series2):
    """
    根据给定的fpr和tpr，绘制ROC曲线
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 在Matplotlib中显示负号
    plt.rcParams['axes.unicode_minus']=False
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 在图形框里只画两幅图
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(series1)
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(series2)
    plt.show()


if __name__ == "__main__":
    series1 = generateData(4096, 200)
    series2 = generateData(2046, 200)
    visualizeData(series1, series2)
