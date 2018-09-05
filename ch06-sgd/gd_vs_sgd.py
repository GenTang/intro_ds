# -*- coding: UTF-8 -*-
"""
此脚本用于比较两种实现方法
"""


import sys

import matplotlib.pyplot as plt
import timeit
from utils import create_summary_writer, generate_linear_data, create_linear_model
from gradient_descent import gradient_descent
from stochastic_gradient_descent import stochastic_gradient_descent


def visualize(data):
    """
    绘制两种算法的运行时间比较图
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    dataSize = [i[0] for i in data]
    gdTime = [i[1] for i in data]
    sgdTime = [i[2] for i in data]
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_xlabel("数据量")
        ax.set_ylabel("算法运行时间")
        ax.plot(dataSize, gdTime, "k", label="%s" % "梯度下降法")
        ax.plot(dataSize, sgdTime, "r-.", label="%s" % "随机梯度下降法")
    else:
        ax.set_xlabel("数据量".decode("utf-8"))
        ax.set_ylabel("算法运行时间".decode("utf-8"))
        ax.plot(dataSize, gdTime, "k", label="%s" % "梯度下降法".decode("utf-8"))
        ax.plot(dataSize, sgdTime, "r-.", label="%s" % "随机梯度下降法".decode("utf-8"))
    legend = plt.legend(shadow=True)
    plt.show()


def compare_with_diff_size():
    """
    在不同数据量下，使用两种算法对同一模型做估计
    """
    re = []
    dimension = 20
    model = create_linear_model(dimension)
    for i in range(1, 11):
        num = 10000 * i
        X, Y = generate_linear_data(dimension, num)
        # 使用梯度下降法估计模型
        start_time = timeit.default_timer()
        gradient_descent(X, Y, model)
        end_time = timeit.default_timer()
        gd_time = end_time - start_time
        # 使用随机梯度下降法估计模型
        start_time = timeit.default_timer()
        stochastic_gradient_descent(X, Y, model)
        end_time = timeit.default_timer()
        sgd_time = end_time - start_time
        re.append((num, gd_time, sgd_time))
    return re


if __name__ == "__main__":
    visualize(compare_with_diff_size())
