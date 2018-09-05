# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用LDA做数据降维
"""


import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_data():
    """
    读取scikit-learn自带数据：手写数字
    """
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    return X, y


def dimension_reduction(X, y):
    """
    使用LDA模型将数据降到3维
    """
    model = LinearDiscriminantAnalysis(n_components=3)
    model.fit(X, y)
    new_X = model.transform(X)
    return new_X


def visualize(new_X, y):
    """
    将结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里画一幅图
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = ["r", "b", "k", "g"]
    markers = ["^", "x", "o", "*"]
    for color, marker, i in zip(colors, markers, [0, 1, 2, 3]):
        ax.scatter(new_X[y == i, 0], new_X[y == i, 1], new_X[y == i, 2],
                   color=color, alpha=.8, lw=1, marker=marker, label=i)
    plt.legend(loc='best', shadow=True)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        plt.title("利用LDA进行数据降维")
    else:
        plt.title("利用LDA进行数据降维".decode("utf-8"))
    plt.show()


if __name__ == "__main__":
    X, y = load_data()
    new_X = dimension_reduction(X, y)
    visualize(new_X, y)
