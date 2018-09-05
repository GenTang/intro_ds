# -*- coding: UTF-8 -*-
"""
此脚本用于展示svm和logit regression的差异
"""


import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def generate_data(n, centers):
    """
    生成训练模型的数据
    """
    np.random.seed(2046)
    X = np.r_[np.random.randn(n, 2) + centers[0], np.random.randn(n, 2) + centers[1]]
    Y = [[0]] * n + [[1]] * n
    data = np.concatenate((Y, X), axis=1)
    data = pd.DataFrame(data, columns=["y", "x1", "x2"])
    return data


def visualize(A, B, re_A, re_B):
    """
    将模型结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    # 在图形框里画两幅图
    ax = fig.add_subplot(1, 2, 1)
    draw_data(ax, A)
    draw_hyperplane(ax, re_A[0].coef_, re_A[0].intercept_, "k")
    draw_hyperplane(ax, re_A[1].coef_, re_A[1].intercept_, "r-.")
    ax.set_xlim([-12, 12])
    ax.set_ylim([-7, 7])
    legend = plt.legend(shadow=True, loc="best")
    ax1 = fig.add_subplot(1, 2, 2)
    draw_data(ax1, B)
    draw_hyperplane(ax1, re_B[0].coef_, re_B[0].intercept_, "k")
    draw_hyperplane(ax1, re_B[1].coef_, re_B[1].intercept_, "r-.")
    ax1.set_xlim([-12, 12])
    ax1.set_ylim([-7, 7])
    legend = plt.legend(shadow=True, loc="best")
    plt.show()


def draw_data(ax, data):
    """
    将数据点描绘出来
    """
    label1 = data[data["y"] > 0]
    ax.scatter(label1[["x1"]], label1[["x2"]], marker="o")
    label0 = data[data["y"] == 0]
    ax.scatter(label0[["x1"]], label0[["x2"]], marker="^", color="k")
    return ax


def draw_hyperplane(ax, coef, intercept, style):
    """
    描绘模型的分离超平面
    """
    a = -coef[0][0] / coef[0][1]
    xx = np.linspace(-8, 12)
    yy = a * xx - (intercept) / coef[0][1]
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(xx, yy, style, label="%s: %.2f" % ("直线斜率为", a))
    else:
        ax.plot(xx, yy, style, label="%s: %.2f" % ("直线斜率为".decode("utf-8"), a))
    return ax


def svm_and_logit(data):
    """
    分别训练SVM和logit regression模型
    """
    svm_model = SVC(C=1, kernel="linear")
    svm_model.fit(data[["x1", "x2"]], data["y"])
    logit_model = LogisticRegression()
    logit_model.fit(data[["x1", "x2"]], data["y"])
    return svm_model, logit_model


def soft_and_hard_margin(data):
    """
    分别训练soft margin SVM和hard margin SVM
    """
    hard_margin = SVC(C=1, kernel="linear")
    hard_margin.fit(data[["x1", "x2"]], data["y"])
    soft_margin = SVC(C=1e-4, kernel="linear")
    soft_margin.fit(data[["x1", "x2"]], data["y"])
    return hard_margin, soft_margin


if __name__ == "__main__":
    A = generate_data(5, [[-1, -1], [3, 3]])
    B = A.append(generate_data(200, [[-7, -2], [8.5, 3]]))
    svm_A, logit_A = svm_and_logit(A)
    svm_B, logit_B = svm_and_logit(B)
    visualize(A, B, (svm_A, logit_A), (svm_B, logit_B))
    soft_A, hard_A = soft_and_hard_margin(A)
    soft_B, hard_B = soft_and_hard_margin(B)
    visualize(A, B, (soft_A, hard_A), (soft_B, hard_B))
