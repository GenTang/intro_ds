# -*- coding: UTF-8 -*-
"""
此脚本用于展示多重共线性对模型的影响
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats.stats as scss


def generate_data(n):
    """
    生成模型数据，其中x1，x2为不相关的变量；x1，x3强相关

    参数：
    ----
    n : int，数据重复次数
    """
    data = []
    np.random.seed(2046)
    for i in range(0, 3):
        for j in range(0, 3):
            data.append({"x1": i, "x2": j})
    data = pd.DataFrame(data * n)
    # 生成强相关自变量
    data["x3"] = data["x1"] + np.random.random(9 * n)
    # 生成被预测值
    error = 0.1 * np.random.random(9 * n)
    data["y"] = 0.7 * data["x1"] - 1.1 * data["x2"] + 0.3 * data["x3"] + error
    return data


def visualize(data):
    """
    将数据可视化
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    axes = scatter_matrix(data, alpha=1, diagonal='kde',
                          range_padding=0.9, figsize=(8, 8))
    corr = data.corr(method="pearson").as_matrix()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            axes[i, j].annotate("%s: %.3f" % ("相关系数", corr[i, j]),
                                (0.5, 0.9), xycoords='axes fraction', ha='center', va='center')
        else:
            axes[i, j].annotate("%s: %.3f" % ("相关系数".decode("utf-8"), corr[i, j]),
                                (0.5, 0.9), xycoords='axes fraction', ha='center', va='center')
    plt.show()


def train_model(X, Y):
    """
    训练线性回归模型
    """
    model = sm.OLS(Y, X)
    re = model.fit()
    return re


def uncorrelated_variable(data):
    """
    用不相关的x1，x2搭建回归模型
    """
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("x1和x2的相关系数为：%s" % scss.pearsonr(data["x1"], data["x2"])[0])
    Y = data["y"]
    X = sm.add_constant(data["x1"])
    re = train_model(X, Y)
    print(re.summary())
    X1 = sm.add_constant(data["x2"])
    re1 = train_model(X1, Y)
    print(re1.summary())
    X2 = sm.add_constant(data[["x1", "x2"]])
    re2 = train_model(X2, Y)
    print(re2.summary())


def correlated_variable(data):
    """
    用强相关的x1，x3搭建模型
    """
    print("x1和x3的相关系数为：%s" % scss.pearsonr(data["x1"], data["x3"])[0])
    Y = data["y"]
    X = sm.add_constant(data["x1"])
    re = train_model(X, Y)
    print(re.summary())
    X1 = sm.add_constant(data["x3"])
    re1 = train_model(X1, Y)
    print(re1.summary())
    X2 = sm.add_constant(data[["x1", "x3"]])
    re2 = train_model(X2, Y)
    print(re2.summary())
    # 检测多重共线性
    print("检测假设x1和x3同时不显著：")
    print(re2.f_test(["x1=0", "x3=0"]))
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
    vif["features"] = X2.columns
    print(vif)


def increase_data_set():
    """
    通过增加数据量来解决共线性问题
    """
    index = []
    x1_std = []
    x1 = []
    vif = []
    for i in range(1, 150):
        data = generate_data(i)
        Y = data["y"]
        X = data[["x1", "x3"]]
        X = sm.add_constant(X)
        re = train_model(X, Y)
        k = re.cov_params()
        index.append(i)
        x1_std.append(np.sqrt(k.loc["x1", "x1"]))
        x1.append(re.params["x1"])
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(index, x1, "r-.", label=u'%s' % "a1的估计值")
        ax.plot(index, x1_std, "b", label=u'%s' % "a1标准差估计值")
    else:
        ax.plot(index, x1, "r-.", label=u'%s' % "a1的估计值".decode("utf-8"))
        ax.plot(index, x1_std, "b", label=u'%s' % "a1标准差估计值".decode("utf-8"))
    legend = plt.legend(loc=4, shadow=True)
    plt.show()


def centering_data(data):
    """
    将变量的中心重置为0
    """
    Y = data["y"]
    X = data[["x3"]]
    X2 = X ** 2
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X, X2)
    ax.set_xlabel("$x3$")
    ax.set_ylabel("$x3^2$")
    ax1 = fig.add_subplot(1, 2, 2)
    center_x = X - X.mean()
    center_x2 = center_x ** 2
    ax1.scatter(center_x, center_x2)
    ax1.set_xlabel(r"$x3 - \overline{x3}$")
    ax1.set_ylabel(r"$(x3 - \overline{x3})^2$")
    plt.show()
    # 使用两种不同的变量组合训练模型
    X = pd.concat([X, X2], axis=1, ignore_index=True)
    X.columns = ["x3", "x3_squared"]
    X = sm.add_constant(X)
    re = train_model(X, Y)
    print(re.summary())
    X = pd.concat([center_x, center_x2], axis=1, ignore_index=True)
    X.columns = ["x3_center", "x3_center_sqaured"]
    X = sm.add_constant(X)
    re1 = train_model(X, Y)
    print(re1.summary())


if __name__ == "__main__":
    data = generate_data(2)
    visualize(data)
    uncorrelated_variable(data)
    correlated_variable(data)
    increase_data_set()
    centering_data(data)
