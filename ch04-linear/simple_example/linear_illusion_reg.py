# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用惩罚项解决模型幻觉的问题
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os
import sys

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


def generate_random_var():
    """
    """
    np.random.seed(4873)
    return np.random.randint(2, size=20)


def train_model(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    res = model.fit()
    return res


def train_regulized_model(X, Y, alpha):
    """
    训练加入惩罚项的线性回归模型
    """
    model = sm.OLS(Y, X)
    res = model.fit_regularized(alpha=alpha)
    return res


def visualize_model(X, Y):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus'] = False
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.logspace(-4, -0.8, 100)
    coefs = []
    for alpha in alphas:
        res = train_regulized_model(X, Y, alpha)
        coefs.append(res.params)
    coefs = np.array(coefs)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(alphas, coefs[:, 1], "r:", label=u'%s' % "x的参数a")
        ax.plot(alphas, coefs[:, 2], "g", label=u'%s' % "z的参数b")
        ax.plot(alphas, coefs[:, 0], "b-.", label=u'%s' % "const的参数c")
    else:
        ax.plot(alphas, coefs[:, 1], "r:", label=u'%s' % "x的参数a".decode("utf-8"))
        ax.plot(alphas, coefs[:, 2], "g", label=u'%s' % "z的参数b".decode("utf-8"))
        ax.plot(alphas, coefs[:, 0], "b-.", label=u'%s' % "const的参数c".decode("utf-8"))
    legend = plt.legend(loc=4, shadow=True)
    legend.get_frame().set_facecolor("#6F93AE")
    ax.set_yticks(np.arange(-1, 1.3, 0.3))
    ax.set_xscale("log")
    ax.set_xlabel("$alpha$")
    plt.show()


def add_reg(data):
    """
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    _X = data[features]
    # 加入新的随机变量，次变量的系数应为0
    _X["z"] = generate_random_var()
    # 加入常量变量
    X = sm.add_constant(_X)
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("加入惩罚项（权重为0.1）的估计结果：\n%s"
          % train_regulized_model(X, Y, 0.1).params)
    # 可视化惩罚项效果
    visualize_model(X, Y)


def read_data(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data\\simple_example.csv" % home_path
    else:
        data_path = "%s/data/simple_example.csv" % home_path
    data = read_data(data_path)
    add_reg(data)
