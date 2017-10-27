# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用惩罚项解决模型幻觉的问题
"""


from os import path

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


def generateRandomVar():
    """
    """
    np.random.seed(4873)
    return np.random.randint(2, size=20)


def trainModel(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    res = model.fit()
    return res


def trainRegulizedModel(X, Y, alpha):
    """
    训练加入惩罚项的线性回归模型
    """
    model = sm.OLS(Y, X)
    res = model.fit_regularized(alpha=alpha)
    return res


def visualizeModel(X, Y):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus']=False
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    alphas = np.logspace(-4, -0.8, 100)
    coefs = []
    for alpha in alphas:
        res = trainRegulizedModel(X, Y, alpha)
        coefs.append(res.params)
    coefs = np.array(coefs)
    ax.plot(alphas, coefs[:, 1], "r:",
        label=u'%s' % "x的参数a".decode("utf-8"))
    ax.plot(alphas, coefs[:, 2], "g",
        label=u'%s' % "z的参数b".decode("utf-8"))
    ax.plot(alphas, coefs[:, 0], "b-.",
        label=u'%s' % "const的参数c".decode("utf-8"))
    legend = plt.legend(loc=4, shadow=True)
    legend.get_frame().set_facecolor("#6F93AE")
    ax.set_yticks(np.arange(-1, 1.3, 0.3))
    ax.set_xscale("log")
    ax.set_xlabel("$alpha$")
    plt.show()


def addReg(data):
    """
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    _X = data[features]
    # 加入新的随机变量，次变量的系数应为0
    _X["z"] = generateRandomVar()
    # 加入常量变量
    X = sm.add_constant(_X)
    print "加入惩罚项（权重为0.1）的估计结果：\n%s" % trainRegulizedModel(X, Y, 0.1).params
    # 可视化惩罚项效果
    visualizeModel(X, Y)


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    homePath = path.dirname(path.abspath(__file__))
    dataPath = "%s/data/simple_example.csv" % homePath
    data = readData(dataPath)
    addReg(data)
