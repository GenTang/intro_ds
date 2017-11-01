# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用statsmodels搭建线性回归模型
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import pandas as pd


def modelSummary(re):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print(re.summary())
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    # 用f test检测x对应的系数a是否显著
    print("检验假设x的系数等于0：")
    print(re.f_test("x=0"))
    # 用f test检测常量b是否显著
    print("检测假设const的系数等于0：")
    print(re.f_test("const=0"))
    # 用f test检测a=1, b=0同时成立的显著性
    print("检测假设x的系数等于1和const的系数等于0同时成立：")
    print(re.f_test(["x=1", "const=0"]))


def visualizeModel(re, data, features, labels):
    """
    模型可视化
    """
    # 计算预测结果的标准差，预测下界，预测上界
    prstd, preLow, preUp = wls_prediction_std(re, alpha=0.05)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_title(u'%s' % "线性回归统计分析示例")
    else:
        ax.set_title(u'%s' % "线性回归统计分析示例".decode("utf-8"))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # 画点图，用蓝色圆点表示原始数据
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.scatter(data[features], data[labels], color='b',
            label=u'%s: $y = x + \epsilon$' % "真实值")
    else:
        ax.scatter(data[features], data[labels], color='b',
            label=u'%s: $y = x + \epsilon$' % "真实值".decode("utf-8"))
    # 画线图，用红色虚线表示95%置信区间
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(data[features], preUp, "r--", label=u'%s' % "95%置信区间")
        ax.plot(data[features], re.predict(data[features]), color='r',
            label=u'%s: $y = %.3f * x$'\
            % ("预测值", re.params[features]))
    else:
        ax.plot(data[features], preUp, "r--", label=u'%s' % "95%置信区间".decode("utf-8"))
        ax.plot(data[features], re.predict(data[features]), color='r',
            label=u'%s: $y = %.3f * x$'\
            % ("预测值".decode("utf-8"), re.params[features]))
    ax.plot(data[features], preLow, "r--")
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    plt.show()


def trainModel(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    re = model.fit()
    return re


def linearModel(data):
    """
    线性回归统计性质分析步骤展示

    参数
    ----
    data : DataFrame，建模数据
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    # 加入常量变量
    X = sm.add_constant(data[features])
    # 构建模型
    re = trainModel(X, Y)
    # 分析模型效果
    modelSummary(re)
    # const并不显著，去掉这个常量变量
    resNew = trainModel(data[features], Y)
    # 输出新模型的分析结果
    print(resNew.summary())
    # 将模型结果可视化
    visualizeModel(resNew, data, features, labels)


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    homePath = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        dataPath = "%s\\data\\simple_example.csv" % homePath
    else:
        dataPath = "%s/data/simple_example.csv" % homePath
    data = readData(dataPath)
    linearModel(data)
