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


def model_summary(re):
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


def visualize_model(re, data, features, labels):
    """
    模型可视化
    """
    # 计算预测结果的标准差，预测下界，预测上界
    pre_std, pre_low, pre_up = wls_prediction_std(re, alpha=0.05)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
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
        ax.plot(data[features], pre_up, "r--", label=u'%s' % "95%置信区间")
        ax.plot(data[features], re.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$' % ("预测值", re.params[features]))
    else:
        ax.plot(data[features], pre_up, "r--", label=u'%s' % "95%置信区间".decode("utf-8"))
        ax.plot(data[features], re.predict(data[features]), color='r',
                label=u'%s: $y = %.3fx$' % ("预测值".decode("utf-8"), re.params[features]))
    ax.plot(data[features], pre_low, "r--")
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    plt.show()


def train_model(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    re = model.fit()
    return re


def linear_model(data):
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
    re = train_model(X, Y)
    # 分析模型效果
    model_summary(re)
    # const并不显著，去掉这个常量变量
    res_new = train_model(data[features], Y)
    # 输出新模型的分析结果
    print(res_new.summary())
    # 将模型结果可视化
    visualize_model(res_new, data, features, labels)


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
    linear_model(data)
