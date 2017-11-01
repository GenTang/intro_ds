# -*- coding: UTF-8 -*-
"""
此脚本用于展示one_way ANOVA方法
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def generateData():
    """
    """
    np.random.seed(2046)
    d1 = np.random.normal(5, 5, 10)
    d2 = np.random.normal(5, 5, 10)
    groups = ["d1"] * 10 + ["d2"] * 10
    d = {"A": np.concatenate((d1, d2)), "B": groups}
    data = pd.DataFrame(d)
    return data


def visualize(data):
    """
    绘制数据的箱型图
    """
    print(data.groupby("B").mean())
    data.boxplot("A", by="B", grid=False)
    plt.show()


def oneWayANOVA(data):
    """
    计算定量变量A与定性变量B之间的eta squared

    参数：
    ----
    data : DataFrame, 包含变量A和变量B
    """
    re = sm.OLS.from_formula("A ~ B", data=data).fit()
    aovTable = sm.stats.anova_lm(re, typ=2)
    # 打印ANOVA分析结果
    print(aovTable)
    # 计算eta sqaured
    etaSquared = aovTable["sum_sq"][0] / (aovTable["sum_sq"][0] + aovTable["sum_sq"][1])
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("Eta squared等于：%.3f" % etaSquared)


if __name__ == "__main__":
    data = generateData()
    visualize(data)
    oneWayANOVA(data)
