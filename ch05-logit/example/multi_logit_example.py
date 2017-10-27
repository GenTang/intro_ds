# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用逻辑回归解决多元分类问题
"""

from os import path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    data.columns = ["label", "x1", "x2"]
    return data
    

def multiLogit(data):
    """
    使用逻辑回归对多元分类问题建模，并可视化结果
    """
    features = ["x1", "x2"]
    labels = "label"
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    methods = ['multinomial', 'ovr']
    # 使用两种不同的方法对数据建模
    for i in range(len(methods)):
        model = LogisticRegression(multi_class=methods[i], solver='sag',
            max_iter=1000, random_state=42)
        model.fit(data[features], data[labels])
        x1Min, x2Min = np.min(data[features]) - 0.5
        x1Max, x2Max = np.max(data[features]) + 0.5
        # 生成Cartesian积
        area = np.dstack(
            np.meshgrid(np.arange(x1Min, x1Max, 0.02), np.arange(x2Min, x2Max, 0.02))
            ).reshape(-1, 2)
        pic = model.predict(area)
        ax = fig.add_subplot(1, 2, i+1)
        colors = np.array(["black", "gray", "white"])
        ax.scatter(area[:, 0], area[:, 1], c=colors[pic], alpha=0.3, s=4, edgecolors=colors[pic])
        ax.scatter(data["x1"], data["x2"], c=colors[data[labels]], edgecolors="k")
        ax.set_xlim(x1Min, x1Max)
        ax.set_ylim(x2Min, x2Max)
    plt.show()


if __name__ == "__main__":
    homePath = path.dirname(path.abspath(__file__))
    dataPath = "%s/data/multi_logit.csv" % homePath
    data = readData(dataPath)
    multiLogit(data)
