# -*- coding: UTF-8 -*-
"""
此脚本用于展示GBTs
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


def generateData(n):
    """
    生成训练数据
    """
    np.random.seed(1010)
    X = np.linspace(0, 3 * np.pi, num=n).reshape(-1, 1)
    error = np.random.normal(0, 0.1, size=n).reshape(-1, 1)
    Y = np.abs(np.sin(X)) + error
    Y = np.where(Y > 0.5, 1, 0)
    data = np.concatenate((Y, X), axis=1)
    data = pd.DataFrame(data, columns=["y", "x"])
    return data


def trainModel(data):
    """
    训练GBTs模型
    """
    model = GradientBoostingRegressor(n_estimators=3, max_depth=2, learning_rate=0.8)
    model.fit(data[["x"]], data["y"])
    return model


def visualize(data, model):
    """
    将模型结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data["x"], data["y"], label="_nolegend_")
    styles = ["b--", "r-.", "gray"]
    labels = ["深度=1".decode("utf-8"), "深度=2".decode("utf-8"), "深度=3".decode("utf-8")]
    for l, s, pred in zip(labels, styles, model.staged_predict(data[["x"]])):
        plt.plot(data[["x"]], pred, s, label=l)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=3, fancybox=True, shadow=True)
    plt.show()


if __name__ == "__main__":
    data = generateData(40)
    model = trainModel(data)
    visualize(data, model)
