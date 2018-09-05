# -*- coding: UTF-8 -*-
"""
此脚本用于随机生成线性回归模型的训练数据
"""


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_data():
    """
    随机生成数据
    """
    np.random.seed(4889)
    # Python2和Python3的range并不兼容，所以使用list(range(10, 29))
    x = np.array([10] + list(range(10, 29)))
    error = np.round(np.random.randn(20), 2)
    y = x + error
    return pd.DataFrame({"x": x, "y": y})


def visualize_data(data):
    """
    数据可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 创建一个图形框，在里面只画一幅图
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_title("%s" % "线性回归示例")
    else:
        ax.set_title("%s" % "线性回归示例".decode("utf-8"))
    ax.set_xlabel("$x$")
    ax.set_xticks(range(10, 31, 5))
    ax.set_ylabel("$y$")
    ax.set_yticks(range(10, 31, 5))
    # 画点图，点的颜色为蓝色
    ax.scatter(data.x, data.y, color="b", label="$y = x + \epsilon$")
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor("#6F93AE")
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效。
    plt.show()


if __name__ == "__main__":
    data = generate_data()
    home_path = os.path.dirname(os.path.abspath(__file__))
    # 存储数据，Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data.to_csv("%s\\data\\simple_example.csv" % home_path, index=False)
    else:
        data.to_csv("%s/data/simple_example.csv" % home_path, index=False)
    visualize_data(data)
