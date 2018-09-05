# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用HMM模型对股票数据进行分析
"""


import os

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ochl
from matplotlib.dates import date2num, YearLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM
import pandas as pd


def read_data(path):
    """
    读取数据
    """
    data = pd.read_csv(path)
    # 由于股改的原因，只使用2005-06-01之后的数据
    data = data[data["date"] > "2005-01-01"]
    return data


def trans_feature(data):
    """
    提取特征，5日收益率，20日收益率，5日成交量增长率以及20日成交量增长率
    """
    data[["amount", "close_price"]] = data[["amount", "close_price"]].apply(pd.to_numeric)
    data["a_5"] = np.log(data["amount"]).diff(-5)
    data["a_20"] = np.log(data["amount"]).diff(-20)
    data["r_5"] = np.log(data["close_price"]).diff(-5)
    data["r_20"] = np.log(data["close_price"]).diff(-20)
    data["date2num"] = data["date"].apply(lambda x: date2num(datetime.strptime(x, "%Y-%m-%d")))
    data = data[data["date"] > "2005-06-01"]
    return data


def get_hidden_status(data):
    """
    使用Gaussian HMM对数据进行建模，并得到预测值
    """
    cols = ["r_5", "r_20", "a_5", "a_20"]
    model = GaussianHMM(n_components=3, covariance_type="full",
                        n_iter=1000, random_state=2010)
    model.fit(data[cols])
    hidden_status = model.predict(data[cols])
    return hidden_status


def visualize(data, hidden_status):
    """
    将模型结果可视化
    """
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax0 = fig.add_subplot(4, 1, 1)
    draw_data(ax0, data)
    for i in range(max(hidden_status)+1):
        _data = data[hidden_status == i]
        ax = fig.add_subplot(4, 1, i + 2, sharex=ax0, sharey=ax0)
        draw_data(ax, _data)
    plt.show()


def draw_data(ax, _data):
    """
    使用柱状图表示股市数据
    """
    candlestick_ochl(ax, _data[["date2num", "open_price",
                                "close_price", "high_price", "low_price"]].values,
                     colorup="r", colordown="g", width=0.5)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    return ax


def stock_HMM(data_path):
    """
    程序入口
    """
    data = read_data(data_path)
    data = trans_feature(data)
    hidden_status = get_hidden_status(data)
    visualize(data, hidden_status)


if __name__ == "__main__":
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data\\stock_sh.txt" % os.path.dirname(os.path.abspath(__file__))
    else:
        data_path = "%s/data/stock_sh.txt" % os.path.dirname(os.path.abspath(__file__))
    stock_HMM(data_path)
