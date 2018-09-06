# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何Spark MLlib中LinearRegressionWithSGD的缺陷
"""


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as sklearnLR

from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD


def read_data(path):
    """
    利用pandas读取数据
    """
    data = pd.read_csv(path, header=None)
    return np.c_[data.values[:, 0], data.values[:, 3]]


def generate_data(n):
    """
    生成训练模型的数据
    """
    np.random.seed(4060)
    x = np.linspace(-7, 7, n)
    error = np.random.randn(n)
    y = 1 * x + 2 + error
    return np.c_[y, x]


def start_spark():
    """
    创建SparkSession，这是Spark程序的入口
    """
    spark = SparkSession.builder.appName("sparkml_vs_sklearn").getOrCreate()
    return spark


def trans_2_RDD(data, sc):
    """
    将Python里的数据转换为RDD
    """
    data = sc.parallelize(data)
    data = data.map(lambda line: LabeledPoint(line[0], line[1:]))
    return data


def train_model(data, rdd):
    """
    分别使用scikit-learn和Spark MLlib训练模型
    """
    sklearn_model = sklearnLR()
    sklearn_model.fit(data[:, 1:], data[:, 0])
    mllib_model = LinearRegressionWithSGD.train(rdd, intercept=True)
    return sklearn_model, mllib_model


def run(spark, data_path):
    """
    程序的入口
    """
    data = read_data(data_path)
    rdd = trans_2_RDD(data, spark._sc)
    sklearn_model, mllib_model = train_model(data, rdd)
    # 创建一个图形框
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    _visualize(sklearn_model, mllib_model, data, ax)
    data = generate_data(200)
    rdd = trans_2_RDD(data, spark._sc)
    sklearn_model, mllib_model = train_model(data, rdd)
    ax = fig.add_subplot(1, 2, 2)
    _visualize(sklearn_model, mllib_model, data, ax)
    plt.show()


def _visualize(sklearn_model, mllib_model, data, ax):
    """
    将模型结果可视化
    """
    ax.set_ylim([data[:, 0].min()-1, data[:, 0].max()+1])
    ax.scatter(data[:, 1], data[:, 0], alpha=0.5)
    x = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
    ax.plot(x, sklearn_model.predict(x.reshape(-1, 1)), "k", linewidth=2, label="scikit-learn")
    ax.plot(x, [mllib_model.predict(i) for i in x.reshape(-1, 1)],
            "r-.", linewidth=2, label="Spark MLlib")
    legend = plt.legend(shadow=True)


if __name__ == "__main__":
    spark = start_spark()
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data\\reg_data.csv" % home_path
    else:
        data_path = "%s/data/reg_data.csv" % home_path
    run(spark, data_path)
