# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用Apache Spark以及梯度下降法训练线性回归模型
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession


def generate_data(n):
    """
    随机生成数据
    """
    x = np.random.randn(n)
    error = np.round(np.random.randn(n), 2)
    const = np.ones(n)
    y = 2 * x + 3 * const + error
    return pd.DataFrame({"x": x, "const": const, "y": y})


def start_spark():
    """
    创建SparkSession，这是Spark程序的入口
    """
    spark = SparkSession.builder.appName("gd_example").getOrCreate()
    return spark


def gradient_descent(data, label_col, features_col, step_size=1, max_iter=100):
    """
    利用梯度下降法训练线性回归模型

    参数
    ----
    data : Spark.DataFrame，训练模型的数据

    labelCol : str，被预测量的名字

    featuresCol : list[str]，自变量的名字列表

    stepSize : float，梯度下降法里的学习速率

    maxIter : float，梯度下降法的迭代次数
    """
    w = np.random.randn(len(features_col))
    # 数据转换为dict形式，并将自变量转换为一个np.array
    df = data.rdd.map(
        lambda item: {"y": item[label_col], "x": np.array([item[i] for i in features_col])})
    # 将数据放到内存里，提高运算速度
    df.cache()
    for i in range(max_iter):
        # 计算线性回归模型的单点梯度以及数据的个数
        _gradient, num = df \
            .map(lambda item: -1 * (item["y"] - w.dot(item["x"])) * item["x"])\
            .map(lambda x: (x, 1))\
            .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        # 计算损失函数的平均单点梯度
        gradient = _gradient / num
        # 更新模型参数
        w -= step_size * gradient
    return w


if __name__ == "__main__":
    np.random.seed(4889)
    spark = start_spark()
    data = spark.createDataFrame(generate_data(1000))
    w = gradient_descent(data, "y", ["x", "const"])
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("模型的参数为：%s" % [2, 3])
    print("参数的估计值为：%s" % w)
