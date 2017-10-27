# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用spark-sklearn并行地训练模型
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from pyspark import SparkContext, SparkConf
from spark_sklearn import GridSearchCV


def generateData(n):
    """
    生成训练模型的数据
    """
    x = np.random.randn(n) * 10
    # 变量z是不相关变量
    z = np.random.randint(10, size=n)
    error = np.round(np.random.randn(n), 2)
    y = x + error
    return pd.DataFrame({"x": x, "y": y, "z": z})


def gridSearch(sc, data, label, features):
    """
    使用grid search寻找最优的超参数
    """
    # 产生备选的超参数集
    parameters = {"alpha": 10 ** np.linspace(-4, 0, 45)}
    # Lasso模型里有超参数alpha，表示惩罚项的权重
    la = Lasso()
    gs = GridSearchCV(sc, la, parameters)
    gs.fit(data[features], data[label])
    return gs


def startSpark():
    """
    创建SparkContext，这是Spark程序的入口
    """
    conf = SparkConf().setAppName("grid search example")
    sc = SparkContext(conf=conf)
    return sc


if __name__ == "__main__":
    np.random.seed(2048)
    sc = startSpark()
    data = generateData(300)
    gs = gridSearch(sc, data, "y", ["x", "z"])
    print "最优的超参数alpha为：%s" % gs.best_params_
    print "相应的模型参数为：%s" %\
        np.append(gs.best_estimator_.coef_, gs.best_estimator_.intercept_)
    print "正确的模型参数为：[1., 0, 0]"
