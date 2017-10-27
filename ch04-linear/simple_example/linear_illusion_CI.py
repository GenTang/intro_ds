# -*- coding: UTF-8 -*-


from os import path

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


def generateRandomVar():
    """
    """
    np.random.seed(4873)
    return np.random.randint(2, size=20) 


def evaluateModel(res):
    """
    分析线性回归模型的统计性质
    """
    # 整体统计分析结果
    print res.summary()
    # 用f test检测x对应的系数a是否显著
    print "检验假设z的系数等于0："
    print res.f_test("z=0")
    # 用f test检测常量b是否显著
    print "检测假设const的系数等于0："
    print res.f_test("const=0")
    # 用f test检测a=1, b=0同时成立的显著性
    print "检测假设z和const的系数同时等于0："
    print res.f_test(["z=0", "const=0"])


def trainModel(X, Y):
    """
    训练模型
    """
    model = sm.OLS(Y, X)
    res = model.fit()
    return res


def confidenceInterval(data):
    """
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    _X = data[features]
    # 加入新的随机变量，次变量的系数应为0
    _X["z"] = generateRandomVar()
    # 加入常量变量
    X = sm.add_constant(_X)
    res = trainModel(X, Y)
    evaluateModel(res)
    

def generateData():
    """
    生成模型数据
    """
    np.random.seed(5320)
    x = np.array(range(0, 20)) / 2
    error = np.round(np.random.randn(20), 2)
    y = 0.05 * x + error
    # 新加入的无关变量z恒等于1
    z = np.zeros(20) + 1
    return pd.DataFrame({"x": x, "z": z, "y": y})


def wrongCoef():
    """
    由于新变量的加入，正效应变为负效应
    """
    features = ["x", "z"]
    labels = ["y"]
    data = generateData()
    X = data[features]
    Y = data[labels]
    # 没有多余变量时，x系数符号估计正确，为正
    model = sm.OLS(Y, X["x"])
    res = model.fit()
    print "没有加入新变量时："
    print res.summary()
    # 加入多余变量时，x系数符号估计错误，为负
    model1 = sm.OLS(Y, X)
    res1 = model1.fit()
    print "加入新变量后："
    print res1.summary()


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data


if __name__ == "__main__":
    homePath = path.dirname(path.abspath(__file__))
    dataPath = "%s/data/simple_example.csv" % homePath
    data = readData(dataPath)
    print "***************************************************"
    print "加入不相关的新变量，新变量的系数被错误估计为不等于0"
    print "***************************************************"
    confidenceInterval(data)
    print "**********************************************"
    print "加入不相关的新变量，旧变量系数的符号被错误估计"
    print "**********************************************"
    wrongCoef()
