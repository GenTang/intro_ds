# -*- coding: UTF-8 -*-


from os import path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures



def evaluateModel(model, testData, features, labels, featurizer):
    """
    计算线性模型的均方差和决定系数

    参数
    ----
    model : LinearRegression, 训练完成的线性模型

    testData : DataFrame，测试数据

    features : list[str]，特征名列表

    labels : list[str]，标签名列表

    返回
    ----
    error : np.float64，均方差

    score : np.float64，决定系数
    """
    # 均方差(The mean squared error)，均方差越小越好
    error = np.mean(
        (model.predict(featurizer.fit_transform(testData[features])) - testData[labels]) ** 2)
    # 决定系数(Coefficient of determination)，决定系数越接近1越好
    score = model.score(featurizer.fit_transform(testData[features]), testData[labels])
    return error, score


def trainModel(trainData, features, labels, featurizer):
    """
    利用训练数据，估计模型参数

    参数
    ----
    trainData : DataFrame，训练数据集，包含特征和标签

    features : 特征名列表

    labels : 标签名列表

    返回
    ----
    model : LinearRegression, 训练好的线性模型
    """
    # 创建一个线性回归模型
    model = linear_model.LinearRegression(fit_intercept=False)
    # 训练模型，估计模型参数
    model.fit(featurizer.fit_transform(trainData[features]), trainData[labels])
    return model


def visualizeModel(model, featurizer, data, features, labels, evaluation):
    """
    模型可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 创建一个图形框
    fig = plt.figure(figsize=(10, 10), dpi=80)
    #fig = plt.figure()
    # 在图形框里只画一幅图
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        _visualization(ax, data, model[i], featurizer[i], evaluation[i], features, labels)
    plt.show()



def _visualization(ax, data, model, featurizer, evaluation, features, labels):
    """
    """
    # 画点图，用蓝色圆点表示原始数据
    ax.scatter(data[features], data[labels], color='b')
    # 画线图，用红色线条表示模型结果
    ax.plot(data[features], model.predict(featurizer.fit_transform(data[features])),
            color="r")
    # 显示均方差和决定系数
    ax.text(0.01, 0.99,
        u'%s%.3f\n%s%.3f'\
        % ("均方差：".decode("utf-8"), evaluation[0],
            "决定系数：".decode("utf-8"), evaluation[1]),
        style="italic", verticalalignment="top", horizontalalignment="left",
        transform=ax.transAxes, color="m", fontsize=13)


def overfitting(data):
    """
    """
    features = ["x"]
    labels = ["y"]
    # 划分训练集和测试集
    trainData = data[:15]
    testData = data[15:]
    featurizer = []
    overfittingModel = []
    overfittingEvaluation = []
    model = []
    evaluation = []
    for i in range(1, 11, 3):
        featurizer.append(PolynomialFeatures(degree=i))
        # 产生并训练模型
        overfittingModel.append(trainModel(trainData, features, labels, featurizer[-1]))
        model.append(trainModel(data, features, labels, featurizer[-1]))
        # 评价模型效果
        overfittingEvaluation.append(
            evaluateModel(overfittingModel[-1], testData, features, labels, featurizer[-1]))
        evaluation.append(evaluateModel(model[-1], data, features, labels, featurizer[-1]))
    # 图形化模型结果
    visualizeModel(model, featurizer, data, features, labels, evaluation)
    visualizeModel(overfittingModel, featurizer, data, features, labels, overfittingEvaluation)


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
    featurizer = PolynomialFeatures(degree=5)
    overfitting(data)
