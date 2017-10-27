# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何处理定量变量
"""


from os import path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import scipy.stats as scs
from sets import Set


def transLabel(data):
    """
    将文字变量转化为数字变量
    """
    data["label_code"] = pd.Categorical(data.label).codes
    return data


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    cols = ["age", "education_num",
        "capital_gain", "capital_loss", "hours_per_week", "label"]
    return data[cols]


def transFeature(data, category):
    """
    根据传入的分段区间，将每星期工作时间转换为定量变量

    参数
    ----
    data : DataFrame，建模数据

    category : list，分段区间
    """
    labels = ["{0}-{1}".format(category[i], category[i+1]) for i in range(len(category) - 1)]
    data["hours_per_week_group"] = pd.cut(data["hours_per_week"],
        category, include_lowest=True, labels=labels)
    return data


def getCategory(data):
    """
    基于卡方检验，得到每星期工作时间的“最优”分段
    """
    interval = [data["hours_per_week"].min(), data["hours_per_week"].max()]
    _category = doDivide(data, interval)
    s = Set()
    for i in _category:
        s = s.union(Set(i))
    category = list(s)
    category.sort()
    return category


def doDivide(data, interval):
    """
    使用贪心算法，得到“最优”的分段
    """
    category = []
    pValue, chi2, index = divideData(data, interval[0], interval[1])
    if chi2 < 15:
        category.append(interval)
    else:
        category += doDivide(data, [interval[0], index])
        category += doDivide(data, [index, interval[1]])
    return category


def divideData(data, minValue, maxValue):
    """
    遍历所有可能的分段，返回卡方统计量最高的分段
    """
    maxChi2 = 0
    index = -1
    maxPValue = 0
    for i in range(minValue+1, maxValue):
        category = pd.cut(data["hours_per_week"], [minValue, i, maxValue],
            include_lowest=True)
        cross = pd.crosstab(data["label"], category)
        chi2, pValue, _, _ = scs.chi2_contingency(cross)
        if chi2 > maxChi2:
            maxPValue = pValue
            maxChi2 = chi2
            index = i
    return maxPValue, maxChi2, index


def trainModel(data):
    """
    利用新生成的定性变量搭建逻辑回归模型，并训练模型
    """
    formula = """label_code ~ education_num + capital_gain
        + capital_loss + C(hours_per_week_group)"""
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def baseModel(data):
    """
    原有模型
    """
    formula = "label_code ~ education_num + capital_gain + capital_loss + hours_per_week"
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def makePrediction(re, testSet, alpha=0.5):
    """
    使用训练好的模型对测试数据做预测
    """
    # 关闭pandas有关chain_assignment的警告
    pd.options.mode.chained_assignment = None
    # 计算事件发生的概率
    data = testSet.copy()
    data["prob"] = re.predict(data)
    # 根据预测的概率，得出最终的预测
    data["pred"] = data.apply(lambda x: 1 if x["prob"] > alpha else 0, axis=1)
    return data


def evaluation(newRe, baseRe):
    """
    展示将每星期工作时间离散化之后，模型效果的变化
    """
    fpr, tpr, _ = metrics.roc_curve(newRe["label_code"], newRe["prob"])
    auc = metrics.auc(fpr, tpr)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    ax.set_title("%s" % "ROC曲线".decode("utf-8"))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(fpr, tpr, "k", label="%s; %s = %0.3f" % ("转换后的ROC曲线".decode("utf-8"),
        "曲线下面积（AUC）".decode("utf-8"), auc))
    # 绘制原模型的ROC曲线
    fpr, tpr, _ = metrics.roc_curve(baseRe["label_code"], baseRe["prob"])
    auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, "b-.", label="%s; %s = %0.3f" % ("转换前的ROC曲线".decode("utf-8"),
        "曲线下面积（AUC）".decode("utf-8"), auc))
    legend = plt.legend(shadow=True)
    plt.show()


def logitRegression(data):
    """
    逻辑回归模型分析步骤展示

    参数
    ----
    data ：DataFrame，建模数据
    """
    data = transLabel(data)
    # 将数据分为训练集和测试集
    trainSet, testSet = train_test_split(data, test_size=0.2, random_state=2310)
    category = getCategory(trainSet)
    #category = range(0, 105, 10)
    trainSet = transFeature(trainSet, category)
    testSet = transFeature(testSet, category)
    # 训练模型并分析模型效果
    newRe = trainModel(trainSet)
    print newRe.summary()
    newRe = makePrediction(newRe, testSet)
    # 计算原模型预测结果
    baseRe = baseModel(trainSet)
    baseRe = makePrediction(baseRe, testSet)
    evaluation(newRe, baseRe)


if __name__ == "__main__":
    # 设置显示格式
    pd.set_option('display.width', 1000)
    homePath = path.dirname(path.abspath(__file__))
    dataPath = "%s/data/adult.data" % homePath
    data = readData(dataPath)
    logitRegression(data)
