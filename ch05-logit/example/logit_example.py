# -*- coding: UTF-8 -*-
"""
此脚本用于展示逻辑回归模型的搭建过程以及统计性质
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic


def modelSummary(re):
    """
    分析逻辑回归模型的统计性质
    """
    # 整体统计分析结果
    print(re.summary())
    # 用f test检验education_num的系数是否显著
    print("检验假设education_num的系数等于0：")
    print(re.f_test("education_num=0"))
    # 用f test检验两个假设是否同时成立
    print("检验假设education_num的系数等于0.32和hours_per_week的系数等于0.04同时成立：")
    print(re.f_test("education_num=0.32, hours_per_week=0.04"))


def transLabel(data):
    """
    将文字变量转化为数字变量
    """
    data["label_code"] = pd.Categorical(data.label).codes
    return data


def visualData(data):
    """
    画直方图，直观了解数据
    """
    data[["age", "hours_per_week", "education_num", "label_code"]].hist(
        rwidth=0.9, grid=False, figsize=(8, 8), alpha=0.6, color="grey")
    plt.show()


def analyseData(data):
    """
    通过统计方法，了解数据性质
    """
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("显示基本统计信息：")
    print(data.describe(include="all"))
    # 计算education_num, label交叉报表
    cross1 = pd.crosstab(pd.qcut(data["education_num"],  [0, .25, .5, .75, 1]), data["label"])
    print("显示education_num, label交叉报表：")
    print(cross1)
    # 将交叉报表图形化
    props = lambda key: {"color": "0.45"} if ' >50K' in key else {"color": "#C6E2FF"}
    mosaic(cross1[[" >50K", " <=50K"]].stack(), properties=props)
    # 计算hours_per_week, label交叉报表
    cross2 = pd.crosstab(pd.cut(data["hours_per_week"], 5), data["label"])
    # 将交叉报表归一化，利于分析数据
    cross2_norm = cross2.div(cross2.sum(1).astype(float), axis=0)
    print("显示hours_per_week, label交叉报表：")
    print(cross2_norm)
    # 图形化归一化后的交叉报表
    cross2_norm.plot(kind="bar", color=["#C6E2FF", "0.45"], rot=0)
    plt.show()
    


def trainModel(data):
    """
    搭建逻辑回归模型，并训练模型
    """
    formula = "label_code ~ age + education_num + capital_gain + capital_loss + hours_per_week"
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "label"]
    return data[cols]


def interpretModel(re):
    """
    理解模型结果

    参数
    ----
    re ：BinaryResults，训练好的逻辑回归模型
    """
    conf = re.conf_int()
    conf['OR'] = re.params
    # 计算各个变量对事件发生比的影响
    # conf里面的三列，分别对应着估计值的下界、上界和估计值本身
    conf.columns = ['2.5%', '97.5%', 'OR']
    print("各个变量对事件发生比的影响：")
    print(np.exp(conf))
    # 计算各个变量的边际效应
    print("各个变量的边际效应：")
    print(re.get_margeff(at="overall").summary())


def makePrediction(re, testSet, alpha=0.5):
    """
    使用训练好的模型对测试数据做预测
    """
    # 关闭pandas有关chain_assignment的警告
    pd.options.mode.chained_assignment = None
    # 计算事件发生的概率
    testSet["prob"] = re.predict(testSet)
    print("事件发生概率（预测概率）大于0.6的数据个数：")
    print(testSet[testSet["prob"] > 0.6].shape[0])  # 输出值为576
    print("事件发生概率（预测概率）大于0.5的数据个数：")
    print(testSet[testSet["prob"] > 0.5].shape[0])  # 输出值为834
    # 根据预测的概率，得出最终的预测
    testSet["pred"] = testSet.apply(lambda x: 1 if x["prob"] > alpha else 0, axis=1)
    return testSet


def evaluation(re):
    """
    计算预测结果的查准查全率以及f1

    参数
    ----
    re ：DataFrame，预测结果，里面包含两列：真实值‘lable_code’、预测值‘pred’
    """
    bins = np.array([0, 0.5, 1])
    label = re["label_code"]
    pred = re["pred"]
    tn, fp, fn, tp = np.histogram2d(label, pred, bins=bins)[0].flatten()
    precision = tp / (tp + fp)  # 0.707
    recall = tp / (tp + fn)  # 0.374
    f1 = 2 * precision * recall / (precision + recall)  # 0.490
    print("查准率: %.3f, 查全率: %.3f, f1: %.3f" % (precision, recall, f1))


def logitRegression(data):
    """
    逻辑回归模型分析步骤展示

    参数
    ----
    data ：DataFrame，建模数据
    """
    data = transLabel(data)
    visualData(data)
    analyseData(data)
    # 将数据分为训练集和测试集
    trainSet, testSet = train_test_split(data, test_size=0.2, random_state=2310)
    # 训练模型并分析模型效果
    re = trainModel(trainSet)
    modelSummary(re)
    interpretModel(re)
    re = makePrediction(re, testSet)
    evaluation(re)


if __name__ == "__main__":
    # 设置显示格式
    pd.set_option('display.width', 1000)
    homePath = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        dataPath = "%s\\data\\adult.data" % homePath
    else:
        dataPath = "%s/data/adult.data" % homePath
    data = readData(dataPath)
    logitRegression(data)
