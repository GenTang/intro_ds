# -*- coding: UTF-8 -*-
"""
此脚本用于展示二分类问题的不同评估指标
"""


import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def transLabel(data):
    """
    将文字变量转化为数字变量
    """
    data["label_code"] = pd.Categorical(data.label).codes
    return data


def trainModel(data, features, labels):
    """
    搭建逻辑回归模型，并训练模型
    """
    model = LogisticRegression(C=1e4)
    model.fit(data[features], data[labels])
    return model


def readData(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "label"]
    return data[cols]


def logitRegression(data):
    """
    训练模型，并画出模型的ROC曲线
    """
    data = transLabel(data)
    features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    labels = "label_code"
    trainSet, testSet = train_test_split(data, test_size=0.3, random_state=2310)
    model = trainModel(trainSet, features, labels)
    # 得到预测的概率
    preds = model.predict_proba(testSet[features])[:,1]
    re = pd.DataFrame()
    re["pred_proba"] = preds
    re["label"] = testSet[labels].tolist()
    return re
    

def PrecisionRecallFscore(pred, label, beta=1):
    """
    计算预测结果的Precision, Recall以及Fscore
    """
    bins = np.array([0, 0.5, 1])
    tn, fp, fn, tp = np.histogram2d(label, pred, bins=bins)[0].flatten()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return precision, recall, fscore


def AUC(predProb, label):
    """
    计算False positive rate, True positive rate和AUC
    """
    # 得到False positive rate和True positive rate
    fpr, tpr, _ = metrics.roc_curve(label, predProb)
    # 得到AUC
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc


def visualizePrecRecFscoreByAlphas(re):
    """
    展示不同的阈值下，模型的Precision, Recall和F1-score
    """
    alphas = np.arange(0.1, 1.01, 0.05)
    precs = []
    recs = []
    fscores = []
    for alpha in alphas:
        pred = re.apply(lambda x: 1 if x["pred_proba"] > alpha else 0, axis=1)
        label = re["label"]
        prec, rec, fscore = PrecisionRecallFscore(pred, label)
        precs.append(prec)
        recs.append(rec)
        fscores.append(fscore)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, precs, "r--", label="Precision")
    ax.plot(alphas, recs, "k-.", label="Recall")
    ax.plot(alphas, fscores, label="F1-score")
    ax.set_xlabel(r"$\alpha$")
    ax.set_xlim([0.1, 1])
    legend = plt.legend(shadow=True)
    plt.show()


def visualizePrecRecFscoreByBetas(re):
    """
    展示不同的beta下，模型的Precision, Recall和F1-score
    """
    betas = np.arange(0, 10, 0.1)
    precs = []
    recs = []
    fscores = []
    pred = re.apply(lambda x: 1 if x["pred_proba"] > 0.5 else 0, axis=1)
    label = re["label"]
    for beta in betas:
        prec, rec, fscore = PrecisionRecallFscore(pred, label, beta)
        precs.append(prec)
        recs.append(rec)
        fscores.append(fscore)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(betas, precs, "r--", label="Precision")
    ax.plot(betas, recs, "k-.", label="Recall")
    ax.plot(betas, fscores, label="F-score")
    ax.set_xlabel(r"$\beta$")
    legend = plt.legend(shadow=True)
    plt.show()


def visualizeRoc(re):
    """
    绘制ROC曲线
    """
    fpr, tpr, auc = AUC(re["pred_proba"], re["label"])
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"]=["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(1, 1, 1)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_title("%s" % "ROC曲线")
    else:
        ax.set_title("%s" % "ROC曲线".decode("utf-8"))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(fpr, tpr, "k", label="%s; %s = %0.2f" % ("ROC曲线",
            "曲线下面积（AUC）", auc))
    else:
        ax.plot(fpr, tpr, "k", label="%s; %s = %0.2f" % ("ROC曲线".decode("utf-8"),
            "曲线下面积（AUC）".decode("utf-8"), auc))
    ax.fill_between(fpr, tpr, color="grey", alpha=0.6)
    legend = plt.legend(shadow=True)
    plt.show()


def evaluation(re):
    """
    用图像化的形式展示评估指标
    """
    visualizePrecRecFscoreByAlphas(re)
    visualizePrecRecFscoreByBetas(re)
    visualizeRoc(re)


def run(dataPath):
    """
    训练模型，并使用Precisiion, Recall, F-score以及AUC评估模型
    """
    data = readData(dataPath)
    re = logitRegression(data)
    evaluation(re)


if __name__ == "__main__":
    homePath = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        dataPath = "%s\\data\\adult.data" % homePath
    else:
        dataPath = "%s/data/adult.data" % homePath
    run(dataPath)
