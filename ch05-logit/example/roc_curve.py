# -*- coding: UTF-8 -*-
"""
此脚本用于展示ROC曲线和AUC
"""


import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def trans_label(data):
    """
    将文字变量转化为数字变量
    """
    data["label_code"] = pd.Categorical(data.label).codes
    return data


def train_model(data, features, labels):
    """
    搭建逻辑回归模型，并训练模型
    """
    model = LogisticRegression()
    model.fit(data[features], data[labels])
    return model


def read_data(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "label"]
    return data[cols]


def visualize_roc(fpr, tpr, auc):
    """
    根据给定的fpr和tpr，绘制ROC曲线
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
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
        ax.plot(fpr, tpr, "k", label="%s; %s = %0.2f" % ("ROC曲线", "曲线下面积（AUC）", auc))
    else:
        ax.plot(fpr, tpr, "k",
                label="%s; %s = %0.2f" % ("ROC曲线".decode("utf-8"),
                                          "曲线下面积（AUC）".decode("utf-8"), auc))
    ax.fill_between(fpr, tpr, color="grey", alpha=0.6)
    legend = plt.legend(shadow=True)
    plt.show()


def logit_regression(data):
    """
    训练模型，并画出模型的ROC曲线
    """
    data = trans_label(data)
    features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    labels = "label_code"
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=2310)
    model = train_model(train_set, features, labels)
    # 得到预测的概率
    preds = model.predict_proba(test_set[features])[:, 1]
    # 得到False positive rate和True positive rate
    fpr, tpr, _ = metrics.roc_curve(test_set[labels], preds)
    # 得到AUC
    auc = metrics.auc(fpr, tpr)
    visualize_roc(fpr, tpr, auc)


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data\\adult.data" % home_path
    else:
        data_path = "%s/data/adult.data" % home_path
    data = read_data(data_path)
    logit_regression(data)
