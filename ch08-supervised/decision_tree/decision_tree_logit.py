# -*- coding: UTF-8 -*-
"""
此脚本用于展示决策树联结逻辑回归模型
"""


import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def generate_data(n):
    """
    生成训练数据
    """
    X, y = make_classification(n_samples=n, n_features=4)
    data = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    data["y"] = y
    return data


def train_model(data, features, label):
    """
    分别使用逻辑回归、决策树和决策树+逻辑回归建模
    """
    res = {}
    train_data, test_data = train_test_split(data, test_size=0.5)
    # 单独使用逻辑回归
    logit_model = LogisticRegression()
    logit_model.fit(train_data[features], train_data[label])
    logit_prob = logit_model.predict_proba(test_data[features])[:, 1]
    res["logit"] = roc_curve(test_data[label], logit_prob)
    # 单独使用决策树
    dt_model = DecisionTreeClassifier(max_depth=2)
    dt_model.fit(train_data[features], train_data[label])
    dt_prob = dt_model.predict_proba(test_data[features])[:, 1]
    res["DT"] = roc_curve(test_data[label], dt_prob)
    # 决策树和逻辑回归联结
    # 为了防止过拟合，使用不同的数据训练决策树和逻辑回归
    train_DT, train_LR = train_test_split(train_data, test_size=0.5)
    # 使用决策树对前两个变量做变换
    m = 2
    _dt = DecisionTreeClassifier(max_depth=2)
    _dt.fit(train_DT[features[:m]], train_DT[label])
    leaf_node = _dt.apply(train_DT[features[:m]]).reshape(-1, 1)
    coder = OneHotEncoder()
    coder.fit(leaf_node)
    new_feature = np.c_[
        coder.transform(_dt.apply(train_LR[features[:m]]).reshape(-1, 1)).toarray(),
        train_LR[features[m:]]]
    _logit = LogisticRegression()
    _logit.fit(new_feature[:, 1:], train_LR[label])
    test_feature = np.c_[
        coder.transform(_dt.apply(test_data[features[:m]]).reshape(-1, 1)).toarray(),
        test_data[features[m:]]]
    dt_logit_prob = _logit.predict_proba(test_feature[:, 1:])[:, 1]
    res["DT + logit"] = roc_curve(test_data[label], dt_logit_prob)
    return res


def visualize(re):
    """
    将模型结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    styles = ["k--", "r-.", "b"]
    model = ["logit", "DT", "DT + logit"]
    for i, s in zip(model, styles):
        fpr, tpr, _ = re[i]
        _auc = auc(fpr, tpr)
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            ax.plot(fpr, tpr, s, label="%s:%s; %s=%0.2f" % ("模型", i,
                                                            "曲线下面积（AUC）", _auc))
        else:
            ax.plot(fpr, tpr, s,
                    label="%s:%s; %s=%0.2f" % ("模型".decode("utf-8"), i,
                                               "曲线下面积（AUC）".decode("utf-8"), _auc))
    legend = plt.legend(loc=4, shadow=True)
    plt.show()


if __name__ == "__main__":
    np.random.seed(4040)
    data = generate_data(4000)
    re = train_model(data, ["x1", "x2", "x3", "x4"], "y")
    visualize(re)
