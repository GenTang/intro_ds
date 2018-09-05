# -*- coding: UTF-8 -*-
"""
此脚本用于展示内生性对模型的影响
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import scipy.stats.stats as scss


def generate_linear_data(n):
    """
    产生有内生变量的线性回归模型数据
    """
    data = pd.DataFrame()
    np.random.seed(4060)
    cov = [[1, 0.9], [0.9, 1]]
    _X = np.random.multivariate_normal([0, 0], cov, n)
    # 变量X1和变量IV1之间存在关联关系，这是使得IV1成为工具变量
    X1 = _X[:, 0]
    IV1 = _X[:, 1]
    X2 = np.random.normal(size=n)
    error = np.random.normal(size=n) * 2
    Y = 10 * X1 + 20 * X2 + 30 + error
    # 加入度量误差，这将使observationX成为内生变量
    ob_error = np.random.normal(size=n) * 2
    observation_x = X1 + ob_error
    data["X1"] = observation_x
    data["X2"] = X2
    data["Y"] = Y
    data["realX1"] = X1
    data["IV1"] = IV1
    data["error"] = Y - 10 * observation_x - 20 * X2 - 30
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    axes = scatter_matrix(data[["X1", "error", "IV1"]], alpha=1, diagonal='kde',
                          range_padding=0.9, figsize=(8, 8))
    corr = data[["X1", "error", "IV1"]].corr(method="pearson").as_matrix()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            axes[i, j].annotate("%s: %.3f" % ("相关系数", corr[i, j]),
                                (0.5, 0.9), xycoords='axes fraction', ha='center', va='center')
        else:
            axes[i, j].annotate("%s: %.3f" % ("相关系数".decode("utf-8"), corr[i, j]),
                                (0.5, 0.9), xycoords='axes fraction', ha='center', va='center')
    plt.show()
    return data


def generate_logit_data(n):
    """
    产生有内生变量的logit回归模型数据
    """
    data = pd.DataFrame()
    np.random.seed(4060)
    cov = [[1, 0.9], [0.9, 1]]
    _X = np.random.multivariate_normal([0, 0], cov, n)
    # 变量X1和变量IV1之间存在关联关系，这是使得IV1成为工具变量
    X1 = _X[:, 0]
    IV1 = _X[:, 1]
    X2 = np.random.normal(size=n)
    error = np.random.logistic(size=n)
    Y = (1 * X1 - 1 * X2 + error > 0) + 0
    # 加入度量误差，这将使observationX成为内生变量
    ob_error = np.random.normal(size=n)
    observation_x = X1 + ob_error
    data["X1"] = observation_x
    data["X2"] = X2
    data["Y"] = Y
    data["realX1"] = X1
    data["IV1"] = IV1
    return data


def IV_regression(data):
    """
    使用工具变量估计模型参数
    """
    # 第一步回归
    re = sm.OLS.from_formula("X1 ~ X2 + IV1", data=data).fit()
    data["X1_resid"] = re.resid
    # 第二步回归
    re1 = sm.OLS.from_formula("Y ~ X1 + X2 + X1_resid", data=data).fit()
    print("使用工具变量")
    print(re1.summary())


def IV_regression2(data):
    """
    使用工具变量估计模型参数
    """
    data = sm.add_constant(data[["Y", "X1", "X2", "IV1"]])
    model = IV2SLS(data[["Y"]], data[["X1", "X2", "const"]], data[["IV1", "X2", "const"]])
    re = model.fit()
    print("使用工具变量")
    print(re.summary())
    print("Durbin–Wu–Hausman检验")
    print(re.spec_hausman())


def IV_logit(data):
    """
    使用工具变量估计logit回归参数
    """
    # 使用工具变量
    # 第一步回归
    re_tmp = sm.OLS.from_formula("X1 ~ X2 + IV1", data=data).fit()
    data["X1_resid"] = re_tmp.resid
    # 第二步回归
    re = sm.Logit.from_formula("Y ~ X2 + X1 + X1_resid", data=data).fit()
    print("使用工具变量")
    print(re.summary())


def linear_model():
    """
    内生性对线性模型的影响
    """
    data = generate_linear_data(1000)
    re = sm.OLS.from_formula("Y ~ realX1 + X2", data=data).fit()
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("正常模型")
    print(re.summary())
    re1 = sm.OLS.from_formula("Y ~ X2 + X1", data=data).fit()
    print("内生性")
    print(re1.summary())
    re2 = sm.OLS.from_formula("Y ~ X2 + X1 + IV1", data=data).fit()
    print("直接使用工具变量会引起共线性问题")
    print(re2.summary())
    IV_regression(data)


def logit_model():
    """
    内生性对logit回归的影响
    """
    data = generate_logit_data(1000)
    re = sm.Logit.from_formula("Y ~ X1 + X2", data=data).fit()
    print("内生性")
    print(re.summary())
    re1 = sm.Logit.from_formula("Y ~ X1 + X2 + IV1", data=data).fit()
    print("直接使用工具变量会引起共线性问题")
    print(re1.summary())
    IV_logit(data)


if __name__ == "__main__":
    linear_model()
    logit_model()
