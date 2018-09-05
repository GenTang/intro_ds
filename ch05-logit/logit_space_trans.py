# -*- coding: UTF-8 -*-
"""
此脚本用于展示sigmod空间变换
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


def generate_data(size):
    """
    """
    np.random.seed(1220)
    x = np.random.normal(size=size)
    y = (x > 0).astype(np.float)
    x *= 1.5
    x += 0.5 * np.random.normal(size=size)
    x = x[:, np.newaxis]
    return x, y


def prob_one(x):
    """
    """
    return 1 / (1 + np.exp(-x))


def space_trans(model, x, y):
    """
    """
    prob = prob_one(x * model.coef_ + model.intercept_).ravel()
    loss = (prob - y) * np.log(prob * (1 - prob))
    trans_y = np.log(prob / (1 - prob)) + loss
    return trans_y


def visual_linear(model, x, y):
    """
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 正确显示负号
    plt.rcParams["axes.unicode_minus"] = False
    # 创建一个图形框，在里面画两幅画
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # 将原始数据表现在图上
    ax.scatter(x[:40], y[:40], color="black")
    # 画拟合的直线
    xline = np.linspace(-3, 5, 100)[:, np.newaxis]
    yline = model.predict(xline)
    ax.plot(xline.ravel(), yline, "r")
    ax1 = fig.add_subplot(1, 2, 2)
    residual = y - model.predict(x)
    n, bins, _ = ax1.hist(residual, 40, normed=1, facecolor="grey", rwidth=0.8, alpha=0.6)
    # 用多项式拟合得到的直方图
    z1 = np.polyfit(bins[:-1], n, 10)
    p1 = np.poly1d(z1)
    ax1.plot(bins[:-1], p1(bins[:-1]), "r-.")
    ax1.set_xlabel("$residual$")
    plt.show()


def visualization(model, x, y):
    """
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 正确显示负号
    plt.rcParams["axes.unicode_minus"] = False
    # 创建一个图形框，在里面画两幅画
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # 将原始数据表现在图上
    ax.scatter(x, y, color="black")
    # logit函数表现在图上
    xlogit = np.linspace(-3, 5, 100)
    prob = prob_one(xlogit * model.coef_ + model.intercept_).ravel()
    ax.plot(xlogit, prob, "r-.")
    # 随机选择两个点
    example_x = x[[8, 17]]
    example_y = y[[8, 17]]
    ax.scatter(example_x, example_y, marker=(5, 1), color="blue", s=200, facecolors="none")
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_xlabel(r"$\tilde{x}$")
    ax1.set_ylabel(r"$\tilde{y}$")
    ax1.text(0.05, 0.95,
             r"$\tilde{x} = x$" + "\n"
             + r"$\tilde{y} = \ln(\frac{p}{1-p}) + (p - y)\ln(1 - p)p$" + "\n"
             + r"$p = \frac{1}{{\rm e}^{-1.4x + 0.2}}$",
             style="italic", verticalalignment="top", horizontalalignment="left",
             transform=ax1.transAxes, color="m", fontsize=13)
    # 将原始数据表现在新的空间里
    trans_y = space_trans(model, x, y)
    ax1.scatter(x, trans_y, color="black")
    # 将logit函数表现在新的空间里
    trans_prob = space_trans(model, xlogit, prob)
    ax1.plot(xlogit, trans_prob, "r-.")
    # 标出之前选择的示例点
    trans_example_y = space_trans(model, example_x, example_y)
    ax1.scatter(example_x, trans_example_y,
                marker=(5, 1), color="blue", s=200, facecolors="none")
    plt.show()


if __name__ == "__main__":
    x, y = generate_data(20)
    model = linear_model.LogisticRegression()
    model.fit(x, y)
    xx, yy = generate_data(10000)
    ols = linear_model.LinearRegression()
    ols.fit(xx, yy)
    visual_linear(ols, xx, yy)
    visualization(model, x, y)
