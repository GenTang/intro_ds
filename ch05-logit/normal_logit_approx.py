# -*- coding: UTF-8 -*-
"""
此脚本用于展示可用逻辑分布近似正态分布
"""


import sys

import matplotlib.pyplot as plt
from scipy.stats import norm, logistic
import numpy as np


def visualization():
    """
    使用逻辑分布近似正态分布。
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams['font.sans-serif']=['SimHei']
    # 正确显示负号
    plt.rcParams['axes.unicode_minus']=False
    # 创建一个图形框，在里面只画一幅图
    fig = plt.figure(figsize=(9, 6), dpi=100)
    ax = fig.add_subplot(111)
    x = np.linspace(-5, 5, 100)
    alpha = 1.702
    normal = norm.cdf(x)
    logit = logistic.cdf(alpha * x)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(x, normal,  
            label=u'%s' % "标准正态分布：" +\
            r"$F(x) = \int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}{\rm e}^{-\frac{t^{2}}{2}}$"
            + r"${\rm d}t$")
        ax.plot(x, logit, "k-.", label=u'%s' %
            r"最佳近似的逻辑分布：$F(x) = \frac{1}{1 + {\rm e}^{-1.702x}}$")
        ax.set_xlabel('$x$')
        ax.set_ylabel(u'%s：$F(x)$' % "累积分布函数")
    else:
        ax.plot(x, normal,  
            label=u'%s' % "标准正态分布：".decode("utf-8") +\
            r"$F(x) = \int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}{\rm e}^{-\frac{t^{2}}{2}}$"
            + r"${\rm d}t$")
        ax.plot(x, logit, "k-.", label=u'%s' % 
            r"最佳近似的逻辑分布：$F(x) = \frac{1}{1 + {\rm e}^{-1.702x}}$".decode("utf-8"))
        ax.set_xlabel('$x$')
        ax.set_ylabel(u'%s：$F(x)$' % "累积分布函数".decode("utf-8"))
    legend = plt.legend(shadow=True, fontsize=13)
    plt.show()


if __name__ == "__main__":
    visualization()
