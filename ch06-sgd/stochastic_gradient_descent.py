# -*- coding: UTF-8 -*-
"""
此脚本用于展示随机梯度下降法的实现
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
import math
from utils import createSummaryWriter, generateLinearData, createLinearModel


def stochasticGradientDescent(X, Y, model, learningRate=0.01,
        miniBatchFraction=0.01, epoch=10000, tol=1.e-6):
    """
    利用随机梯度下降法训练模型。

    参数
    ----
    X : np.array, 自变量数据

    Y : np.array, 因变量数据

    model : dict, 里面包含模型的参数，损失函数，自变量，应变量
    """
    # 确定最优化算法
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model["loss_function"])
    # 增加日志
    tf.summary.scalar("loss_function", model["loss_function"])
    tf.summary.histogram("params", model["model_params"])
    tf.summary.scalar("first_param", tf.reduce_mean(model["model_params"][0]))
    tf.summary.scalar("last_param", tf.reduce_mean(model["model_params"][-1]))
    summary = tf.summary.merge_all()
    # 在程序运行结束之后，运行如下命令，查看日志
    # tensorboard --logdir logs/
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        summaryWriter = createSummaryWriter("logs\\stochastic_gradient_descent")
    else:
        summaryWriter = createSummaryWriter("logs/stochastic_gradient_descent")
    # tensorflow开始运行
    sess = tf.Session()
    # 产生初始参数
    init = tf.global_variables_initializer()
    # 用之前产生的初始参数初始化模型
    sess.run(init)
    # 迭代梯度下降法
    step = 0
    batchSize = int(X.shape[0] * miniBatchFraction)
    batchNum = int(math.ceil(1 / miniBatchFraction))
    prevLoss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大训练轮次，则停止迭代
    while (step < epoch) & (diff > tol):
        for i in range(batchNum):
            # 选取小批次训练数据
            batchX = X[i * batchSize: (i + 1) * batchSize]
            batchY = Y[i * batchSize: (i + 1) * batchSize]
            # 迭代模型参数
            sess.run([optimizer],
                feed_dict={model["independent_variable"]: batchX,
                    model["dependent_variable"]: batchY})
            # 计算损失函数并写入日志
            summaryStr, loss = sess.run(
                [summary, model["loss_function"]],
                feed_dict={model["independent_variable"]: X,
                    model["dependent_variable"]: Y})
            # 将运行细节写入目录
            summaryWriter.add_summary(summaryStr, step * batchNum + i)
            # 计算损失函数的变动
            diff = abs(prevLoss - loss)
            prevLoss = loss
            if diff <= tol:
                break
        step += 1
    summaryWriter.close()
    # 输出最终结果
    print("模型参数：\n%s" % sess.run(model["model_params"]))
    print("训练轮次：%s" % step)
    print("损失函数值：%s" % loss)


def run():
    """
    程序入口
    """
    # dimension表示自变量的个数，num表示数据集里数据的个数。
    dimension = 30
    num = 10000
    # 随机产生模型数据
    X, Y = generateLinearData(dimension, num)
    # 定义模型
    model = createLinearModel(dimension)
    # 使用梯度下降法，估计模型参数
    stochasticGradientDescent(X, Y, model)

    
if __name__ == "__main__":
    run()
