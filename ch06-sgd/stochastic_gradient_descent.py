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
from utils import create_summary_writer, generate_linear_data, create_linear_model


def stochastic_gradient_descent(X, Y, model, learning_rate=0.01,
                                mini_batch_fraction=0.01, epoch=10000, tol=1.e-6):
    """
    利用随机梯度下降法训练模型。

    参数
    ----
    X : np.array, 自变量数据

    Y : np.array, 因变量数据

    model : dict, 里面包含模型的参数，损失函数，自变量，应变量
    """
    # 确定最优化算法
    method = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
        summary_writer = create_summary_writer("logs\\stochastic_gradient_descent")
    else:
        summary_writer = create_summary_writer("logs/stochastic_gradient_descent")
    # tensorflow开始运行
    sess = tf.Session()
    # 产生初始参数
    init = tf.global_variables_initializer()
    # 用之前产生的初始参数初始化模型
    sess.run(init)
    # 迭代梯度下降法
    step = 0
    batch_size = int(X.shape[0] * mini_batch_fraction)
    batch_num = int(math.ceil(1 / mini_batch_fraction))
    prev_loss = np.inf
    diff = np.inf
    # 当损失函数的变动小于阈值或达到最大训练轮次，则停止迭代
    while (step < epoch) & (diff > tol):
        for i in range(batch_num):
            # 选取小批次训练数据
            batch_x = X[i * batch_size: (i + 1) * batch_size]
            batch_y = Y[i * batch_size: (i + 1) * batch_size]
            # 迭代模型参数
            sess.run([optimizer],
                     feed_dict={model["independent_variable"]: batch_x,
                                model["dependent_variable"]: batch_y})
            # 计算损失函数并写入日志
            summary_str, loss = sess.run(
                [summary, model["loss_function"]],
                feed_dict={model["independent_variable"]: X,
                           model["dependent_variable"]: Y})
            # 将运行细节写入目录
            summary_writer.add_summary(summary_str, step * batch_num + i)
            # 计算损失函数的变动
            diff = abs(prev_loss - loss)
            prev_loss = loss
            if diff <= tol:
                break
        step += 1
    summary_writer.close()
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
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
    X, Y = generate_linear_data(dimension, num)
    # 定义模型
    model = create_linear_model(dimension)
    # 使用梯度下降法，估计模型参数
    stochastic_gradient_descent(X, Y, model)


if __name__ == "__main__":
    run()
