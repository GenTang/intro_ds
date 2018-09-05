# -*- coding: UTF-8 -*-
"""
此脚本用于随机生成线性模型数据、定义模型以及其他工具
"""


import numpy as np
import tensorflow as tf


def generate_linear_data(dimension, num):
    """
    随机产生线性模型数据

    参数
    ----
    dimension ：int，自变量个数

    num ：int，数据个数

    返回
    ----
    x ：np.array，自变量

    y ：np.array，因变量
    """
    np.random.seed(1024)
    beta = np.array(range(dimension)) + 1
    x = np.random.random((num, dimension))
    epsilon = np.random.random((num, 1))
    # 将被预测值写成矩阵形式，会极大加快速度
    y = x.dot(beta).reshape((-1, 1)) + epsilon
    return x, y


def create_linear_model(dimension):
    """
    搭建模型，包括数据中的自变量，应变量和损失函数

    参数
    ----
    dimension : int，自变量的个数

    返回
    ----
    model ：dict，里面包含模型的参数，损失函数，自变量，应变量
    """
    np.random.seed(1024)
    # 定义自变量和应变量
    x = tf.placeholder(tf.float64, shape=[None, dimension], name='x')
    # 将被预测值写成矩阵形式，会极大加快速度
    y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
    # 定义参数估计值和预测值
    beta_pred = tf.Variable(np.random.random([dimension, 1]))
    y_pred = tf.matmul(x, beta_pred, name="y_pred")
    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y_pred - y))
    model = {"loss_function": loss, "independent_variable": x,
             "dependent_variable": y, "prediction": y_pred, "model_params": beta_pred}
    return model


def create_summary_writer(log_path):
    """
    检查所给路径是否已存在，如果存在删除原有日志。并创建日志写入对象

    参数
    ----
    logPath ：string，日志存储路径

    返回
    ----
    summaryWriter ：FileWriter，日志写入器
    """
    if tf.gfile.Exists(log_path):
        tf.gfile.DeleteRecursively(log_path)
    summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
    return summary_writer
