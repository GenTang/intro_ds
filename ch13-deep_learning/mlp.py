# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用tensorflow来实现MLP，
并在训练过程中使用防止过拟合的方案
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os
import sys

if sys.version_info[0] == 3:
    from functools import reduce
else:
    pass

from util import load_data

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ANN(object):

    def __init__(self, size, log_path, train_set, validation_set, test_set, lambda_=1e-3):
        """
        创建一个神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.log_path = log_path
        self.layer_num = len(size)
        self.size = size
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.lambda_ = lambda_

    def define_ANN(self):
        """
        定义神经网络的结构
        """
        # self.input是训练数据里自变量
        prev_size = self.input.shape[1].value
        prev_out = self.input
        # self.size是神经网络的结构，也就是每一层的神经元个数
        size = self.size
        layer = 1
        self.W = []
        # 定义隐藏层
        for current_size in size[:-1]:
            weights = tf.Variable(
                tf.truncated_normal(
                    [prev_size, current_size], stddev=1.0 / np.sqrt(float(prev_size))),
                name="hidden%s_weights" % layer)
            # 将模型中的权重项记录下来，用于之后的惩罚项
            self.W.append(weights)
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            biases = tf.Variable(tf.zeros([current_size]), name="hidden%s_biases" % layer)
            layer += 1
            # 定义这一层神经元的输出
            neural_out = tf.nn.relu(tf.matmul(prev_out, weights) + biases)
            # 对隐藏层里的神经元使用dropout
            prev_out = tf.nn.dropout(neural_out, self.keep_prob)
            prev_size = current_size
        # 定义输出层
        weights = tf.Variable(
            tf.truncated_normal(
                [prev_size, size[-1]], stddev=1.0 / np.sqrt(float(prev_size))),
            name="output_weights")
        biases = tf.Variable(tf.zeros([size[-1]]), name="output_biases")
        self.out = tf.matmul(prev_out, weights) + biases
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        return self

    def define_loss(self):
        """
        定义神经网络的损失函数
        """
        # 定义单点损失，self.label是训练数据里的标签变量
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.out)
        loss = tf.reduce_mean(loss)
        # L2惩罚项
        _norm = map(lambda x: tf.nn.l2_loss(x), self.W)
        regularization = reduce(lambda a, b: a + b, _norm)
        # 定义整体损失
        self.loss = tf.reduce_mean(loss + self.lambda_ * regularization, name="loss")
        # 记录训练的细节
        tf.summary.scalar("loss", self.loss)
        return self

    def _do_eval(self, X, Y):
        """
        计算预测模型结果的准确率
        """
        prob = self.predict_proba(X)
        accuracy = float(np.sum(np.argmax(prob, 1) == np.argmax(Y, 1))) / prob.shape[0]
        return accuracy

    def evaluation(self, epoch):
        """
        输出模型的评估结果
        """
        print("epoch %s" % epoch)
        # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
        print("训练集的准确率 %.3f" % self._do_eval(self.train_set["X"], self.train_set["Y"]))
        print("验证集的准确率 %.3f" %
              self._do_eval(self.validation_set["X"], self.validation_set["Y"]))
        print("测试集的准确率 %.3f" % self._do_eval(self.test_set["X"], self.test_set["Y"]))

    def SGD(self, X, Y, start_learning_rate, mini_batch_fraction, epoch, keep_prob):
        """
        使用随机梯度下降法训练模型

        参数
        ----
        X : np.array, 自变量

        Y : np.array, 因变量
        """
        summary = tf.summary.merge_all()
        train_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(start_learning_rate, train_step,
                                                   1000, 0.96, staircase=True)
        method = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = method.minimize(self.loss, global_step=train_step)
        batch_size = int(X.shape[0] * mini_batch_fraction)
        batch_num = int(np.ceil(1 / mini_batch_fraction))
        sess = tf.Session()
        self.sess = sess
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        step = 0
        while (step < epoch):
            for i in range(batch_num):
                batch_X = X[i * batch_size: (i + 1) * batch_size]
                batch_Y = Y[i * batch_size: (i + 1) * batch_size]
                sess.run([optimizer],
                         feed_dict={self.input: batch_X,
                                    self.label: batch_Y, self.keep_prob: keep_prob})
            step += 1
            # 评估模型效果，并将日志写入文件
            self.evaluation(step)
            summary_str = sess.run(summary,
                                   feed_dict={self.input: X, self.label: Y,
                                              self.keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        return self

    def fit(self, start_learning_rate=0.1, mini_batch_fraction=0.01, epoch=200, keep_prob=0.7):
        """
        训练模型
        """
        X = self.train_set["X"]
        Y = self.train_set["Y"]
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.int64, shape=[None, self.size[-1]], name="Y")
        self.keep_prob = tf.placeholder(tf.float32)
        self.define_ANN()
        self.define_loss()
        self.SGD(X, Y, start_learning_rate, mini_batch_fraction, epoch, keep_prob)

    def predict_proba(self, X):
        """
        使用神经网络对未知数据进行预测
        """
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name="pred")
        prob = sess.run(pred, feed_dict={self.input: X, self.keep_prob: 1.0})
        return prob


if __name__ == "__main__":
    data = load_data()
    train_data, validation_data, train_label, validation_label = train_test_split(
        data[0], data[1], test_size=0.3, random_state=1001)
    train_set = {"X": train_data, "Y": train_label}
    validation_set = {"X": validation_data, "Y": validation_label}
    test_set = {"X": data[2], "Y": data[3]}
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        ann = ANN([30, 20, 10], "logs\\mnist", train_set, validation_set, test_set)
    else:
        ann = ANN([30, 20, 10], "logs/mnist", train_set, validation_set, test_set)
    ann.fit()
