# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用tensorflow来实现MLP
"""


import numpy as np
import tensorflow as tf


class ANN(object):

    def __init__(self, size, log_path):
        """
        创建一个神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.log_path = log_path
        self.layer_num = len(size)
        self.size = size

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
        # 定义隐藏层
        for current_size in size[:-1]:
            weights = tf.Variable(tf.truncated_normal(
                [prev_size, current_size],
                stddev=1.0 / np.sqrt(float(prev_size))))
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            layer += 1
            biases = tf.Variable(tf.zeros([current_size]))
            prev_out = tf.nn.sigmoid(tf.matmul(prev_out, weights) + biases)
            prev_size = current_size
        # 定义输出层
        weights = tf.Variable(tf.truncated_normal(
            [prev_size, size[-1]],
            stddev=1.0 / np.sqrt(float(prev_size))))
        biases = tf.Variable(tf.zeros([size[-1]]))
        self.out = tf.matmul(prev_out, weights) + biases
        return self

    def define_loss(self):
        """
        定义神经网络的损失函数
        """
        # 定义单点损失，self.label是训练数据里的标签变量
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.out, name="loss")
        # 定义整体损失
        self.loss = tf.reduce_mean(loss, name="average_loss")
        return self

    def SGD(self, X, Y, learning_rate, mini_batch_fraction, epoch):
        """
        使用随机梯度下降法训练模型

        参数
        ----
        X : np.array, 自变量

        Y : np.array, 因变量
        """
        # 记录训练的细节
        tf.summary.scalar("loss", self.loss)
        summary = tf.summary.merge_all()
        method = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = method.minimize(self.loss)
        batch_size = int(X.shape[0] * mini_batch_fraction)
        batch_num = int(np.ceil(1 / mini_batch_fraction))
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
        step = 0
        while (step < epoch):
            for i in range(batch_num):
                batch_X = X[i * batch_size: (i + 1) * batch_size]
                batch_Y = Y[i * batch_size: (i + 1) * batch_size]
                sess.run([optimizer], feed_dict={self.input: batch_X, self.label: batch_Y})
            step += 1
            # 将日志写入文件
            summary_str = sess.run(summary, feed_dict={self.input: X, self.label: Y})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        self.sess = sess
        return self

    def fit(self, X, Y, learning_rate=0.3, mini_batch_fraction=0.1, epoch=2500):
        """
        训练模型

        参数
        ----
        X : np.array, 自变量

        Y : np.array, 因变量
        """
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.int64, shape=[None, self.size[-1]], name="Y")
        self.define_ANN()
        self.define_loss()
        self.SGD(X, Y, learning_rate, mini_batch_fraction, epoch)

    def predict_proba(self, X):
        """
        使用神经网络对未知数据进行预测
        """
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name="pred")
        prob = sess.run(pred, feed_dict={self.input: X})
        return prob
