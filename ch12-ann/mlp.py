# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用tensorflow来实现MLP
"""


import numpy as np
import tensorflow as tf


class ANN(object):
    
    def __init__(self, size, logPath):
        """
        创建一个神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.layerNum = len(size)
        self.size = size

    def defineANN(self):
        """
        定义神经网络的结构
        """
        # self.input是训练数据里自变量
        prevSize = self.input.shape[1].value
        prevOut = self.input
        # self.size是神经网络的结构，也就是每一层的神经元个数
        size = self.size
        layer = 1
        # 定义隐藏层
        for currentSize in size[:-1]:
            weights = tf.Variable(
                tf.truncated_normal([prevSize, currentSize],
                    stddev=1.0 / np.sqrt(float(prevSize))))
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            layer += 1
            biases = tf.Variable(tf.zeros([currentSize]))
            prevOut = tf.nn.sigmoid(tf.matmul(prevOut, weights) + biases)
            prevSize = currentSize
        # 定义输出层
        weights = tf.Variable(
            tf.truncated_normal([prevSize, size[-1]],
                stddev=1.0 / np.sqrt(float(prevSize))))
        biases = tf.Variable(tf.zeros([size[-1]]))
        self.out = tf.matmul(prevOut, weights) + biases
        return self

    def defineLoss(self):
        """
        定义神经网络的损失函数
        """
        # 定义单点损失，self.label是训练数据里的标签变量
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.out, name="loss")
        # 定义整体损失
        self.loss = tf.reduce_mean(loss, name="average_loss")
        return self

    def SGD(self, X, Y, learningRate, miniBatchFraction, epoch):
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
        method = tf.train.GradientDescentOptimizer(learningRate)
        optimizer= method.minimize(self.loss)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1 / miniBatchFraction))
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.logPath, graph=tf.get_default_graph())
        step = 0
        while (step < epoch):
            for i in range(batchNum):
                batchX = X[i * batchSize: (i + 1) * batchSize]
                batchY = Y[i * batchSize: (i + 1) * batchSize]
                sess.run([optimizer],
                    feed_dict={self.input: batchX, self.label: batchY})
            step += 1
            # 将日志写入文件
            summary_str = sess.run(summary, feed_dict={self.input: X, self.label: Y})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        self.sess = sess
        return self

    def fit(self, X, Y, learningRate=0.3, miniBatchFraction=0.1, epoch=2500):
        """
        训练模型

        参数
        ----
        X : np.array, 自变量
        
        Y : np.array, 因变量
        """
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.int64, shape=[None, self.size[-1]], name="Y")
        self.defineANN()
        self.defineLoss()
        self.SGD(X, Y, learningRate, miniBatchFraction, epoch)

    def predict_proba(self, X):
        """
        使用神经网络对未知数据进行预测
        """
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name="pred")
        prob = sess.run(pred, feed_dict={self.input: X})
        return prob
