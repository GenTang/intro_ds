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

from util import loadData

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ANN(object):
    
    def __init__(self, size, logPath, trainSet, validationSet, testSet, lambda_=1e-3):
        """
        创建一个神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.layerNum = len(size)
        self.size = size
        self.trainSet = trainSet
        self.validationSet = validationSet
        self.testSet = testSet
        self.lambda_ = lambda_

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
        self.W = []
        # 定义隐藏层
        for currentSize in size[:-1]:
            weights = tf.Variable(tf.truncated_normal(
                [prevSize, currentSize], stddev=1.0 / np.sqrt(float(prevSize))),
                name="hidden%s_weights" % layer)
            # 将模型中的权重项记录下来，用于之后的惩罚项
            self.W.append(weights)
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            biases = tf.Variable(tf.zeros([currentSize]),
                name="hidden%s_biases" % layer)
            layer += 1
            # 定义这一层神经元的输出
            neuralOut = tf.nn.relu(tf.matmul(prevOut, weights) + biases)
            # 对隐藏层里的神经元使用dropout
            prevOut = tf.nn.dropout(neuralOut, self.keepProb)
            prevSize = currentSize
        # 定义输出层
        weights = tf.Variable(tf.truncated_normal(
            [prevSize, size[-1]], stddev=1.0 / np.sqrt(float(prevSize))),
            name="output_weights")
        biases = tf.Variable(tf.zeros([size[-1]]), name="output_biases")
        self.out = tf.matmul(prevOut, weights) + biases
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        return self

    def defineLoss(self):
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
        self.loss = tf.reduce_mean(loss + self.lambda_ * regularization,
            name="loss")
        # 记录训练的细节
        tf.summary.scalar("loss", self.loss)
        return self

    def _doEval(self, X, Y):
        """
        计算预测模型结果的准确率
        """
        prob = self.predict_proba(X)
        accurary = float(np.sum(np.argmax(prob, 1) == np.argmax(Y, 1))) / prob.shape[0]
        return accurary

    def evaluation(self, epoch):
        """
        输出模型的评估结果
        """
        print("epoch %s" % epoch)
        # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
        print("训练集的准确率 %.3f" % self._doEval(self.trainSet["X"], self.trainSet["Y"]))
        print("验证集的准确率 %.3f" % self._doEval(self.validationSet["X"],
            self.validationSet["Y"]))
        print("测试集的准确率 %.3f" % self._doEval(self.testSet["X"], self.testSet["Y"]))

    def SGD(self, X, Y, startLearningRate, miniBatchFraction, epoch, keepProb):
        """
        使用随机梯度下降法训练模型

        参数
        ----
        X : np.array, 自变量
        
        Y : np.array, 因变量
        """
        summary = tf.summary.merge_all()
        trainStep = tf.Variable(0)
        learningRate = tf.train.exponential_decay(startLearningRate, trainStep,
            1000, 0.96, staircase=True)
        method = tf.train.GradientDescentOptimizer(learningRate)
        optimizer= method.minimize(self.loss, global_step=trainStep)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1 / miniBatchFraction))
        sess = tf.Session()
        self.sess = sess
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.logPath, graph=tf.get_default_graph())
        step = 0
        while (step < epoch):
            for i in range(batchNum):
                batchX = X[i * batchSize: (i + 1) * batchSize]
                batchY = Y[i * batchSize: (i + 1) * batchSize]
                sess.run([optimizer], feed_dict={self.input: batchX,
                    self.label: batchY, self.keepProb: keepProb})
            step += 1
            # 评估模型效果，并将日志写入文件
            self.evaluation(step)
            summary_str = sess.run(summary, feed_dict={self.input: X,
                self.label: Y, self.keepProb: 1.0})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        return self

    def fit(self, startLearningRate=0.1, miniBatchFraction=0.01, epoch=200, keepProb=0.7):
        """
        训练模型
        """
        X = self.trainSet["X"]
        Y = self.trainSet["Y"]
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.int64, shape=[None, self.size[-1]], name="Y")
        self.keepProb = tf.placeholder(tf.float32)
        self.defineANN()
        self.defineLoss()
        self.SGD(X, Y, startLearningRate, miniBatchFraction, epoch, keepProb)

    def predict_proba(self, X):
        """
        使用神经网络对未知数据进行预测
        """
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name="pred")
        prob = sess.run(pred, feed_dict={self.input: X, self.keepProb: 1.0})
        return prob


if __name__ == "__main__":
    data = loadData()
    trainData, validationData, trainLabel, validationLabel = train_test_split(
        data[0], data[1], test_size=0.3, random_state=1001)
    trainSet = {"X": trainData, "Y": trainLabel}
    validationSet = {"X": validationData, "Y": validationLabel}
    testSet = {"X": data[2], "Y": data[3]}
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        ann = ANN([30, 20, 10], "logs\\mnist", trainSet, validationSet, testSet)
    else:
        ann = ANN([30, 20, 10], "logs/mnist", trainSet, validationSet, testSet)
    ann.fit()
