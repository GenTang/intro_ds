# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何利用tensorflow来实现CNN，
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


class CNN(object):

    def __init__(self, logPath, trainSet, validationSet, testSet, lambda_=1e-4):
        """
        创建一个卷积神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.trainSet = trainSet
        self.validationSet = validationSet
        self.testSet = testSet
        self.lambda_ = lambda_
        self.W = []

    def defineCNN(self):
        """
        定义卷积神经网络的结构
        """
        # MNIST图片集是一个784的行向量，将其转换为28 x 28的方阵
        # 由于图片是黑白的，所以channel（最后一个值）等于1，如果是RGB彩色图片，channel应等于3
        img = tf.reshape(self.input, [-1, 28, 28, 1])
        # 定义卷积层1和池化层1，其中卷积层1里有20个feature map
        # convPool1的形状为[-1, 12, 12, 20]
        convPool1 = self.defineConvPool(img, filterShape=[5, 5, 1, 20],
            poolSize=[1, 2, 2, 1])
        # 定义卷积层2和池化层2，其中卷积层2里有40个feature map
        # convPool2的形状为[-1, 4, 4, 40]
        convPool2 = self.defineConvPool(convPool1, filterShape=[5, 5, 20, 40],
            poolSize=[1, 2, 2, 1])
        # 将池化层2的输出变成行向量，后者将作为全连接层的输入
        convPool2 = tf.reshape(convPool2, [-1, 40 * 4 * 4])
        # 定义全连接层
        self.out = self.defineFullConnected(convPool2, size=[30, 10])

    def defineConvPool(self, inputLayer, filterShape, poolSize):
        """
        定义卷积层和池化层
        """
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1))
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        biases = tf.Variable(tf.zeros(filterShape[-1]))
        # 定义卷积层
        _conv2d = tf.nn.conv2d(inputLayer, weights, strides=[1, 1, 1, 1], padding="VALID")
        convOut = tf.nn.relu(_conv2d + biases)
        # 定义池化层
        poolOut = tf.nn.max_pool(convOut, ksize=poolSize, strides=poolSize, padding="VALID")
        return poolOut

    def defineFullConnected(self, inputLayer, size):
        """
        定义全连接层的结构
        """
        prevSize = inputLayer.shape[1].value
        prevOut = inputLayer
        layer = 1
        # 定义隐藏层
        for currentSize in size[:-1]:
            weights = tf.Variable(tf.truncated_normal(
                [prevSize, currentSize], stddev=1.0 / np.sqrt(float(prevSize))),
                name="fc%s_weights" % layer)
            # 将模型中的权重项记录下来，用于之后的惩罚项
            self.W.append(weights)
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            biases = tf.Variable(tf.zeros([currentSize]),
                name="fc%s_biases" % layer)
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
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        biases = tf.Variable(tf.zeros([size[-1]]), name="output_biases")
        out = tf.matmul(prevOut, weights) + biases
        return out

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

    def fit(self, startLearningRate=0.1, miniBatchFraction=0.01, epoch=200, keepProb=0.5):
        """
        训练模型
        """
        X = self.trainSet["X"]
        Y = self.trainSet["Y"]
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.int64, shape=[None, Y.shape[1]], name="Y")
        self.keepProb = tf.placeholder(tf.float32)
        self.defineCNN()
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
        ann = CNN("logs\\mnist_cnn", trainSet, validationSet, testSet)
    else:
        ann = CNN("logs/mnist_cnn", trainSet, validationSet, testSet)
    ann.fit()
