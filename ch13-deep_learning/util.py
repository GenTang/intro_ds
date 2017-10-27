# -*- coding: UTF-8 -*-
"""
MNIST图片数据的下载地址是：http://yann.lecun.com/exdb/mnist/
此脚本用于定义读取MNIST图片数据的函数。
此脚步参考自
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""


import gzip
import os
import numpy as np
import matplotlib.pyplot as plt


def _read32(bytestream):
    """
    从文件中读取32 bit
    """
    dt = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def readImages(f):
    """
    读取图片文件，并将每个图片转换为行向量
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError("Invalid magic number %d in MNIST image file: %s" %
                (magic, f.name))
        numImages = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * numImages)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(numImages, rows * cols).astype(np.float32) / 255.0
        return data


def readLabels(f):
    """
    读取图片的标签
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError("Invalid magic number %d in MNIST label file: %s" %
                (magic, f.name))
        numItems = _read32(bytestream)
        buf = bytestream.read(numItems)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def OneHotEncoder(labels, classNum=10):
    """
    """
    _cond = np.array([range(classNum), ] * labels.shape[0])
    cond = _cond == labels.reshape(-1, 1)
    oneHot = np.zeros((labels.shape[0], classNum))
    oneHot[cond] = 1
    return oneHot


def loadData():
    """
    读取MNIST图片数据
    """
    homePath = "%s/data" % os.path.dirname(os.path.abspath(__file__))
    trainImgFile = "train-images-idx3-ubyte.gz"
    trainLabelFile = "train-labels-idx1-ubyte.gz"
    testImgFile = "t10k-images-idx3-ubyte.gz"
    testLabelFile = "t10k-labels-idx1-ubyte.gz"
    with open("%s/%s" % (homePath, trainImgFile), "rb") as f:
        trainImg = readImages(f)
    with open("%s/%s" % (homePath, trainLabelFile), "rb") as f:
        trainLabel = OneHotEncoder(readLabels(f))
    with open("%s/%s" % (homePath, testImgFile), "rb") as f:
        testImg = readImages(f)
    with open("%s/%s" % (homePath, testLabelFile), "rb") as f:
        testLabel = OneHotEncoder(readLabels(f))
    return trainImg, trainLabel, testImg, testLabel    


def visualize(img):
    """
    将数字格式的图片可视化
    """
    imgPerRow = 8
    fig = plt.figure(figsize=(10, 10), dpi=80)
    for i in range(imgPerRow * imgPerRow):
        ax = fig.add_subplot(imgPerRow, imgPerRow, i+1)
        ax.imshow(img[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    data = loadData()
    visualize(data[0])
