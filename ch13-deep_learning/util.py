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


def read_images(f):
    """
    读取图片文件，并将每个图片转换为行向量
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError("Invalid magic number %d in MNIST image file: %s"
                             % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols).astype(np.float32) / 255.0
        return data


def read_labels(f):
    """
    读取图片的标签
    """
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError("Invalid magic number %d in MNIST label file: %s"
                             % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def one_hot_encoder(labels, class_num=10):
    """
    """
    _cond = np.array([list(range(class_num)), ] * labels.shape[0])
    cond = _cond == labels.reshape(-1, 1)
    one_hot = np.zeros((labels.shape[0], class_num))
    one_hot[cond] = 1
    return one_hot


def load_data():
    """
    读取MNIST图片数据
    """
    # Windows下的存储路径与Linux并不相同
    train_img_file = "train-images-idx3-ubyte.gz"
    train_label_file = "train-labels-idx1-ubyte.gz"
    test_img_file = "t10k-images-idx3-ubyte.gz"
    test_label_file = "t10k-labels-idx1-ubyte.gz"
    if os.name == "nt":
        home_path = "%s\\data" % os.path.dirname(os.path.abspath(__file__))
        with open("%s\\%s" % (home_path, train_img_file), "rb") as f:
            train_img = read_images(f)
        with open("%s\\%s" % (home_path, train_label_file), "rb") as f:
            train_label = one_hot_encoder(read_labels(f))
        with open("%s\\%s" % (home_path, test_img_file), "rb") as f:
            test_img = read_images(f)
        with open("%s\\%s" % (home_path, test_label_file), "rb") as f:
            test_label = one_hot_encoder(read_labels(f))
    else:
        home_path = "%s/data" % os.path.dirname(os.path.abspath(__file__))
        with open("%s/%s" % (home_path, train_img_file), "rb") as f:
            train_img = read_images(f)
        with open("%s/%s" % (home_path, train_label_file), "rb") as f:
            train_label = one_hot_encoder(read_labels(f))
        with open("%s/%s" % (home_path, test_img_file), "rb") as f:
            test_img = read_images(f)
        with open("%s/%s" % (home_path, test_label_file), "rb") as f:
            test_label = one_hot_encoder(read_labels(f))
    return train_img, train_label, test_img, test_label


def visualize(img):
    """
    将数字格式的图片可视化
    """
    img_per_row = 8
    fig = plt.figure(figsize=(10, 10), dpi=80)
    for i in range(img_per_row * img_per_row):
        ax = fig.add_subplot(img_per_row, img_per_row, i+1)
        ax.imshow(img[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    data = load_data()
    visualize(data[0])
