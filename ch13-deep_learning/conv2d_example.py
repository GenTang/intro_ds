# -*- coding: UTF-8 -*-
"""
此脚本用于展示卷积层(conv2d)的计算过程
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import tensorflow as tf


def run():
    """
    是一个简单的例子来展示卷积层对多个input channel的处理过程
    """
    # 对应输入
    image = tf.constant([
        [1, 2],
        [0, 2],
        [1, 3],
        [2, 4]], dtype=tf.float32)
    # 输入的数据有两个input channel，每个channel为2x2的矩阵
    image = tf.reshape(image, [1, 2, 2, 2], name="image")
    # 相应地定义shared weights
    kernel = tf.constant([
        [1, -2],
        [2, 0],
        [-3, 4],
        [5, 3]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [2, 2, 2, 1], name="kernel")
    re = tf.nn.conv2d(image, kernel, strides=[1, 1, 1, 1], padding='VALID')
    with tf.Session() as sess:
        print("输入数据：", sess.run(image))
        print("shared weights：", sess.run(kernel))
        print("结果：", sess.run(re))


if __name__ == "__main__":
    run()
