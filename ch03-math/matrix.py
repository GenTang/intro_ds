# -*- coding: UTF-8 -*-
"""
此脚本用于展示矩阵运算
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import numpy as np
from numpy.linalg import inv


# 创建矩阵
A = np.matrix([[1, 2], [3, 4], [5, 6]])
print(A)
B = np.array(range(1, 7)).reshape(3, 2)
print(B)
try:
    A * A
except ValueError as e:
    print(repr(e))
print(B * B)
# 创建特殊矩阵
print(np.zeros((3, 2)))
print(np.identity(3))
print(np.diag([1, 2, 3]))
# 矩阵中向量的提取
m = np.array(range(1, 10)).reshape(3, 3)
print(m)
# 提取行向量
print(m[[0, 2]])
print(m[[True, False, True]])
# 提取列向量
print(m[:, [1, 2]])
print(m[:, [False, True, True]])
# 矩阵的计算
n = np.array(range(1, 5)).reshape(2, 2)
print(n)
print(np.transpose(n))
print(n + n)
print(n - n)
print(3 * n)
# Hadamard乘积
print(n * n)
# 矩阵乘法
print(n.dot(n))
# 矩阵的逆矩阵
print(inv(n))
print(np.dot(inv(n), n))
