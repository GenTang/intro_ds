# -*- coding: UTF-8 -*-
"""
此脚本用于展示矩阵运算
"""
import numpy as np
from numpy.linalg import inv


# 创建矩阵
A = np.matrix([[1, 2], [3, 4], [5, 6]])
B = np.array(range(1, 7)).reshape(3, 2)
A * A
B * B
# 创建特殊矩阵
np.zeros((3, 2))
np.identity(3)
np.diag([1, 2, 3])
# 矩阵中向量的提取
m = np.array(range(1, 10)).reshape(3, 3)
# 提取行向量
m[[0, 2]]
m[[True, False, True]]
# 提取列向量
m[:, [1, 2]]
m[:, [False, True, True]]
# 矩阵的计算
n = np.array(range(1, 5)).reshape(2, 2)
np.transpose(n)
n + n
n - n
3 * n
## Hadamard乘积
n * n
## 矩阵乘法
n.dot(n)
## 矩阵的逆矩阵
inv(n)
np.dot(inv(n), n)

