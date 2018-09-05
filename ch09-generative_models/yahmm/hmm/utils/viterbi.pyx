# -*- coding: UTF-8 -*-
"""
此脚本用于展示使用Cython实现viterbi算法
"""


cimport cython
cimport numpy as np
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi(obs,
            np.ndarray[ndim=1, dtype=np.float64_t] init_prob,
            np.ndarray[ndim=2, dtype=np.float64_t] trans_prob,
            np.ndarray[ndim=2, dtype=np.float64_t] emit_prob):
    """
    viterbi算法

    参数
    ----
    obs : {np.array 或者scipy.sparse.csr_matrix}，维度为(样本数，特征数)
          数据的特征矩阵

    init_prob : {np.array}，维度为(状态数)，表示各状态的初始分布

    trans_prob : {np.array}，维度为(状态数，状态数)，状态间的转移矩阵

    emit_prob : {np.array}，维度为(状态数，特征数)，各状态下，特征的条件概率
    
    返回
    ----
    score : {np.array}，维度为(样本数，状态数)，viterbi算法中间概率

    path : {np.array}，维度为(样本数)，最终结果，表示每个样本的隐藏状态
    """
    cdef np.ndarray[ndim=2, dtype=np.npy_intp, mode="c"] backp
    cdef np.ndarray[ndim=1, dtype=np.npy_intp, mode="c"] path
    cdef np.ndarray[ndim=2, dtype=np.float64_t, mode="c"] score
    cdef np.float64_t candidate, max_val
    cdef np.npy_intp i, j, k, sample_num, state_num, max_ind
        
    sample_num, state_num = obs.shape[0], init_prob.shape[0]
    backp = np.empty((sample_num, state_num), dtype=np.intp)
    score = safe_sparse_dot(obs, emit_prob.T)

    for i in range(state_num):
        score[0, i] += init_prob[i]

    for i in range(1, sample_num):
        for j in range(state_num):
            max_ind = 0
            max_val = -np.inf
            for k in range(state_num):
                candidate = score[i - 1, k] + trans_prob[k, j] + score[i, j]
                if candidate > max_val:
                    max_ind = k
                    max_val = candidate
            score[i, j] = max_val
            backp[i, j] = max_ind

    path = np.empty(sample_num, dtype=np.intp)
    path[sample_num - 1] = score[sample_num - 1, :].argmax()
    for i in range(sample_num - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]
    # Return score just for testing
    return score, path
