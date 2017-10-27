# -*- coding: UTF-8 -*-
"""
此脚本用于实现viterbi算法
"""


import numpy as np
from sklearn.utils.extmath import safe_sparse_dot


def viterbi(obs, initProb, transProb, emitProb):
    """
    viterbi算法
    
    参数
    ----
    obs : {np.array 或者scipy.sparse.csr_matrix}，维度为(样本数，特征数)，
        数据的特征矩阵

    initProb : {np.array或者scipy.sparse.csr_matrix}，维度为(状态数)，
        表示各状态的初始分布

    transProb : {np.array或者scipy.sparse.csr_matrix}，维度为(状态数，状态数)，
        状态间的转移矩阵

    emitProb : {np.array或者scipy.sparse.csr_matrix}，维度为(状态数，特征数)，
        各状态下，特征的条件概率
    
    返回
    ----
    score : {np.array}，维度为(样本数，状态数)，viterbi算法中间概率

    path : {np.array}，维度为(样本数)，最终结果表示每个样本的隐藏状态
    """
    sampleNum, stateNum = obs.shape[0], initProb.shape[0]
    backp = np.empty((sampleNum, stateNum), dtype=np.intp)
    score = safe_sparse_dot(obs, emitProb.T)

    for i in range(stateNum):
        score[0, i] += initProb[i]

    for i in range(1, sampleNum):
        for j in range(stateNum):
            maxInd = 0
            maxVal = -np.inf
            for k in range(stateNum):
                candidate = score[i - 1, k] + transProb[k, j] + score[i, j]
                if candidate > maxVal:
                    maxInd = k
                    maxVal = candidate
            score[i, j] = maxVal
            backp[i, j] = maxInd

    path = np.empty(sampleNum, dtype=np.intp)
    path[sampleNum - 1] = score[sampleNum - 1, :].argmax()
    for i in range(sampleNum - 2, -1, -1):
        path[i] = backp[i + 1, path[i + 1]]
    # Return score just for testing
    return score, path
