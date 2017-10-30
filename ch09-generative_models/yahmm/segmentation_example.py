# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用HMM模型对中文进行分词
数据来源于https://github.com/liwenzhu/corpusZh
"""


# 保证脚本与Python3兼容
from __future__ import print_function

from os import path
import os
os.sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import sys

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hmm.multinomialHMM import MultinomialHMM


def readData(dataPath, testRatio):
    """
    读取数据，根据比例将数据分为训练集和测试集
    """
    with open(dataPath, "rb") as f:
        rawContent = f.read()
    rawContent = rawContent.rstrip("\n").split("\n")
    # 数据中有一些特殊格式需要处理
    content = [re.sub(r"\s\[|\]\s|^\d+", "", i).decode("utf-8").split() for i in rawContent]
    return train_test_split(content, test_size=testRatio, random_state=2017)


def extractFeature(data):
    """
    根据词的长度，给词打上分词的标签
    """
    lengths = []
    X = []
    y = []
    for sentence in data:
        length = 0
        _y = []
        _x = []
        for word in sentence:
            # 将不符规范的文字排除掉
            _word = word.split("/")
            if (len(_word) == 2) & (len(_word[0]) > 0):
                word, pos = word.split("/")
                length += len(word)
                _x += list(word)
                if len(word) == 1:
                    _y.append("+".join(["S", pos]))
                else:
                    _y += ["+".join(["B", pos])] + ["+".join(["M", pos])] * (len(word) - 2)\
                        + ["+".join(["E", pos])]
            else:
                pass
        lengths.append(length)
        y += _y
        X += _x
    return X, y, lengths


def trainModel(X, y, lengths):
    """
    训练模型
    """
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    XX = vect.fit_transform(X)
    model = MultinomialHMM(alpha=0.01)
    model.fit(XX, y, lengths)
    return vect, model


def printSegmentation(X, y):
    """
    输出分词结果，根据分词标签，在词之前加入空格和斜线
    """
    assert len(X) == len(y), "The size of document must equal to the size of label"
    xx = list(X)
    for i in range(len(y)):
        if y[i].startswith("S") | y[i].startswith("B"):
            xx[i] = " / " + xx[i]
    print("".join(xx))


def chineseSegmentation(dataPath):
    """
    对中文分词，并评估模型效果
    """
    trainSet, testSet = readData(dataPath, 0.2)
    trainX, trainY, trainLengths = extractFeature(trainSet)
    testX, testY, testLengths = extractFeature(testSet)
    vect, model = trainModel(trainX, trainY, trainLengths)
    pred = model.predict(vect.transform(testX), testLengths)
    print(classification_report([i[0] for i in testY], [i[0] for i in pred]))
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        examplStr = list("我爱北京天安门")
    else:
        examplStr = list("我爱北京天安门".decode("utf-8"))
    printSegmentation(examplStr, model.predict(vect.transform(examplStr)))


if __name__ == "__main__":
    # 为了计算速度，只使用部分数据
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        dataPath = "%s\\data\\corpus_不_20140804161122.txt" % \
            path.dirname(path.abspath(__file__))
    else:
        dataPath = "%s/data/corpus_不_20140804161122.txt" % \
            path.dirname(path.abspath(__file__))
    chineseSegmentation(dataPath)
