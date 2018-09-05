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


def read_data(data_path, test_ratio):
    """
    读取数据，根据比例将数据分为训练集和测试集
    """
    # 在Python3中，读取文件时就会decode
    if sys.version_info[0] == 3:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()
            raw_content = raw_content.rstrip("\n").split("\n")
            # 数据中有一些特殊格式需要处理
            content = [re.sub(r"\s\[|\]\s|^\d+", "", i).split()
                       for i in rawContent]
    else:
        with open(data_path, "r") as f:
            raw_content = f.read()
            raw_content = raw_content.rstrip("\n").split("\n")
            # 数据中有一些特殊格式需要处理
            content = [re.sub(r"\s\[|\]\s|^\d+", "", i).decode("utf-8").split()
                       for i in raw_content]
    return train_test_split(content, test_size=test_ratio, random_state=2017)


def extract_feature(data):
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


def train_model(X, y, lengths):
    """
    训练模型
    """
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    XX = vect.fit_transform(X)
    model = MultinomialHMM(alpha=0.01)
    model.fit(XX, y, lengths)
    return vect, model


def print_segmentation(X, y):
    """
    输出分词结果，根据分词标签，在词之前加入空格和斜线
    """
    assert len(X) == len(y), "The size of document must equal to the size of label"
    xx = list(X)
    for i in range(len(y)):
        if y[i].startswith("S") | y[i].startswith("B"):
            xx[i] = " / " + xx[i]
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("".join(xx))


def chinese_segmentation(data_path):
    """
    对中文分词，并评估模型效果
    """
    train_set, test_set = read_data(data_path, 0.2)
    train_X, train_Y, train_lengths = extract_feature(train_set)
    test_X, test_Y, test_lengths = extract_feature(test_set)
    vect, model = train_model(train_X, train_Y, train_lengths)
    pred = model.predict(vect.transform(test_X), test_lengths)
    print(classification_report([i[0] for i in test_Y], [i[0] for i in pred]))
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        exampl_str = list("我爱北京天安门")
    else:
        exampl_str = list("我爱北京天安门".decode("utf-8"))
    print_segmentation(exampl_str, model.predict(vect.transform(exampl_str)))


if __name__ == "__main__":
    # 为了计算速度，只使用部分数据
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data\\corpus_不_20140804161122.txt" % \
            path.dirname(path.abspath(__file__))
    else:
        data_path = "%s/data/corpus_不_20140804161122.txt" % \
            path.dirname(path.abspath(__file__))
    chinese_segmentation(data_path)
