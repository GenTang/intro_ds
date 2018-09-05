# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用朴素贝叶斯进行文本分类
所用语料来源于：复旦大学计算机信息与技术系国际数据库中心自然语言处理小组
由复旦大学李荣陆提供
"""


# 保证脚本与Python3兼容
from __future__ import print_function

from os import listdir, path
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def read_data(data_path, category, test_ratio):
    """
    根据跟定的类别，读取数据，并将数据分为训练集和测试集
    """
    np.random.seed(2046)
    train_data = []
    test_data = []
    labels = [i for i in listdir(data_path) if i in category]
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        for i in labels:
            for j in listdir("%s\\%s" % (data_path, i)):
                content = read_content("%s\\%s\\%s" % (data_path, i, j))
                if np.random.random() <= test_ratio:
                    test_data.append({"label": i, "content": content})
                else:
                    train_data.append({"label": i, "content": content})
    else:
        for i in labels:
            for j in listdir("%s/%s" % (data_path, i)):
                content = read_content("%s/%s/%s" % (data_path, i, j))
                if np.random.random() <= test_ratio:
                    test_data.append({"label": i, "content": content})
                else:
                    train_data.append({"label": i, "content": content})
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    return train_data, test_data


def read_content(data_path):
    """
    读取文件里的内容，并略去不能正确解码的行
    """
    # 在Python3中，读取文件时就会decode
    if sys.version_info[0] == 3:
        with open(data_path, "r", errors="ignore") as f:
            raw_content = f.read()
    else:
        with open(data_path, "r") as f:
            raw_content = f.read()
    # 语料库使用GBK编码，对于不能编码的问题，选择略过
    content = ""
    for i in raw_content.split("\n"):
        try:
            # 在Python3中，str不需要decode
            if sys.version_info[0] == 3:
                content += i
            else:
                content += i.decode("GBK")
        except UnicodeDecodeError:
            pass
    return content


def train_multinomialNB(data):
    """
    使用多项式模型对数据进行建模
    """
    pipe = Pipeline([("vect", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
                     ("model", MultinomialNB())])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    pipe.fit(data["content"], Y)
    return le, pipe


def train_multinomialNB_with_TFIDF(data):
    """
    使用TFIDF+多项式模型对数据建模
    """
    pipe = Pipeline([("vect", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
                     ("tfidf", TfidfTransformer(norm=None, sublinear_tf=True)),
                     ("model", MultinomialNB())])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    pipe.fit(data["content"], Y)
    return le, pipe


def train_bernoulliNB(data):
    """
    使用伯努利模型对数据建模
    """
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b", binary=True)
    X = vect.fit_transform(data["content"])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    model = BernoulliNB()
    model.fit(X, Y)
    return vect, le, model


def print_result(doc, pred):
    """
    输出样例的预测结果
    """
    for d, p in zip(doc, pred):
        # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
        print("%s ==> %s" % (d.replace(" ", ""), p))


def train_model(train_data, test_data, test_docs, docs):
    """
    对分词后的文本数据分别使用多项式和伯努利模型进行分类
    """
    # 伯努利模型
    vect, le, model = train_bernoulliNB(train_data)
    pred = le.classes_[model.predict(vect.transform(test_docs))]
    print("Use Bernoulli naive Bayes: ")
    print_result(docs, pred)
    print(classification_report(
        le.transform(test_data["label"]),
        model.predict(vect.transform(test_data["content"])),
        target_names=le.classes_))
    # 多项式模型
    le, pipe = train_multinomialNB(train_data)
    pred = le.classes_[pipe.predict(test_docs)]
    print("Use multinomial naive Bayes: ")
    print_result(docs, pred)
    print(classification_report(
        le.transform(test_data["label"]),
        pipe.predict(test_data["content"]),
        target_names=le.classes_))
    # TFIDF+多项式模型
    le, pipe = train_multinomialNB_with_TFIDF(train_data)
    pred = le.classes_[pipe.predict(test_docs)]
    print("Use TFIDF + multinomial naive Bayes: ")
    print_result(docs, pred)
    print(classification_report(
        le.transform(test_data["label"]),
        pipe.predict(test_data["content"]),
        target_names=le.classes_))


def text_classifier(data_path, category):
    """
    不进行中文分词，对文本进行分类
    """
    train_data, test_data = read_data(data_path, category, 0.3)
    train_data["content"] = train_data.apply(lambda x: " ".join(x["content"]), axis=1)
    test_data["content"] = test_data.apply(lambda x: " ".join(x["content"]), axis=1)
    _docs = ["前国际米兰巨星雷科巴正式告别足坛", "达芬奇：伟大的艺术家"]
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        test_docs = [" ".join(i) for i in _docs]
    else:
        test_docs = [" ".join(i.decode("utf-8")) for i in _docs]
    train_model(train_data, test_data, test_docs, _docs)


def text_classifier_with_jieba(data_path, category):
    """
    使用第三方库jieba对文本进行分词，然后再进行分类
    """
    train_data, test_data = read_data(data_path, category, 0.3)
    train_data["content"] = train_data.apply(
        lambda x: " ".join(jieba.cut(x["content"], cut_all=True)), axis=1)
    test_data["content"] = test_data.apply(
        lambda x: " ".join(jieba.cut(x["content"], cut_all=True)), axis=1)
    _docs = ["前国际米兰巨星雷科巴正式告别足坛", "达芬奇：伟大的艺术家"]
    test_docs = [" ".join(jieba.cut(i, cut_all=True)) for i in _docs]
    train_model(train_data, test_data, test_docs, _docs)


if __name__ == "__main__":
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\data" % path.dirname(path.abspath(__file__))
    else:
        data_path = "%s/data" % path.dirname(path.abspath(__file__))
    category = ["C3-Art", "C11-Space", "C19-Computer", "C39-Sports"]
    if len(sys.argv) == 1:
        text_classifier(data_path, category)
    elif (len(sys.argv) == 2) & (sys.argv[1] == "use_jieba"):
        import jieba
        text_classifier_with_jieba(data_path, category)
    else:
        print(
            """
            Usage: python naive_bayes.py | python naive_bayes.py use_jieba
            """,
            file=sys.stderr)
