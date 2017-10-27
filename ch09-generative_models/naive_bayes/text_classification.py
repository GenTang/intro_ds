# -*- coding: UTF-8 -*-
"""
此脚本用于展示如何使用朴素贝叶斯进行文本分类
所用语料来源于：复旦大学计算机信息与技术系国际数据库中心自然语言处理小组
由复旦大学李荣陆提供
"""


from os import listdir, path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


def readData(dataPath, category, testRatio):
    """
    根据跟定的类别，读取数据，并将数据分为训练集和测试集
    """
    np.random.seed(2046)
    trainData = []
    testData = []
    labels = [i for i in listdir(dataPath) if i in category]
    for i in labels:
        for j in listdir("%s/%s" % (dataPath, i)):
            content = readContent("%s/%s/%s" % (dataPath, i, j))
            if np.random.random() <= testRatio:
                testData.append({"label": i, "content": content})
            else:
                trainData.append({"label": i, "content": content})
    trainData = pd.DataFrame(trainData)
    testData = pd.DataFrame(testData)
    return trainData, testData


def readContent(dataPath):
    """
    读取文件里的内容，并略去不能正确解码的行
    """
    with open(dataPath, "rb") as f:
        rawContent = f.read()
    # 语料库使用GBK编码，对于不能编码的问题，选择略过
    content = ""
    for i in rawContent.split("\n"):
        try:
            content += i.decode("GBK")
        except UnicodeDecodeError:
            pass
    return content


def trainMultinomialNB(data):
    """
    使用多项式模型对数据进行建模
    """
    pipe = Pipeline([("vect", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ("model", MultinomialNB())])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    pipe.fit(data["content"], Y)
    return le, pipe


def trainMultinomialNBWithTFIDF(data):
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


def trainBernoulliNB(data):
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


def printResult(doc, pred):
    """
    输出样例的预测结果
    """
    for d, p in zip(doc, pred):
        print "%s ==> %s" % (d.replace(" ", ""), p)


def trainModel(trainData, testData, testDocs, docs):
    """
    对分词后的文本数据分别使用多项式和伯努利模型进行分类
    """
    # 伯努利模型
    vect, le, model = trainBernoulliNB(trainData)
    pred = le.classes_[model.predict(vect.transform(testDocs))]
    print "Use Bernoulli naive Bayes: "
    printResult(docs, pred)
    print classification_report(
        le.transform(testData["label"]),
        model.predict(vect.transform(testData["content"])),
        target_names=le.classes_)
    # 多项式模型
    le, pipe = trainMultinomialNB(trainData)
    pred = le.classes_[pipe.predict(testDocs)]
    print "Use multinomial naive Bayes: "
    printResult(docs, pred)
    print classification_report(
        le.transform(testData["label"]),
        pipe.predict(testData["content"]),
        target_names=le.classes_)
    # TFIDF+多项式模型
    le, pipe = trainMultinomialNBWithTFIDF(trainData)
    pred = le.classes_[pipe.predict(testDocs)]
    print "Use TFIDF + multinomial naive Bayes: "
    printResult(docs, pred)
    print classification_report(
        le.transform(testData["label"]),
        pipe.predict(testData["content"]),
        target_names=le.classes_)


def textClassifier(dataPath, category):
    """
    不进行中文分词，对文本进行分类
    """
    trainData, testData = readData(dataPath, category, 0.3)
    trainData["content"] = trainData.apply(lambda x: " ".join(x["content"]), axis=1)
    testData["content"] = testData.apply(lambda x: " ".join(x["content"]), axis=1)
    _docs = ["前国际米兰巨星雷科巴正式告别足坛","达芬奇：伟大的艺术家"]
    testDocs = [" ".join(i.decode("utf-8")) for i in _docs]
    trainModel(trainData, testData, testDocs, _docs)


def textClassifierWithJieba(dataPath, category):
    """
    使用第三方库jieba对文本进行分词，然后再进行分类
    """
    trainData, testData = readData(dataPath, category, 0.3)
    trainData["content"] = trainData.apply(
        lambda x: " ".join(jieba.cut(x["content"], cut_all=True)), axis=1)
    testData["content"] = testData.apply(
        lambda x: " ".join(jieba.cut(x["content"], cut_all=True)), axis=1)
    _docs = ["前国际米兰巨星雷科巴正式告别足坛", "达芬奇：伟大的艺术家"]
    testDocs = [" ".join(jieba.cut(i, cut_all=True)) for i in _docs]
    trainModel(trainData, testData, testDocs, _docs)


if __name__ == "__main__":
    dataPath = "%s/data" % path.dirname(path.abspath(__file__))
    category = ["C3-Art", "C11-Space", "C19-Computer", "C39-Sports"]
    if len(sys.argv) == 1:
        textClassifier(dataPath, category)
    elif (len(sys.argv) == 2) & (sys.argv[1] == "use_jieba"):
        import jieba
        textClassifierWithJieba(dataPath, category)
    else:
        print >> sys.stderr, """
        Usage: python naive_bayes.py | python naive_bayes.py use_jieba
        """
