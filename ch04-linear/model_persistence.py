# -*- coding: UTF-8 -*-
"""
以线性回归模型为例子，此脚本用于展示如何保存和读取模型
"""


# 保证脚本与Python3兼容
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml


def read_data(path):
    """
    使用pandas读取数据
    """
    data = pd.read_csv(path)
    return data


def save_as_PMML(data, modelPath):
    """
    利用sklearn2pmml将模型存储为PMML
    """
    model = PMMLPipeline([("regressor", linear_model.LinearRegression())])
    model.fit(data[["x"]], data["y"])
    sklearn2pmml(model, "linear.pmml", with_repr=True)


def train_and_save_model(data, modelPath):
    """
    使用pickle保存训练好的模型
    """
    model = linear_model.LinearRegression()
    model.fit(data[["x"]], data[["y"]])
    pickle.dump(model, open(modelPath, "wb"))
    return model


def load_model(modelPath):
    """
    使用pickle读取已有模型
    """
    model = pickle.load(open(modelPath, "rb"))
    return model


if __name__ == "__main__":
    home_path = os.path.dirname(os.path.abspath(__file__))
    # Windows下的存储路径与Linux并不相同
    if os.name == "nt":
        data_path = "%s\\simple_example\\data\\simple_example.csv" % home_path
    else:
        data_path = "%s/simple_example/data/simple_example.csv" % home_path
    data = read_data(data_path)
    model_path = "linear_model"
    original_model = train_and_save_model(data, model_path)
    model = load_model(model_path)
    print("保存的模型对1的预测值：%s" % original_model.predict([[1]]))
    print("读取的模型对1的预测值：%s" % model.predict([[1]]))
    save_as_PMML(data, "linear.pmml")
