# -*- coding: UTF-8 -*-
"""
此脚本用于测试getFrequentItem
"""


# 保证脚本与Python3兼容
from __future__ import print_function

from os import sys, path


if __name__ == "__main__":
    # 得到mini_project所在的绝对路径
    packagePath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    # 将mini_project所在路径，加到系统路径里。这样就可以将mini_project作为库使用了
    sys.path.append(packagePath)
    from mini_project.components.selecter import getFrequentItem
    data = ["a", "a", "b", 1, 2, 2]
    print(getFrequentItem(data))
